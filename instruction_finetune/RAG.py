import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Dict, Tuple
import torch.nn.functional as F
from ins import (
    GPTModel, tokenizer, text_to_token_ids,
    token_ids_to_text, generate_text
)


class TextChunker:
    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting using regex
        sentences = re.split(r'[.]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[Tuple[str, Dict]]:
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())  # split using standard tokenizer

            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Store chunk with metadata
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, {"start_idx": i - len(current_chunk), "end_idx": i}))

                # Handle overlap
                overlap_sentences = current_chunk[-self.overlap:] if self.overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, {"start_idx": len(sentences) - len(current_chunk), "end_idx": len(sentences)}))  # check for start and end idx

        return chunks


class QueryExpander:
    def __init__(self):
        # You could make this more sophisticated with templates or ML
        self.expansion_templates = [
            "{query}",
            "What does the text say about {query}",
            "Find information related to {query}",
            "Extract details about {query}"
        ]

    def expand_query(self, query: str) -> List[str]:
        # Simple query expansion using templates
        expanded = [template.format(query=query) for template in self.expansion_templates]
        return expanded


class RAGRetriever:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.query_expander = QueryExpander()

    def _create_index(self, embeddings: np.ndarray):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))

    def add_document(self, text: str):
        # Create chunks
        chunker = TextChunker()
        text_chunks = chunker.chunk_text(text)

        # Get embeddings for chunks
        chunk_texts = [chunk[0] for chunk in text_chunks]
        embeddings = self.embedding_model.encode(chunk_texts)

        # Initialize or add to FAISS index
        if self.index is None:
            self._create_index(embeddings)
        else:
            self.index.add(embeddings.astype('float32'))

        # Store chunks
        start_idx = len(self.chunks)
        self.chunks.extend(text_chunks)

        return start_idx

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        # Expand query
        expanded_queries = self.query_expander.expand_query(query)

        # Get embeddings for expanded queries
        query_embeddings = self.embedding_model.encode(expanded_queries)

        # Search for each expanded query
        all_retrieved = []
        seen_chunks = set()

        for query_embedding in query_embeddings:
            D, I = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)

            for idx in I[0]:
                if idx not in seen_chunks:
                    seen_chunks.add(idx)
                    all_retrieved.append(self.chunks[idx][0])

                    if len(all_retrieved) >= k:
                        break

            if len(all_retrieved) >= k:
                break

        return all_retrieved


class RAGPipeline:
    def __init__(self, model_config: dict, model_path: str, device: str = 'cpu'):
        self.retriever = RAGRetriever()

        # Initialize your GPT model
        self.model = GPTModel(model_config)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.device = device

    def add_document(self, text: str):
        return self.retriever.add_document(text)

    def _format_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        # Format context into a concise form
        context = "\n".join(retrieved_chunks)

        # Format following the training instruction format
        prompt = f"""Below is an instruction that describes a task. 
                Write a response that appropriately completes the request.

                ### Instruction:
                Given the following Input, provide a direct and specific answer to the question: {query}

                ### Input:
                {context}

                ### Response: """

        return prompt

    def generate(self, query: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query)

        # Format prompt using training format
        prompt = self._format_prompt(query, retrieved_chunks)

        # Generate response
        input_tokens = text_to_token_ids(prompt, tokenizer)
        generated = generate_text(
            self.model,
            input_tokens,
            self.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            eos_id=50256  # Stop at end of text token
        )

        # Extract only the response part
        full_text = token_ids_to_text(generated, tokenizer)

        # Find the response after "### Response:"
        try:
            response_start = full_text.index("### Response:") + len("### Response:")
            response = full_text[response_start:].strip()

            # Remove any trailing instructions or new queries
            if "<|endoftext|>" in response:
                response = response.split("<|endoftext|>")[0].strip()

            return response
        except ValueError:
            # Fallback if format is not found
            return full_text.strip()

    def _clean_response(self, response: str) -> str:
        # Remove any special tokens
        response = response.replace("<|endoftext|>", "").strip()

        # Remove any trailing questions or new instructions
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0].strip()

        if "Below is an instruction" in response:
            response = response.split("Below is an instruction")[0].strip()

        return response


def generate_text(model, start_tokens, device, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
    model.eval()
    start_tokens = start_tokens.to(device)
    generated_tokens = start_tokens.clone()

    for _ in range(max_new_tokens):
        # Take last context_length tokens as context
        context = generated_tokens[:, -model.config['context_length']:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and (next_token.item() == eos_id):
            break

        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

    model.train()
    return generated_tokens


# Example usage
if __name__ == "__main__":
    config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 1024,
        'n_layers': 24,
        'n_heads': 16,
        'drop_rate': 0.1,
        'qkv_bias': True,
    }

    # Initialize RAG pipeline
    rag = RAGPipeline(model_config=config, model_path='best_weights.pth', device='cpu')

    # Add a document
    document = (["Aston Martin have announced that Andy Cowell will assume the role of Team Principal alongside his position as CEO with immediate effect, with Mike Krack moving to the position of "
                 "Chief Trackside Officer as part of an organisational restructure. In a shift towards a flatter structure, Cowell – who previously joined the squad in October as CEO – will also become "
                 "Team Principal, with the squad's Aerodynamics, Engineering and Performance Departments both trackside and at the AMR Technology Campus reporting into him. Krack – who has held the role of "
                 "Team Principal since 2022 – will focus on getting the most performance out of the car at the track in his role as Chief Trackside Officer.The AMR Technology Campus-based team, meanwhile, "
                 "are set to be led by new Chief Technical Officer Enrico Cardile, while Tom McCullough will remain with the group in a leadership position. Cardile will oversee the architecture, design "
                 "and build of new race cars, having joined the team after leaving his position as Technical Director Chassis and Aerodynamics at Ferrari last year.McCullough – who held the position of "
                 "Performance Director and has worked with the outfit for 11 years – is set to play a role in the expansion of the team's broader range of racing categories."])

    for i in range(len(document)):
        rag.add_document(document[i])

        # Generate a response
        query = "What is the role of Mike Krack?"
        response = rag.generate(query, temperature=0.7)
        print(f"Query: {query}")
        print(f"Response: {response}")

        model = SentenceTransformer("all-mpnet-base-v2")
        embedding_doc1 = model.encode(query)
        embedding_doc2 = model.encode(response)
        similarity = model.similarity(embedding_doc1, embedding_doc2)
        print("\nSimilarity: ",similarity)