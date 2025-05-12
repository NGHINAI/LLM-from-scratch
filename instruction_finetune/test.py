import json
from torch.utils.data import DataLoader
from functools import partial
import math
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel

tokenizer = tiktoken.get_encoding('gpt2')

sample_prompt = {
        "instruction": "Provide reasons why people should continue using single-use plastics.",
        "input": "",
        "output": "Using single-use plastics can have detrimental effects on the environment, leading to pollution and harm to wildlife. We should encourage sustainable alternatives like reusable materials to protect our planet."
    }

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['n_heads']
        self.head_size = config['emb_dim'] // config['n_heads']

        self.W_query = nn.Linear(config['emb_dim'], config['emb_dim'], bias=True)
        self.W_key = nn.Linear(config['emb_dim'], config['emb_dim'], bias=True)
        self.W_value = nn.Linear(config['emb_dim'], config['emb_dim'], bias=True)
        self.out_proj = nn.Linear(config['emb_dim'], config['emb_dim'], bias=True)

        self.dropout = nn.Dropout(config['drop_rate'])
        self.register_buffer("mask", torch.triu(torch.ones(config['context_length'], config['context_length']), diagonal=1).bool())

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            nn.GELU(approximate='tanh'),
            nn.Dropout(config['drop_rate']),
            nn.Linear(4 * config['emb_dim'], config['emb_dim']),
            nn.Dropout(config['drop_rate'])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config['emb_dim'])
        self.norm2 = nn.LayerNorm(config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        attended = x + self.dropout(self.attention(self.norm1(x)))
        output = attended + self.dropout(self.feed_forward(self.norm2(attended)))
        return output

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])

        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])

        self.final_norm = nn.LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        B, T = idx.shape

        token_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(token_emb + pos_emb)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

def assign(param, value):
    if isinstance(param, nn.Parameter):
        param.data.copy_(torch.tensor(value))
    else:
        param.copy_(torch.tensor(value))
    return param

def load_pretrained_gpt2_params(model):
    model = GPT2LMHeadModel.from_pretrained(model)

    params = {
        'wpe': model.transformer.wpe.weight.detach().numpy(),
        'wte': model.transformer.wte.weight.detach().numpy(),
        'blocks': [],
        'g': model.transformer.ln_f.weight.detach().numpy(),
        'b': model.transformer.ln_f.bias.detach().numpy()
    }

    for block in model.transformer.h:
        block_params = {
            'attn': {
                'c_attn': {
                    'w': block.attn.c_attn.weight.detach().numpy(),
                    'b': block.attn.c_attn.bias.detach().numpy()
                },
                'c_proj': {
                    'w': block.attn.c_proj.weight.detach().numpy(),
                    'b': block.attn.c_proj.bias.detach().numpy()
                }
            },
            'mlp': {
                'c_fc': {
                    'w': block.mlp.c_fc.weight.detach().numpy(),
                    'b': block.mlp.c_fc.bias.detach().numpy()
                },
                'c_proj': {
                    'w': block.mlp.c_proj.weight.detach().numpy(),
                    'b': block.mlp.c_proj.bias.detach().numpy()
                }
            },
            'ln_1': {
                'g': block.ln_1.weight.detach().numpy(),
                'b': block.ln_1.bias.detach().numpy()
            },
            'ln_2': {
                'g': block.ln_2.weight.detach().numpy(),
                'b': block.ln_2.bias.detach().numpy()
            }
        }
        params['blocks'].append(block_params)

    return params

def load_weights_into_gpt(model, params):
    model.pos_emb.weight = assign(model.pos_emb.weight, params['wpe'])
    model.tok_emb.weight = assign(model.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model.trf_blocks[b].attention.W_query.weight = assign(model.trf_blocks[b].attention.W_query.weight, q_w.T)
        model.trf_blocks[b].attention.W_key.weight = assign(model.trf_blocks[b].attention.W_key.weight, k_w.T)
        model.trf_blocks[b].attention.W_value.weight = assign(model.trf_blocks[b].attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model.trf_blocks[b].attention.W_query.bias = assign(model.trf_blocks[b].attention.W_query.bias, q_b)
        model.trf_blocks[b].attention.W_key.bias = assign(model.trf_blocks[b].attention.W_key.bias, k_b)
        model.trf_blocks[b].attention.W_value.bias = assign(model.trf_blocks[b].attention.W_value.bias, v_b)

        model.trf_blocks[b].attention.out_proj.weight = assign(model.trf_blocks[b].attention.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model.trf_blocks[b].attention.out_proj.bias = assign(model.trf_blocks[b].attention.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        model.trf_blocks[b].feed_forward.layers[0].weight = assign(model.trf_blocks[b].feed_forward.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model.trf_blocks[b].feed_forward.layers[0].bias = assign(model.trf_blocks[b].feed_forward.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])

        model.trf_blocks[b].feed_forward.layers[3].weight = assign(model.trf_blocks[b].feed_forward.layers[3].weight,params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model.trf_blocks[b].feed_forward.layers[3].bias = assign(model.trf_blocks[b].feed_forward.layers[3].bias,params["blocks"][b]["mlp"]["c_proj"]["b"])

        model.trf_blocks[b].norm1.weight = assign(model.trf_blocks[b].norm1.weight, params["blocks"][b]["ln_1"]["g"])
        model.trf_blocks[b].norm1.bias = assign(model.trf_blocks[b].norm1.bias, params["blocks"][b]["ln_1"]["b"])

        model.trf_blocks[b].norm2.weight = assign(model.trf_blocks[b].norm2.weight,params["blocks"][b]["ln_2"]["g"])
        model.trf_blocks[b].norm2.bias = assign(model.trf_blocks[b].norm2.bias, params["blocks"][b]["ln_2"]["b"])

    model.final_norm.weight = assign(model.final_norm.weight, params["g"])
    model.final_norm.bias = assign(model.final_norm.bias, params["b"])
    model.out_head.weight = assign(model.out_head.weight, params["wte"])

    return model

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'min_lr': self.min_lr
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.min_lr = state_dict['min_lr']

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text(model, start_tokens, device, max_new_tokens, temperature=1.0, top_k=None, eos_id=50256):
    model.eval()
    start_tokens = start_tokens.to(device)

    for _ in range(max_new_tokens):
        context = start_tokens[:, -model.config['context_length']:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token == eos_id:
            break

        start_tokens = torch.cat((start_tokens, next_token), dim=1)

    model.train()
    return start_tokens

def calculate_loss(model, input_ids, target_ids, device):

    logits = model(input_ids)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = target_ids.view(-1)
    loss = F.cross_entropy(logits, targets)

    del input_ids, target_ids
    return loss

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            loss = calculate_loss(model, input_ids, target_ids, device)
            total_loss += loss.item()
            total_batches += 1

            del input_ids, target_ids

    model.train()
    return total_loss / total_batches

def train_model(config, model, train_loader, val_loader, optimizer, scheduler, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    #gradient_accumulation_steps = 4

    try:
        for epoch in range(config['n_epochs']):
            model.train()
            total_loss = 0
            total_batches = 0
            optimizer.zero_grad()

            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):

                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                loss = calculate_loss(model, input_ids, target_ids, device)
                #loss = loss / gradient_accumulation_steps
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

                optimizer.step()

                #if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    #optimizer.step()
                    #if scheduler is not None:
                        #scheduler.step()

                scheduler.step()
                optimizer.zero_grad()

                #total_loss += loss.item() * gradient_accumulation_steps
                total_loss += loss.item()
                total_batches += 1

                if batch_idx % config['log_interval'] == 0:
                    avg_loss = total_loss / total_batches
                    #validation_loss = evaluate_model(model, val_loader, device)
                    lr = scheduler.get_lr() if scheduler else optimizer.param_groups[0]['lr']
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Train Loss: {avg_loss:.4f}, LR: {lr:.2e}')

                    if (batch_idx % 20) == 0 and batch_idx > 0:
                        validation_loss = evaluate_model(model, val_loader, device)
                        print(f'Val Loss: {validation_loss:.4f}')

                    if (batch_idx % 500) == 0 and batch_idx > 0:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        }, f'checkpoint_batch_{batch_idx}.pth')

                    if batch_idx % config['generate_interval'] == 0:
                        context = format_input(sample_prompt)
                        tokens = text_to_token_ids(context, tokenizer)
                        generated = generate_text(model, tokens, device, max_new_tokens=20, temperature=0.8, top_k=30)
                        print("\nSample generation:")
                        print(token_ids_to_text(generated, tokenizer))
                        print()

                del input_ids, target_ids, loss

            val_loss = evaluate_model(model, val_loader, device)
            print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')

            train_losses.append(total_loss / total_batches)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print("Early stopping triggered!")
                    break

    except KeyboardInterrupt as e:
        print("Training interrupted. Saving model checkpoint...")

    return train_losses, val_losses

# Load and prepare data
with open('../datasets/processed_instruction_dataset_sample.json', "r", encoding="utf-8") as file:
    data = json.load(file)

# Split data
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
device = torch.device("cpu")

# Prepare data loaders
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# Model configuration
config = {
    # Model architecture
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 1024,
    'n_layers': 24,
    'n_heads': 16,
    'drop_rate': 0.1,
    'qkv_bias': True,

    # Training
    'batch_size': 2,
    'n_epochs': 5,
    'learning_rate': 0.000003,
    'weight_decay': 0.1,
    'warmup_steps': 100,
    'grad_clip': 1,
    'patience': 3,

    # Data
    'stride_length': 256,
    'num_workers': 0,

    # Logging
    'log_interval': 10,
    'generate_interval': 500
}

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=0
)


# Initialize model
model = GPTModel(config)
checkpoint = torch.load('best_weights.pth', weights_only=True, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
#pretrained_params = load_pretrained_gpt2_params(model='gpt2-medium')
#model = load_weights_into_gpt(model, pretrained_params).to(device)

#prompt = test_data[201]
prompt = 'What is Sentence Transformers?'
#context = format_input(prompt)

tokens = text_to_token_ids(prompt, tokenizer)
generated = generate_text(model, tokens, device, max_new_tokens=100, temperature=0.7, top_k=30)
print("\nSample generation:")
print(token_ids_to_text(generated, tokenizer))
#print("\n\n",prompt['output'])