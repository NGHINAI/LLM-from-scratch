from GPT import *

config = {
    # Model architecture
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,

    # Training
    'batch_size': 8,
    'n_epochs': 3,
    'learning_rate': 0.00005,
    'weight_decay': 0.04,
    'warmup_steps': 300,
    'grad_clip': 1.0,
    'patience': 2,

    # Data
    'stride_length': 512,
    'num_workers': 0,

    # Logging
    'log_interval': 20,
    'generate_interval': 100
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# Example text generation
model = GPTModel(config).to(device)
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

prompt = ("Shareholder's value")

input_tokens = text_to_token_ids(prompt, tokenizer).to(device)
generated = generate_text(model, input_tokens, max_new_tokens=100, temperature=0.8, top_k=20)
generated_text = token_ids_to_text(generated, tokenizer)

print(f"\nPrompt: {prompt}")
print(f"Generated text: {generated_text}")