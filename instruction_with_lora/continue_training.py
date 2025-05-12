import gc
import json

from transformers import GPT2LMHeadModel

from instruction_finetune_lora import (
    GPTModel,
    InstructionDataset,
    WarmupCosineScheduler,
    train_model,
    custom_collate_fn,
    tokenizer, replace_linear_with_lora,
)
import torch
from torch.utils.data import DataLoader

with open('../datasets/processed_instruction_dataset.json', "r", encoding="utf-8") as file:
    data = json.load(file)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
torch.mps.empty_cache()
gc.collect()

# New configuration with adjusted parameters
config = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'drop_rate': 0.1,      # Reduced dropout
    'batch_size': 4,
    'n_epochs': 8,        # More epochs
    'learning_rate': 0.000005,  # Higher learning rate
    'weight_decay': 0.01,  # Lower weight decay
    'warmup_steps': 100,   # Adjusted warmup
    'patience': 4,         # More patience
    'log_interval': 20     # More frequent logging
}

# Create dataloaders with updated batch size
train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)

collate_fn = lambda batch: custom_collate_fn(batch, allowed_max_length=1024, device=device)

torch.manual_seed(123)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=False,
    drop_last=False
)

# Initialize model
model = GPTModel(config)

pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Copy weights from pretrained model to our model
model.tok_emb.weight.data.copy_(pretrained_model.transformer.wte.weight.data)
model.pos_emb.weight.data.copy_(pretrained_model.transformer.wpe.weight.data)

for i, block in enumerate(model.trf_blocks):
    # Copy attention weights
    block.attention.W_query.weight.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.weight.data[:, :config['emb_dim']].t()
    )
    block.attention.W_key.weight.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.weight.data[:, config['emb_dim']:2 * config['emb_dim']].t()
    )
    block.attention.W_value.weight.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.weight.data[:, 2 * config['emb_dim']:].t()
    )

    block.attention.W_query.bias.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.bias.data[:config['emb_dim']]
    )
    block.attention.W_key.bias.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.bias.data[config['emb_dim']:2 * config['emb_dim']]
    )
    block.attention.W_value.bias.data.copy_(
        pretrained_model.transformer.h[i].attn.c_attn.bias.data[2 * config['emb_dim']:]
    )

    # Copy projection weights
    block.attention.out_proj.weight.data.copy_(
        pretrained_model.transformer.h[i].attn.c_proj.weight.data.t()
    )
    block.attention.out_proj.bias.data.copy_(
        pretrained_model.transformer.h[i].attn.c_proj.bias.data
    )

    # Copy MLP weights
    block.feed_forward.layers[0].weight.data.copy_(
        pretrained_model.transformer.h[i].mlp.c_fc.weight.data.t()
    )
    block.feed_forward.layers[0].bias.data.copy_(
        pretrained_model.transformer.h[i].mlp.c_fc.bias.data
    )
    block.feed_forward.layers[3].weight.data.copy_(
        pretrained_model.transformer.h[i].mlp.c_proj.weight.data.t()
    )
    block.feed_forward.layers[3].bias.data.copy_(
        pretrained_model.transformer.h[i].mlp.c_proj.bias.data
    )

    # Copy layer norm weights
    block.norm1.weight.data.copy_(
        pretrained_model.transformer.h[i].ln_1.weight.data
    )
    block.norm1.bias.data.copy_(
        pretrained_model.transformer.h[i].ln_1.bias.data
    )
    block.norm2.weight.data.copy_(
        pretrained_model.transformer.h[i].ln_2.weight.data
    )
    block.norm2.bias.data.copy_(
        pretrained_model.transformer.h[i].ln_2.bias.data
    )

# Copy final layer norm weights
model.final_norm.weight.data.copy_(
    pretrained_model.transformer.ln_f.weight.data
)
model.final_norm.bias.data.copy_(
    pretrained_model.transformer.ln_f.bias.data
)

# Move model to device
model = model.to(device)

# Freeze base model weights
for param in model.parameters():
    param.requires_grad = False

replace_linear_with_lora(model, rank=8, alpha=8, device=device)

# Load LoRA weights
lora_state = torch.load('best_instruction_lora_weights.pth', map_location=device, weights_only=True)
for name, param in model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        if name in lora_state['weights']:
            param.data.copy_(lora_state['weights'][name])

# Set up optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])

# Set up scheduler
total_steps = len(train_loader) * config['n_epochs']
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config['warmup_steps'], max_steps=total_steps, min_lr=1e-6)

# Print trainable parameter count
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {trainable_params:,}')

# Continue training
try:
    train_losses, val_losses = train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Save LoRA weights separately
    lora_state_dict = {
        'weights': {},
        'config': {}
    }
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict['weights'][name] = param.data

    torch.save(lora_state_dict, 'continued_lora_weights.pth')

except KeyboardInterrupt:
    lora_state_dict = {'weights': {}, 'config': {}}

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict['weights'][name] = param.data

    torch.save(lora_state_dict, 'continued_lora_weights_interrupted.pth')
    print("\nTraining interrupted by user. Saving checkpoint...")
