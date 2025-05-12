import gc
import torch
import torch.nn as nn
import math
from GPT import create_dataloader, tokenizer, GPTModel, load_pretrained_gpt2_params, load_weights_into_gpt, WarmupCosineScheduler, train_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.mps.empty_cache()
gc.collect()

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, device):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))
        self.alpha = alpha

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        x = (x @ self.lora_A @ self.lora_B) * self.alpha
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, device):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, device)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, rank, alpha, device):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha, device))
        else:
            replace_linear_with_lora(module, rank, alpha, device)


def save_lora_weights(model, path):
    """Save only the LoRA weights and configuration"""
    lora_state_dict = {
        'weights': {},
        'config': {}
    }
    # Save weights
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict['weights'][name] = param.data

    # Save LoRA configuration (get from first LoRA layer)
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_state_dict['config']['rank'] = module.rank
            lora_state_dict['config']['alpha'] = module.alpha
            break

    torch.save(lora_state_dict, path)


# Modified training configuration
lora_config = {
    # Model architecture
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,

    # Training
    'batch_size': 4,
    'n_epochs': 10,
    'learning_rate': 0.0002,
    'weight_decay': 0.02,
    'warmup_steps': 1000,
    'grad_clip': 1.0,
    'patience': 5,
    
    # Data
    'stride_length': 512,
    'num_workers': 0,

    # Logging
    'log_interval': 20,
    'generate_interval': 500
}

with open('finance_dataset.txt', 'r', encoding='utf-8') as f:
    combined_text = f.read()

train_ratio = 0.90
split_idx = int(len(combined_text) * train_ratio)
train_text = combined_text[:split_idx]
val_text = combined_text[split_idx:]

# Create dataloaders
train_loader = create_dataloader(train_text, lora_config, tokenizer)
val_loader = create_dataloader(val_text, lora_config, tokenizer)

# Initialize model
model = GPTModel(lora_config)
pretrained_params = load_pretrained_gpt2_params()
model = load_weights_into_gpt(model, pretrained_params)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

replace_linear_with_lora(model, rank=8, alpha=16, device=device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params to train with LoRA:',total_params)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lora_config['learning_rate'], weight_decay=lora_config['weight_decay'])

total_steps = len(train_loader) * lora_config['n_epochs']
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=lora_config['warmup_steps'], max_steps=total_steps, min_lr=0.000001)

train_losses, val_losses = train_model(
    config=lora_config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'config': lora_config
}, 'lora_final_checkpoint.pth')

# Save LoRA weights separately
save_lora_weights(model, 'final_lora_weights.pth')

