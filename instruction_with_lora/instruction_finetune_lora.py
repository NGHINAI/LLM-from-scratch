import json
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel
import gc
from transformers import Adafactor

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load instruction dataset
with open('../datasets/processed_instruction_dataset.json', "r", encoding="utf-8") as file:
    data = json.load(file)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# Split data
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# LoRA implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, device):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.alpha


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
    lora_state_dict = {
        'weights': {},
        'config': {}
    }
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict['weights'][name] = param.data

    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_state_dict['config']['rank'] = module.rank
            lora_state_dict['config']['alpha'] = module.alpha
            break

    torch.save(lora_state_dict, path)


# Transformer implementation
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
        self.register_buffer("mask", torch.triu(torch.ones(config['context_length'], config['context_length']),
                                                diagonal=1).bool())

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
        self.trf_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layers'])])
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
        return self.out_head(x)


# Training utilities
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
            lr = self.base_lr * (self.current_step / self.warmup_steps) ** 2
        else:
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Return the state of the scheduler for saving."""
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'min_lr': self.min_lr
        }

    def load_state_dict(self, state_dict):
        """Load the state of the scheduler for resuming."""
        self.current_step = state_dict['current_step']
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.min_lr = state_dict['min_lr']


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
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

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def train_model(config, model, train_loader, val_loader, optimizer, scheduler, device):
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    grad_accumulation_steps = 4

    for epoch in range(config['n_epochs']):
        total_loss = 0
        total_batches = 0

        torch.mps.empty_cache()
        gc.collect()

        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):

            torch.mps.empty_cache()
            gc.collect()

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), target_ids.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_idx % config['log_interval'] == 0:
                #print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_lr()}')
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            if (batch_idx % 500) == 0 and batch_idx > 0:

                # Validation
                model.eval()
                val_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for input_ids, target_ids in val_loader:
                        input_ids = input_ids.to(device)
                        target_ids = target_ids.to(device)

                        logits = model(input_ids)
                        B, T, C = logits.shape
                        loss = F.cross_entropy(logits.view(B * T, C), target_ids.view(-1))
                        val_loss += loss.item()
                        val_batches += 1

                val_loss = val_loss / val_batches
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Validation Loss: {val_loss:.4f}')
                model.train()

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B * T, C), target_ids.view(-1))
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, 'best_model.pth')
            save_lora_weights(model, 'best_instruction_lora_weights.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping triggered!")
                break

        model.train()

        train_losses.append(total_loss / total_batches)
        val_losses.append(val_loss)

    return train_losses, val_losses


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Configuration
    config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_layers': 12,
        'n_heads': 12,
        'drop_rate': 0.1,
        'batch_size': 4,
        'n_epochs': 8,
        'learning_rate': 0.000001,
        'weight_decay': 0.001,
        'warmup_steps': 1000,
        'patience': 4,
        'log_interval': 20
    }

    # Create dataloaders
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    collate_fn = lambda batch: custom_collate_fn(batch, allowed_max_length=1024 ,device=device)

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

    # Add LoRA layers and make them trainable
    replace_linear_with_lora(model, rank=8, alpha=8, device=device)

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {trainable_params:,}')

    # Initialize optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Initialize learning rate scheduler
    total_steps = len(train_loader) * config['n_epochs']
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config['warmup_steps'], max_steps=total_steps, min_lr=1e-6)

    # Train the model
    try:
        train_losses, val_losses = train_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=False,
            device=device
        )

        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config
        }, 'instruction_lora_final.pth')

        # Save LoRA weights separately
        save_lora_weights(model, 'instruction_lora_weights.pth')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")