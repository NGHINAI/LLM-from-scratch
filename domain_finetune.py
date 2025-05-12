from GPT import *


if __name__ == "__main__":

    # Load the JSON file
    with open('finance_dataset.txt', 'r', encoding="utf-8") as f:
        data = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    # Configuration matching GPT-2 (124M)
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
        'batch_size': 6,
        'n_epochs': 4,
        'learning_rate': 0.000003,
        'weight_decay': 0.1,
        'warmup_steps': 1000,
        'grad_clip': 2.5,
        'patience': 3,

        # Data
        'stride_length': 256,
        'num_workers': 0,

        # Logging
        'log_interval': 20,
        'generate_interval': 1000
    }

    train_ratio = 0.90
    split_idx = int(len(data) * train_ratio)
    train_text = data[:split_idx]
    val_text = data[split_idx:]

    # Create dataloaders
    train_loader = create_dataloader(train_text, config, tokenizer)
    val_loader = create_dataloader(val_text, config, tokenizer)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.mps.empty_cache()
    gc.collect()

    model = GPTModel(config).to(device)
    #pretrained_params = load_pretrained_gpt2_params()
    #model = load_weights_into_gpt(model, pretrained_params)
    checkpoint = torch.load("best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.out_head.parameters():
        param.requires_grad = True

    for i in range(-2, 0):
        for param in model.trf_blocks[i].parameters():
            param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])

    total_steps = len(train_loader) * config['n_epochs']
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=config['warmup_steps'], max_steps=total_steps, min_lr=1e-6)

    train_losses, val_losses = train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device
    )

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
                },
               'final_model.pth'
               )

    # Example text generation
    #prompt = "What is value of put if underlying stays below strike?"
    #input_tokens = text_to_token_ids(prompt, tokenizer).to(device)
    #generated = generate_text(model, input_tokens, max_new_tokens=50, temperature=0.8, top_k=40)
    #generated_text = token_ids_to_text(generated, tokenizer)

    #print(f"\nPrompt: {prompt}")
    #print(f"Generated text: {generated_text}")
