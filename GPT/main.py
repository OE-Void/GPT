import torch
from GPT.configs.config import n_embd, n_head, n_layer, dropout, block_size, device
from GPT.data.dataset import GPTDataLoader
from GPT.model.model import GPTLanguageModel, GPTCustomConfig
from GPT.trainer.trainer import Trainer

def main():
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    loader = GPTDataLoader(device)
    loader.prepare_data()
    
    # 2. Initialize Model
    print("Initializing model...")
    # Create config using the hyperparameters and vocab size from loader
    config = GPTCustomConfig(
        vocab_size=loader.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    
    model = GPTLanguageModel(config)
    m = model.to(device)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
    
    # 3. Initialize Trainer
    trainer = Trainer(model, loader)
    
    # 4. Train
    trainer.train()
    
    # 5. Inference / Generation
    print("Generating text...")
    # We start with the BOS token
    context = torch.tensor([[loader.bos_id]], dtype=torch.long, device=device)
    # generate uses the simpler custom loop in model.py
    generated_ids = m.generate(context, max_new_tokens=256)[0].tolist()
    print(loader.tokenizer.decode(generated_ids))
    
    # 6. Save Model
    trainer.save_model()

if __name__ == "__main__":
    main()
