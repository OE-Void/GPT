import torch
import os
import json
from safetensors.torch import save_file
from GPT.configs.config import max_iters, eval_interval, eval_iters, learning_rate, device
from GPT.model.model import GPTLanguageModel, GPTCustomConfig

class Trainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.data_loader.get_batch(split)
                # Model (PreTrainedModel) returns (loss, logits) when labels are provided
                loss, logits = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        print("Starting training...")
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            xb, yb = self.data_loader.get_batch("train")
            # Model (PreTrainedModel) returns (loss, logits) when labels are provided
            loss, logits = self.model(xb, yb)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        print("Training complete!")

    def save_model(self, output_dir="my_model"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Read modeling code from model.py to save as a standalone file
        model_file_path = os.path.join(os.path.dirname(__file__), '../model/model.py')
        try:
            with open(model_file_path, "r") as f:
                modeling_code = f.read()
        except FileNotFoundError:
            # Fallback if running from a different location or path issues
            # We construct it from the classes (not implemented here to keep it simple, expecting file presence)
            print(f"Warning: Could not correct read {model_file_path}. Saving empty modeling file.")
            modeling_code = ""

        print("Saving model configuration...")
        with open(os.path.join(output_dir, "modeling_gpt_custom.py"), "w") as f:
            f.write(modeling_code)

        config = self.model.config
        config_dict = {
            "architectures": ["GPTLanguageModel"],
            "model_type": "gpt_custom",
            "auto_map": {
                "AutoConfig": "modeling_gpt_custom.GPTCustomConfig",
                "AutoModel": "modeling_gpt_custom.GPTLanguageModel",
                "AutoModelForCausalLM": "modeling_gpt_custom.GPTLanguageModel"
            },
            "vocab_size": config.vocab_size,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
            "bos_token_id": self.data_loader.bos_id,
            "eos_token_id": self.data_loader.eos_id,
            "pad_token_id": self.data_loader.tokenizer.pad_token_id
        }

        print("Saving config.json...")
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        print("Saving model weights...")
        save_file(self.model.state_dict(), os.path.join(output_dir, "model.safetensors"))

        print("Saving tokenizer...")
        self.data_loader.tokenizer.save_pretrained(output_dir)

        print("="*50)
        print(f"Model saved successfully to '{output_dir}/'")
        print("="*50)
        print("HOW TO LOAD THIS MODEL:")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
