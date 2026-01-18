import torch
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer
from GPT.configs.config import batch_size, block_size

class GPTDataLoader:
    def __init__(self, device):
        self.device = device
        self.tokenizer = None
        self.train_data = None
        self.val_data = None
        self.vocab_size = 0
        self.bos_id = 0
        self.eos_id = 0
        
    def prepare_data(self):
        # loading the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        special_tokens_dict = { "eos_token": "<eos>", "bos_token": "<bos>", "pad_token": "<pad>" } # adding extra tokens
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.vocab_size = len(self.tokenizer)
        print(f"Vocab size: {self.vocab_size}")
        
        # lets preprocess data for training
        dataset = load_dataset("roneneldan/TinyStories", split="train[:10000]") # only 10k samples are dowloaded for training

        print("Processing data...")

        def encode_function(examples):
            ids_list = []
            for text in examples["text"]: # change the text to desired row if dataset is changed by you
                ids = self.tokenizer.encode(text, verbose=False)
                ids = [self.bos_id] + ids + [self.eos_id] # manually adding eos and bos as tokenizer
                ids_list.append(ids)
            return {"input_ids": ids_list}

        tokenized_dataset = dataset.map(encode_function, batched=True, remove_columns=["text"])

        # Flatten (packing the sequence)
        ids = list(itertools.chain(*tokenized_dataset["input_ids"]))
        data = torch.tensor(ids, dtype=torch.long)

        # Split
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data_source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(0, len(data_source) - block_size - 1, (batch_size,))
        x = torch.stack([data_source[i:i+block_size] for i in ix])
        y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)
