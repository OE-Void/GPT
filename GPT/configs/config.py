import torch
from transformers import PretrainedConfig

# standard hyperparameters to train and make a standard transformers model

batch_size = 32 # number of sequences processed at once decrease to 4-6 for t4
block_size = 4096 # maximum context length we took 4096 use 1024 for t4 as dataset have average length of around 800 to 900 tokens if its low like 256 model will not learn about eos token
max_iters = 800 # maximum number of training steps
eval_interval = 200 # evaluate every 200 steps
learning_rate = 3e-4 # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # logging step
n_embd = 384 # embedding dimension
n_head = 6 # number of attention heads per tranforms layer now its 6 heads per layer with 64-dim each head
# yoo if you think i haven't defined the head dim check it head_dim = n_embed/nhead
n_layer = 12 # number of transformer layers
dropout = 0.2 # dropout rate it avoids overfitting if you want to know more google it
torch.manual_seed(1337) # seed for reproducibility

print(f"Using device: {device}")

class GPTCustomConfig(PretrainedConfig):
    model_type = "gpt_custom"
    def __init__(
        self,
        vocab_size=50257,
        n_embd=384,
        block_size=4096,
        n_head=6,
        n_layer=6,
        dropout=0.2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        super().__init__(**kwargs)
