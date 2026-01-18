# GPT From Scratch

`[Open in Colab](https://colab.research.google.com/github/OE-Void/GPT/blob/main/model_from_scr.ipynb)`

This repository contains a PyTorch implementation of a GPT-style language model, structured for educational purposes and mass usage.

## usage after training and uploading to hf

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('oe-void/transformers', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('oe-void/transformers')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Generating text...")
context = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
generated_ids = m.generate(context, max_new_tokens=256)[0].tolist()
print(tokenizer.decode(generated_ids))

```
## Structure

- `GPT/`: Main package
  - `configs/`: Configuration files (`config.py`)
  - `data/`: Data loading and processing (`dataset.py`)
  - `model/`: Model definition (`model.py`)
  - `trainer/`: Training loop and saving logic (`trainer.py`)
  - `main.py`: Entry point for training and generation

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model:

```bash
python -m GPT.main
```

The model will be saved to the `my_model` directory.

## Configuration

Edit `GPT/configs/config.py` to change hyperparameters like `n_embd`, `n_layer`, `n_head`, etc.
