# GPT From Scratch

[`https://colab.research.google.com/github/OE-Void/GPT/blob/main/model_from_scr.ipynb`](https://colab.research.google.com/github/OE-Void/GPT/blob/main/model_from_scr.ipynb)

A **PyTorch implementation of a GPT-style language model**, built from scratch for **educational purposes** and **scalable usage**.  
This project demonstrates how transformer-based language models can be trained, evaluated, and deployed.

---

## ✨ Features
- Minimal, modular PyTorch implementation of GPT
- Configurable hyperparameters (`n_embd`, `n_layer`, `n_head`, etc.)
- Training loop with evaluation and checkpoint saving
- Hugging Face integration for easy upload and inference
- Colab notebook for quick experimentation

---

## 🚀 Usage After Training (with Hugging Face)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained('your_repo_id', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('your_repo_id')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Generating text...")

# Start with BOS token
context = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

# Generate sequence
generated_ids = model.generate(context, max_new_tokens=256)[0].tolist()
print(tokenizer.decode(generated_ids))
```

---

## 📂 Project Structure

```
GPT/
├── configs/     # Configuration files (config.py)
├── data/        # Data loading and preprocessing (dataset.py)
├── model/       # Model definition (model.py)
├── trainer/     # Training loop and saving logic (trainer.py)
└── main.py      # Entry point for training and generation
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OE-Void/GPT.git
   cd GPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training

To train the model:

```bash
python -m GPT.main
```

The trained model will be saved in the `my_model` directory.

---

## 🔧 Configuration

Edit `GPT/configs/config.py` to adjust hyperparameters such as:
- `n_embd` → embedding dimension size
- `n_layer` → number of transformer layers
- `n_head` → number of attention heads
- `block_size` → maximum sequence length
- `batch_size` → training batch size

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License — see the `LISENCE` file for details.

