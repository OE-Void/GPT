# GPT
A basic GPT from scratch

## final loading script
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('your_hf-repo_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('your_hf-repo_id)

print("Generating text...")
# We start with the BOS token this is for a open ended generation from bos token
context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
generated_ids = m.generate(context, max_new_tokens=256)[0].tolist()
print(tokenizer.decode(generated_ids))
```
