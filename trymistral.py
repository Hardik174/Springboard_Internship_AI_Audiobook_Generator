from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistral-7B-instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"   # auto-uses GPU if available
)

prompt = "Extract names and emails from: John Doe, john@example.com"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
