from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Example usage
# python load_qwen_model.py
if __name__ == "__main__":
    # Test the model with a simple prompt
    text = "Hello, how are you?"
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Response: {response}") 