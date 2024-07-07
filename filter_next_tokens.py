import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_with_filter(prompt, max_length=50, top_k=10, forbidden_words=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if forbidden_words is None:
        forbidden_words = []
    
    forbidden_ids = set()
    for word in forbidden_words:
        forbidden_ids.update(tokenizer.encode(word, add_special_tokens=False))
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
        
        # get the top k token probabilities
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        print("\nNext token possibilities:")
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            token = tokenizer.decode([idx])
            # tokenizer.decode(top_k_indices[0])

            print(f"{token}: {prob.item():.4f}")

        # printout the next token before filtering: 
        next_token = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[0][next_token]
        print(f"\nGenerated token before filtering: '{tokenizer.decode(next_token[0])}'")
        
        # filter out forbidden tokens
        for i, idx in enumerate(top_k_indices[0]):
            if idx.item() in forbidden_ids:
                top_k_probs[0][i] = 0
                print(f"Filtered token: '{tokenizer.decode([idx.item()])}'")
                # tokenizer.decode(idx.item())
                # tokenizer.decode(304)
                # tokenizer.decode(358)
                
        # renormalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        # sample the next token
        next_token = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[0][next_token]
        
        # ddd the token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        print(f"\nGenerated token: '{tokenizer.decode(next_token[0])}'")
        
        # check if we've reached the end of the sequence
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0])

prompt = "Once upon a time"
forbidden_words = ["a", " in", " I", ",", "king", "queen"]
generated_text = generate_with_filter(prompt, max_length=20, forbidden_words=forbidden_words)
print("\nFinal generated text:")
print(generated_text)