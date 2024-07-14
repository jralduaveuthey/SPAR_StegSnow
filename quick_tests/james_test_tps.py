# %%
import transformers
import torch
from time import perf_counter
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


# %%
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_dir = "saved_models/meta-llama_Meta-Llama-3-8B-Instruct"


# Check if the model is saved locally
if os.path.exists(local_model_dir): #NOTE: this seems to make not much difference on the speed (at least for 8B) so maybe not worth it pre-saving the models
    print(f"Loading model from local directory: {local_model_dir}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(local_model_dir, "model"),
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(local_model_dir, "tokenizer"))
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
else:
    print(f"Local model not found. Downloading model: {model_id}")
    hf_token = os.getenv("HF_TOKEN")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=hf_token
    )

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

token_total = 0
t1_start = perf_counter() 

for _ in tqdm(range(100)):
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    token_total += len(outputs[0]["generated_text"][-1]['content'])
t1_stop = perf_counter()
print(f"Got {token_total} tokens in {t1_stop-t1_start} seconds, {token_total/(t1_stop-t1_start)} tokens per second.")
# %%