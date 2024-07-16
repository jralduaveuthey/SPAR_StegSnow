"""
To download models from HF in my local machine. To speed up experiments.
Atm not so useful since Lamma3 70B does not fit in my local machine a-100.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def download_and_save_model(model_name, save_dir, model_type="causal"):
    print(f"Downloading and saving model: {model_name}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    
    # Download and save model
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
    model.save_pretrained(os.path.join(save_dir, "model"))
    
    print(f"Model and tokenizer saved in {save_dir}")

def list_saved_models(base_dir):
    print("Saved models:")
    saved_models = []
    for model_dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, model_dir)):
            print(f"- {model_dir}")
            saved_models.append(model_dir)
    return saved_models

def interactive_mode():
    base_dir = input("Enter the base directory for saving models (default: saved_models): ").strip() or "saved_models"
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Download a new model")
        print("2. List saved models")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            model_name = input("Enter the model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct): ").strip()
            model_type = input("Is this a causal language model? (y/n, default: y): ").strip().lower()
            model_type = "causal" if model_type != "n" else "other"
            
            save_dir = os.path.join(base_dir, model_name.replace("/", "_"))
            download_and_save_model(model_name, save_dir, model_type)
        
        elif choice == "2":
            list_saved_models(base_dir)
        
        elif choice == "3":
            print("Exiting. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save Hugging Face models locally.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", type=str, help="Name of the model to download")
    parser.add_argument("--dir", type=str, default="saved_models", help="Base directory to save models")
    parser.add_argument("--type", type=str, default="causal", choices=["causal", "other"], help="Type of the model")
    parser.add_argument("--list", action="store_true", help="List saved models")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.list:
        list_saved_models(args.dir)
    elif args.model:
        save_dir = os.path.join(args.dir, args.model.replace("/", "_"))
        download_and_save_model(args.model, save_dir, args.type)
    else:
        print("Please use --interactive for guided mode or specify a model to download.")