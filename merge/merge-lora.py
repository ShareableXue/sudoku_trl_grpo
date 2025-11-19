import os
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import argparse

def merge_lora_model(lora_path,base_model_path,output_path):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="bfloat16",      # 保持和原模型一致
        device_map="cpu"         # 合并阶段放 CPU 省显存
    )

    # Load the LoRA model
    lora_model = PeftModel.from_pretrained(
        base_model, 
        lora_path)

    # Merge LoRA weights into the base model
    merged_model = lora_model.merge_and_unload()

    # Load tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Save the merged model
    os.makedirs(output_path, exist_ok=True)
    
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save the merged model.")
    parser.add_argument("--lora_path", required=True, type=str, help="LoRA 适配器目录路径")
    parser.add_argument("--base_model_path", required=True, type=str, help="原始模型目录路径")
    parser.add_argument("--merge_path", required=True, type=str, help="合并后模型保存目录路径")
    args = parser.parse_args()

    merge_lora_model(args.lora_path, args.base_model_path, args.merge_path)
    