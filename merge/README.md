# 合并模型参数

```bash
python ./merge/merge-lora.py \
        --lora_path /root/autodl-tmp/models/outputs/grpo_qwen2.5-7b-it_lora \
        --base_model_path /root/autodl-tmp/models/qwen/qwen2.5-7b-it \
        --merge_path /root/autodl-tmp/models/outputs/merged-7b-gpu
```