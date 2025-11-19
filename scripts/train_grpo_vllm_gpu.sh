accelerate launch \
    --num_processes 4 \
    --config_file /root/project/sudoku-grpo/configs/deepspeed_zero3.yaml \
    /root/project/sudoku-grpo/train_grpo.py \
    --config /root/project/sudoku-grpo/configs/lxy_grpo_qwen2.5-7b-it_lora.yaml
