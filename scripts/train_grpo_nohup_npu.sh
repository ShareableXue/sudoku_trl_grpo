# nohup accelerate launch --config_file configs/multi_npu.yaml train_grpo.py --config configs/grpo_qwen2.5-3b-it_lora.yaml > run.log 2>&1 &
# echo $! > run.pid
# echo "训练已启动，PID: $(cat run.pid)"

nohup accelerate launch --config_file configs/multi-npu_dsz2.yaml train_grpo.py --config configs/grpo_qwen2.5-3b-it_lora.yaml > run.log 2>&1 &
echo $! > run.pid
echo "训练已启动，PID: $(cat run.pid)"