nohup accelerate launch --config_file config/multi_npu.yaml train_sft.py  > run.log 2>&1 &
echo $! > run.pid
echo "训练已启动，PID: $(cat run.pid)"