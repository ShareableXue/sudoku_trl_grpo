## 生成数据

1. 启动vllm服务

```bash
vllm serve /root/autodl-tmp/models/qwen2.5-3b-it
```

2. 生成数据

```bash
bash generate.sh
```

## 测试

1. 修改eval中数据集地址

2. 运行脚本

```bash
python eval.py
```