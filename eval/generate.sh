python generate_completions.py \
    --input_file /root/project/sudoku-grpo/data/sudoku_4x4_qa.jsonl \
    --output_file /root/project/sudoku-grpo/eval/data/qwen-7b-chat-temperature0.jsonl \
    --num_completions_to_generate 300 \
    --model_name /root/project/sudoku-grpo/models/qwen2.5-0.5b \
    --temperature 0 \
    --top_p 1.0