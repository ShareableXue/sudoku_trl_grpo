import random
import os
import json
import tqdm
import re

import dataclasses
import logging
import os
import json
from typing import Sequence, Union
import ast

import openai

import argparse


## 提示词设置
PROMPT_TEMPLATE = """请作为数独专家完成4x4数独题目。数独题目中已给出数字为限定条件，“_”表示待填入1-4的空白位置。
## 数独规则
- 根据数独矩阵中已给出的数字推测“_”处的数字
- 每行：数字1-4各出现一次
- 每列：数字1-4各出现一次  
- 每个2×2宫格：数字1-4各出现一次
## 解题流程
1. 选择一处空白位置“_”，按行、列、宫格分析应该填入的数字
2. 打印填补一处空白的矩阵，判断该次填充是否正确
3. 重复流程1，直至所有空白位置“_”被填完。
## 输出格式
输出格式为<think>...</think>\n<answer>...</answer>
在<think>...</think>中填入思考过程
在<answer>...</answer>中填写python List格式的最终矩阵
输出格式案例：
<think>
详细推理过程，包括：
- 一步步的思考过程
- 对题目的复述，以及要分析的空白位置
- 填入空白位置的数字以及理由，按行、列、宫格规则分析
- 检查填入后的数独矩阵，确保符合数独规则，确保没有修改题目限定条件
</think>
<answer>[[1,2,3,4],[4,3,...],...]</answer>
## 数独题目：
{}"""

SUDOKU_FORMAT = "[[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]"

### openai接口
client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    client.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1

### chat生成数据
def openai_completion(
    model_path:str,
    prompt: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
):
    message=[
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model_path,
        messages=message,
        max_tokens=decoding_args.max_tokens,
        temperature=decoding_args.temperature,
        top_p=decoding_args.top_p,
    )

    return completion

# 生成答案，并且将回答的内容保存到一个output.jsonl文件中
def generate_completions(
        output_file: str = "./data/output_completions.jsonl",
        input_file: str = "./data/sudoku_4x4_qa.jsonl",
        num_completions_to_generate: int = 300,
        model_name: str = "/root/autodl-tmp/outputs/merged-model-gpuvllm",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1648,
):
    print("Starting completion generation...")
    os.makedirs("./data", exist_ok=True)

    # 1. 准备解码参数
    decoding_args = OpenAIDecodingArguments(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 2. 从文件加载并准备 prompts
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    simple_data = [item for item in all_data if item.get('label') == 'simple']
    # 生成num_completions_to_generate条数据，仅考虑simple
    if len(simple_data) < num_completions_to_generate:
        num_to_generate = len(simple_data)
    else:
        num_to_generate = num_completions_to_generate
        
    random.seed(42)
    sampled_data = random.sample(simple_data, num_to_generate)
    print("总共数据数量：",len(sampled_data))

    # 逐行写入jsonl文件
    with open(output_file, "w", encoding="utf-8") as f:
        for data in tqdm.tqdm(sampled_data, desc="Generating completions"):
            # 预处理提示词数据
            sudo_matrix = SUDOKU_FORMAT.format(*[c for c in data["question"]])
            prompt = PROMPT_TEMPLATE.format(sudo_matrix)

            # 3. 调用 vLLM 生成答案
            completion = openai_completion(
                prompt=prompt,
                model_path=model_name,
                decoding_args=decoding_args,
            )
            # completion=completion.choices[0].text
            completion=completion.choices[0].message.content
            # 4. 整理并保存结果
            results={
                    "prompt": prompt,
                    "question": data["question"],
                    "completion": completion,
                    }
            f.write(json.dumps(results, ensure_ascii=False) + "\n")

    print(f"Successfully saved completions to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,help="input data path")
    parser.add_argument("--output_file", type=str,help="output data path")
    parser.add_argument("--num_completions_to_generate", type=int, default=300, help="number of completions to generate")
    parser.add_argument("--model_name", type=str, help="model name or path")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1648, help="maximum number of new tokens to generate")
    args = parser.parse_args()

    generate_completions(
        input_file=args.input_file,
        output_file=args.output_file,
        num_completions_to_generate=args.num_completions_to_generate,  # 测试时可适当减小数量
        model_name=args.model_name,  # 替换为vLLM加载的模型名
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )




