import json
import random
from typing import List

# 批量生成数据，要求如下：
# 1. 先生成完整的数独结果，总共应该288个
# 2. 生成对应的question，question是针对answer挖空后要求填写的数独游戏的面板
# 3. 添加提示词模板，补充数独游戏规则

#############################################
# 1. 生成数独结果(共288条符合规则的4x4数独终盘)
#############################################
# 4×4数独核心配置
GRID_SIZE = 4  # 数独网格：4行4列
BOX_SIZE = 2  # 小宫格：2×2（4×4网格分为4个2×2小宫格）


def sudoku_answer_generation() -> List[List[int]]:
    """
    生成完整的4×4数独终盘（满足行/列/2×2宫格无重复数字）
    Returns:
        List[List[int]]: 4×4数独终盘，每个元素为1-4的整数（0表示未填充）
    example:
    [
        [1, 3, 4, 2],
        [2, 4, 1, 3],
        [3, 1, 2, 4],
        [4, 2, 3, 1]
    ]
    """
    # 1. 初始化4×4空白网格（0表示未填充）
    sudoku = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    def is_num_used_in_row(row: int, num: int) -> bool:
        """检查数字num是否已在第row行存在"""
        return num in sudoku[row]

    def is_num_used_in_col(col: int, num: int) -> bool:
        """检查数字num是否已在第col列存在"""
        # 遍历每一行，查看当前列是否有目标数字
        for row in range(GRID_SIZE):
            if sudoku[row][col] == num:
                return True
        return False

    def is_num_used_in_box(row: int, col: int, num: int) -> bool:
        """检查数字num是否已在(row, col)所在的2×2小宫格中存在"""
        # 计算当前小宫格的左上角坐标（核心：2×2宫格的起始位置）
        box_start_row = (row // BOX_SIZE) * BOX_SIZE  # 宫格起始行（0或2）
        box_start_col = (col // BOX_SIZE) * BOX_SIZE  # 宫格起始列（0或2）

        # 遍历小宫格内的2×2个单元格
        for r in range(box_start_row, box_start_row + BOX_SIZE):
            for c in range(box_start_col, box_start_col + BOX_SIZE):
                if sudoku[r][c] == num:
                    return True
        return False

    def can_place_num(row: int, col: int, num: int) -> bool:
        """判断数字num能否放在(row, col)位置（满足数独规则）"""
        # 三个条件同时满足：行未用、列未用、宫格未用
        return (
            not is_num_used_in_row(row, num)
            and not is_num_used_in_col(col, num)
            and not is_num_used_in_box(row, col, num)
        )

    def fill_grid(pos: int) -> bool:
        """递归填充网格：pos是当前要填充的「线性位置」（0-15，对应4×4=16个单元格）"""
        # 终止条件：所有16个单元格都填充完成，返回成功
        if pos == GRID_SIZE * GRID_SIZE:
            return True

        # 2. 把线性位置pos转换成网格的（行号，列号）
        # 例：pos=5 → 5//4=1（第2行），5%4=1（第2列）→ (1,1)
        current_row, current_col = divmod(pos, GRID_SIZE)

        # 3. 如果当前单元格已填充（非0），直接跳过，处理下一个位置
        if sudoku[current_row][current_col] != 0:
            return fill_grid(pos + 1)

        # 4. 随机打乱1-4的顺序（确保每次生成的数独不同）
        random_nums = random.sample(range(1, GRID_SIZE + 1), GRID_SIZE)

        # 5. 尝试给当前单元格填充每个可能的数字
        for num in random_nums:
            # 检查数字是否可以放置
            if can_place_num(current_row, current_col, num):
                # 放置数字（做出选择）
                sudoku[current_row][current_col] = num

                # 递归填充下一个单元格（pos+1）
                # 如果后续所有单元格都填充成功，直接返回True（终止递归）
                if fill_grid(pos + 1):
                    return True

                # 回溯：如果后续填充失败，撤销当前选择（恢复为0）
                sudoku[current_row][current_col] = 0

        # 6. 所有数字都尝试过仍失败，返回False，让上一层换数字重试
        return False

    # 从第0个位置开始填充网格
    fill_grid(pos=0)
    return sudoku


# 检查生成的数独数据集是否符合规则
def is_valid_sudoku(grid):
    """检查4x4数独是否有效"""
    # 检查每行
    for row in grid:
        if sorted(row) != [1, 2, 3, 4]:
            return False

    # 检查每列
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if sorted(column) != [1, 2, 3, 4]:
            return False

    # 检查每个2x2宫格
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            box = [grid[box_row + i][box_col + j] for i in range(2) for j in range(2)]
            if sorted(box) != [1, 2, 3, 4]:
                return False

    return True


# 生成10万条数据，确保能覆盖所有的数独的结果
def build_dataset(num: int = 100000, output_path: str = "./data/answer/answer.jsonl"):
    """生成 num 条 4×4 终盘，写入 jsonl"""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 生成并写入10万条数据到jsonl文件中，同时检查是否符合数独游戏规则
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num):
            # 生成数独结果（随机）
            grid = sudoku_answer_generation()
            # 检查是否符合数独游戏规则
            if not is_valid_sudoku(grid):
                print("generated an invalid sudoku!")
                continue

            # 整理输出格式，从List[list(int)]到str
            def format_sudoku_compact(grid):
                """
                将 4×4 数独转成紧凑 4 行字符串：
                1234341221434321
                """
                return "".join("".join(str(cell) for cell in row) for row in grid)

            compact = format_sudoku_compact(grid)
            # 保存到jsonl文件中
            f.write(json.dumps({"answer": compact}, ensure_ascii=False) + "\n")
    print(f"✅ 已生成 {num} 条数据 -> {output_path}")

    # 检查是否有重复数据
    def remove_duplicates(output_path: str = "./data/answer2.jsonl"):
        """检测并删除./data/answer.jsonl中的重复数据"""
        # 读取现有的answer.jsonl文件
        answers = []
        unique_answers = set()
        duplicates_count = 0

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                answer_obj = json.loads(line)
                answer_str = answer_obj["answer"]

                # 检查是否重复
                if answer_str not in unique_answers:
                    unique_answers.add(answer_str)
                    answers.append(answer_obj)
                else:
                    duplicates_count += 1

        # 将去重后的数据重新写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            for answer_obj in answers:
                f.write(json.dumps(answer_obj) + "\n")

        print(
            f"检测完成：发现并删除了 {duplicates_count} 条重复数据，保留了 {len(answers)} 条唯一数据"
        )
        return len(answers)

    remove_duplicates(output_path)


##############################################################################
# 2. 生成对应的question，question是针对answer挖空后要求填写的数独游戏的面板
##############################################################################
# 针对每一条answer数据来生成question随机挖空的数独问题
def create_question(grid, num_clues):
    """
    创建数独问题，保留指定数量的线索，严格确保每行、每列、每个2x2宫格至少保留一个数字
    grid: 数独矩阵：List[List[int]]
    num_clues: 保留的线索数量：int
    """
    # 创建一个深拷贝，避免修改原始网格
    question_grid = [row[:] for row in grid]

    # 计算总单元格数
    total_cells = 16
    # 需要挖空的单元格数
    cells_to_remove = total_cells - num_clues

    # 多次尝试，以避免因约束导致无法达到指定保留数量
    max_attempts = 50
    for _ in range(max_attempts):
        # 初始化计数：每行、每列、每个2x2宫格当前保留的数量
        row_counts = [4] * 4
        col_counts = [4] * 4
        box_counts = [4] * 4  # box 索引使用 (i//2)*2 + (j//2)
        removed = set()

        # 随机顺序遍历所有单元格，尝试挖空
        positions = [(i, j) for i in range(4) for j in range(4)]
        random.shuffle(positions)

        for i, j in positions:
            if len(removed) == cells_to_remove:
                break
            b = (i // 2) * 2 + (j // 2)
            # 不能挖空导致任何一行/列/宫计数归零
            if row_counts[i] <= 1 or col_counts[j] <= 1 or box_counts[b] <= 1:
                continue
            # 执行挖空
            removed.add((i, j))
            row_counts[i] -= 1
            col_counts[j] -= 1
            box_counts[b] -= 1

        # 如果成功挖空到目标数量，则应用更改并返回
        if len(removed) == cells_to_remove:
            for i, j in removed:
                question_grid[i][j] = "_"
            return question_grid

    # 若在多次尝试后仍无法达到目标（极少发生），尽可能多地挖空但仍保持约束
    row_counts = [4] * 4
    col_counts = [4] * 4
    box_counts = [4] * 4
    positions = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(positions)
    removed = set()
    for i, j in positions:
        b = (i // 2) * 2 + (j // 2)
        if row_counts[i] <= 1 or col_counts[j] <= 1 or box_counts[b] <= 1:
            continue
        removed.add((i, j))
        row_counts[i] -= 1
        col_counts[j] -= 1
        box_counts[b] -= 1
    for i, j in removed:
        question_grid[i][j] = "_"
    return question_grid


# 生成包含question的数据集
def create_question_dataset(input_path, output_path):
    # 读取answer.jsonl文件
    answers = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            answers.append(json.loads(line))

    # 设置难度级别及对应的线索数
    """difficulty_levels = [
        {"label": "simple", "clues_range": (12, 14)},    # 保留12-14个数字
        {"label": "medium", "clues_range": (8, 10)},     # 保留8-10个数字
        {"label": "difficult", "clues_range": (4, 6)},   # 保留4-6个数字
    ]"""

    difficulty_levels = [
        {"label": "simple", "clues_range": (12, 12)},    # 保留12-14个数字
        {"label": "medium", "clues_range": (8, 8)},     # 保留8-10个数字
        {"label": "difficult", "clues_range": (4, 4)},   # 保留4-6个数字
    ]

    data, used = [], set()

    # 3. 为每个答案生成题目
    for ans_obj in answers:
        ans_str = ans_obj["answer"]  # "1234341221434321"
        # 先转成 int 二维数组，供 create_question 做约束检查，每四个组成一个矩阵
        grid = [[int(ch) for ch in ans_str[i : i + 4]] for i in range(0, 16, 4)]

        for diff in difficulty_levels:
            label, clues_range = diff["label"], diff["clues_range"]
            generated, attempts = 0, 0
            max_attempts = 20

            while generated < 2 and attempts < max_attempts:
                # 3.1 挖空（返回仍是二维数组）
                q_grid = create_question(grid, random.randint(*clues_range))

                # 3.2 再变回紧凑字符串
                q_str = "".join("".join(str(cell) for cell in row) for row in q_grid)

                if q_str not in used:
                    used.add(q_str)
                    data.append({"question": q_str, "answer": ans_str, "label": label})
                    generated += 1

                attempts += 1

            if generated < 2:
                print(
                    f"警告：答案 {ans_str[:15]}... 的难度 {label} 仅生成 {generated} 个不同问题"
                )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"生成包含question的数据集：{len(data)} 个数独问题，已保存到 {output_path}")


if __name__ == "__main__":
    # 如果不存在data文件夹，则创建
    import os
    if not os.path.exists("data"):
        os.makedirs("data")

    answer_path = "./data/sudoku_4x4_answer.jsonl"
    question_path = "./data/sudoku_4x4_qa.jsonl"

    random.seed(42)

    build_dataset(num=100000, output_path=answer_path)
    create_question_dataset(input_path=answer_path, output_path=question_path)
