from __future__ import annotations
import importlib
import types
import sys
import warnings

# 引入 MLS_Test 类（假设该类代码存放在 mls_test.py 文件中）
from evaluation import MLS_Test


def main() -> None:
    """
    主程序：
      1. 创建 MLS_Test 实例（仅加载 test 数据）。
      2. 动态加载并重载候选算法模块 "heuristic"。
      3. 调用 evaluateHeuristic 计算所有测试实例的平均 fitness。
      4. 打印评估结果。
    """
    # 创建评估器实例（加载测试数据）
    evaluator: MLS_Test = MLS_Test(epsilon=0.01, k=2)

    # 动态加载候选算法模块 "heuristic"
    heuristic_module: types.ModuleType = importlib.import_module("heuristic")
    heuristic_module = importlib.reload(heuristic_module)

    # 调用评估器对所有测试实例进行评估
    avg_fitness: float = evaluator.evaluateHeuristic(heuristic_module)

    # 输出评估结果（fitness 越低越好）
    print(f"Average fitness on test instances: {avg_fitness:.4f}")


if __name__ == "__main__":
    main()
