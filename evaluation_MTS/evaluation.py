import numpy as np
import types
import warnings
import sys

# 从 get_instance 模块中加载所需组件
from .get_instance import (
    GetData,
    calc_fitness,
    place_facilities_quantiles,  # 如有需要直接调用
)


class MLS_Test:
    """
    针对多设施选址问题的评估类，仅使用 test 数据进行评估。

    流程：
      1. 通过 GetData 加载数据时只使用 test 数据；
      2. evaluateHeuristic 方法要求用户模块提供 place_facilities(peaks, weights, k) 函数，
         依次对所有实例计算 fitness（加权社会成本 + 罚分）；
      3. evaluate 方法接收用户提供的代码字符串，
         将其动态加载后调用 evaluateHeuristic 得到平均 fitness（越低越好）。
    """

    def __init__(self, epsilon=0.01, k=2):
        self.epsilon = epsilon
        self.k = k
        getdata = GetData()
        # 只取 test 数据
        _, test_data = getdata.get_instances()
        self.instances = test_data
        self.prompts = GetPrompts()

    def evaluateHeuristic(self, alg) -> float:
        """
        对所有 test 实例使用用户提供的 place_facilities(...) 函数进行评估，
        并返回所有样本 fitness（平均值）。
        """
        # 尝试从候选模块中获取 place_facilities 函数
        place_func = getattr(alg, "place_facilities", None)
        if place_func is None:
            raise ValueError(
                "候选模块中未定义 'place_facilities(peaks, weights, k)' 函数。"
            )

        total_fitness = 0.0
        total_samples = 0

        # 遍历所有 test 数据实例
        for key, val in self.instances.items():
            if "peaks" not in val:
                continue
            peaks_arr = val["peaks"]  # 形状 (num_samples, n_agents)
            misr_arr = val.get("misreports", None)
            weights_info = val.get("weights", None)
            k_local = self.k  # 可扩展为使用实例内部的 k

            # 简单检查 peaks 数组维度
            if len(peaks_arr.shape) != 2:
                print(f"Warning: {key} 中 'peaks' 形状非二维，跳过。")
                continue

            num_samples, n_agents = peaks_arr.shape

            for i in range(num_samples):
                peaks_i = peaks_arr[i]
                # 处理权重数据：可能是一维（全局）或二维（每个样本一行）
                if weights_info is not None:
                    if len(weights_info.shape) == 1:
                        weights_i = weights_info
                    elif len(weights_info.shape) == 2:
                        weights_i = weights_info[i]
                    else:
                        print(
                            f"Warning: {key} 中 weights 形状异常 {weights_info.shape}，采用均匀权重。"
                        )
                        weights_i = np.ones(n_agents, dtype=float)
                else:
                    weights_i = np.ones(n_agents, dtype=float)

                # 如果存在 misreports 且为三维，则取出当前样本的数据
                misreports_i = None
                if misr_arr is not None and len(misr_arr.shape) == 3:
                    if (
                        misr_arr.shape[0] == num_samples
                        and misr_arr.shape[1] == n_agents
                    ):
                        misreports_i = misr_arr[i]
                    else:
                        print(
                            f"Warning: {key} 中 misreports 形状不匹配：{misr_arr.shape}"
                        )

                # 计算当前样本的 fitness
                fitness_i = calc_fitness(
                    peaks_i,
                    misreports_i,
                    weights_i,
                    place_func,
                    k_local,
                    epsilon=self.epsilon,
                )
                total_fitness += fitness_i
                total_samples += 1

        if total_samples == 0:
            print("Warning: 未找到有效样本，返回 9999。")
            return 9999.0

        return total_fitness / total_samples

    def evaluate(self, code_string):
        """
        对用户提供的代码字符串进行评估。
        代码中必须定义如下函数：

            def place_facilities(peaks, weights, k):
                ...
                return np.array([...])

        该方法将动态加载此代码并调用 evaluateHeuristic 得到平均 fitness。
        """
        try:
            print("正在评估用户提供的多设施选址代码:\n", code_string)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateHeuristic(heuristic_module)
                return fitness

        except Exception as e:
            print("evaluate 出现错误:", e)
            return None
