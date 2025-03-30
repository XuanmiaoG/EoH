# Enables postponed evaluation of type annotations
from __future__ import annotations
import numpy as np
import os


def dp_knapsack_01(weights: list[float], values: list[float], capacity: float) -> float:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    using a Branch and Bound approach (exact) for float weights/capacity.

    We keep the original function signature, but internally do NOT use DP anymore.

    Parameters
    ----------
    weights : list[float]
        List of item weights (float).
    values : list[float]
        List of item values (float).
    capacity : float
        Total knapsack capacity (float).

    Returns
    -------
    float
        The maximum total value achievable under 0-1 constraints.
    """

    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0

    # 先按价值密度( v_i / w_i ) 从大到小排序，用于估计上界时更紧的剪枝
    items_sorted = [(values[i] / weights[i], i) for i in range(n)]
    items_sorted.sort(key=lambda x: x[0], reverse=True)

    best_value = 0.0  # 全局最优解

    def bound(idx: int, current_val: float, remaining_cap: float) -> float:
        """
        计算从 items_sorted[idx:] 开始能得到的最大潜在价值(上界)，
        使用分数背包思路(能放下就全放，否则放分数)。
        """
        total_val = current_val
        for j in range(idx, n):
            ratio_j, item_idx = items_sorted[j]
            w_j = weights[item_idx]
            v_j = values[item_idx]
            if w_j <= remaining_cap:
                remaining_cap -= w_j
                total_val += v_j
            else:
                total_val += ratio_j * remaining_cap
                break
        return total_val

    def backtrack(idx: int, current_val: float, remaining_cap: float, used_indices: set[int]):
        nonlocal best_value

        # 更新最优解
        if current_val > best_value:
            best_value = current_val

        # 终止条件
        if idx >= n or remaining_cap <= 1e-9:
            return

        # 估计上界
        est = bound(idx, current_val, remaining_cap)
        if est <= best_value:
            return  # 剪枝

        ratio_i, real_idx = items_sorted[idx]
        w_i = weights[real_idx]
        v_i = values[real_idx]

        # 分支1: 不选该物品
        backtrack(idx + 1, current_val, remaining_cap, used_indices)

        # 分支2: 若可装下则选
        if w_i <= remaining_cap:
            used_indices.add(real_idx)
            backtrack(idx + 1, current_val + v_i,
                      remaining_cap - w_i, used_indices)
            used_indices.remove(real_idx)
        else:
            # 装不下则跳过
            pass

    backtrack(0, 0.0, capacity, set())
    return best_value


class GetData:
    """
    A class for loading existing Knapsack datasets and computing 
    their optimal solutions using the dp_knapsack_01 function.
    """

    def __init__(self) -> None:
        """
        Constructor initializing an empty datasets dictionary.
        """
        self.datasets: dict[str, dict[str, float | list[float]]] = {}

    def get_instances(
        self,
        size: str = "100",
        capacity: float | None = None,
        mode: str = "test",
        dataset_path: str = "evaluation_kp/dataset",
    ) -> tuple[dict[str, dict[str, float | list[float]]], dict[str, float]]:
        """
        Load multiple 0-1 Knapsack instances from the existing dataset files,
        and compute the **optimal** solution values for each instance.

        Each instance includes:
            - capacity   (float)      : the knapsack capacity
            - num_items  (int)        : number of items in the instance
            - weights    (list[float]): list of item weights
            - values     (list[float]): list of item values

        :param N:
            A string representing the number of items (e.g., '50', '100', '200', '500').
        :param W:
            (Optional) If provided, this value will be used as the fixed knapsack capacity
            for all instances. If None, capacity is set based on problem size:
              - 12.5 for size '50'
              - 25.0 for other sizes
        :param count:
            (已不再生效) 原本可指定加载多少实例，现在默认会加载文件中的所有实例。
        :param mode:
            Dataset mode: 'train', 'val', or 'test'. The code will look for 
            files named like '{mode}{N}_dataset.npy'.
        :param dataset_path:
            Path to the dataset directory. Default is 'dataset'.

        :return:
            (instances, baseline) as a 2-tuple:
              - instances: dict[str, dict[str, float | list[float]]]
                  { 
                    "instance_name": {
                      "capacity": float,
                      "num_items": int,
                      "weights": list[float],
                      "values": list[float]
                    }
                  }
              - baseline: dict[str, float]
                  { 
                    "instance_name": optimal_value 
                  }
        """
        file_name = f"{mode}{size}_dataset.npy"
        file_path = os.path.join(dataset_path, file_name)

        # 加载数据
        try:
            data = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # 若 W 未指定，根据 N 给默认容量
        if capacity is None:
            if size == "50":
                capacity = 12.5
            else:
                capacity = 25.0

        size_int = int(size)
        instances: dict[str, dict[str, float | list[float]]] = {}
        baseline: dict[str, float] = {}

        # 形状应为 (num_instances, size_int, 2)
        num_instances_in_file = data.shape[0]
        # 直接加载全部，不再 min(...) 任何 count
        n_instances = num_instances_in_file

        for i in range(n_instances):
            instance_name = f"{mode}_{size}_{i+1}"

            # data[i] 是 (size_int, 2)
            weights_arr = data[i, :, 0].tolist()
            values_arr = data[i, :, 1].tolist()

            instance_data = {
                "capacity": capacity,
                "num_items": size_int,
                "weights": weights_arr,
                "values": values_arr,
            }
            instances[instance_name] = instance_data

            # 分支定界 (dp_knapsack_01) 求解
            try:
                optimal_value = dp_knapsack_01(
                    weights_arr, values_arr, capacity)
                baseline[instance_name] = optimal_value
            except Exception as e:
                print(f"[ERROR] {instance_name}: {e}")
                baseline[instance_name] = -1.0

        return instances, baseline


if __name__ == "__main__":
    # 示例：从 evaluation_kp/dataset/train100_dataset.npy 里加载所有实例并求最优解
    data_loader = GetData()
    instances, baseline = data_loader.get_instances(
        size="100",
        capacity=None,  # 默认: N!=50 -> 容量25.0
        mode="train",
        dataset_path="evaluation_kp/dataset",
    )

    # 打印首个实例的信息
    if instances:
        first_instance_name = next(iter(instances))
        inst_data = instances[first_instance_name]
        print(f"\nInstance: {first_instance_name}")
        print(f"  capacity: {inst_data['capacity']}")
        print(f"  num_items: {inst_data['num_items']}")
        print(f"  first 5 weights: {inst_data['weights'][:5]}")
        print(f"  first 5 values: {inst_data['values'][:5]}")
        print(f"  optimal_value: {baseline[first_instance_name]}")
    else:
        print("No instances loaded.")
