from __future__ import annotations  # Enables postponed evaluation of type annotations
import numpy as np


def dp_knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    via dynamic programming, using a 1D DP array.

    Let:
      - n = number of items
      - W = capacity of the knapsack
      - w_i = weight of item i
      - v_i = value of item i

    We define the DP array as:
        dp[c] = maximum possible value achievable using capacity c

    Transition Formula (iterating in reverse order for capacity):
        dp[c] = max( dp[c], dp[c - w_i] + v_i ),  if c >= w_i

    The time complexity is O(n * W).

    :param weights: list of item weights
    :param values: list of item values
    :param capacity: total knapsack capacity
    :return: the maximum total value achievable
    """
    n: int = len(weights)
    dp: list[int] = [0] * (capacity + 1)

    for i in range(n):
        w_i: int = weights[i]
        v_i: int = values[i]
        # Iterate capacity in reverse to avoid using updated dp values prematurely
        for c in range(capacity, w_i - 1, -1):
            dp[c] = max(dp[c], dp[c - w_i] + v_i)

    return dp[capacity]


class GetData:
    """
    A class for generating random 0-1 Knapsack instances and computing
    their optimal solutions using the dp_knapsack_01 function.
    """

    def __init__(self) -> None:
        """
        Constructor initializing an empty datasets dictionary.
        """
        self.datasets: dict[str, dict[str, int | list[int]]] = {}

    def get_instances(
        self,
        size: str = "500",
        capacity: int | None = None,
        count: int = 50,
    ) -> tuple[dict[str, dict[str, int | list[int]]], dict[str, int]]:
        """
        Generate multiple 0-1 Knapsack instances with a given number of items,
        and compute the **optimal** solution values for each instance.

        Each instance includes:
            - capacity   (int)        : the knapsack capacity
            - num_items  (int)        : number of items in the instance
            - weights    (list[int])  : list of item weights
            - values     (list[int])  : list of item values

        :param size:
            A string representing the number of items (e.g., '50', '100', '200', '500').
        :param capacity:
            (Optional) If provided, this integer will be used as the fixed knapsack capacity
            for all instances. If None, the capacity is set to floor(0.5 * sum_of_weights).
        :param count:
            Number of instances to generate for the given size. Default is 50.

        We generate random weights in [1, 100] and values in [10, 200].
        If capacity is None, the capacity is set to floor(0.5 * sum_of_weights).

        :return:
            (instances, baseline) as a 2-tuple:
              - instances (dict[str, dict[str, int | list[int]]]):
                  A dictionary of generated instances.
                  Each key is an instance name, each value is a dictionary containing:
                    {
                      "capacity": int,
                      "num_items": int,
                      "weights": list[int],
                      "values": list[int]
                    }
              - baseline (dict[str, int]):
                  A dictionary mapping each instance name to the optimal solution value.
        """
        # Set a fixed random seed for reproducibility
        np.random.seed(2025)

        size_int: int = int(size)
        instances: dict[str, dict[str, int | list[int]]] = {}
        baseline: dict[str, int] = {}

        for i in range(1, count + 1):
            instance_name: str = f"test_{i}"

            # Generate random weights in [1..100] and values in [10..200]
            weights: list[int] = np.random.randint(1, 101, size_int).tolist()
            values_arr: list[int] = np.random.randint(10, 201, size_int).tolist()

            # Compute capacity: if None, use 50% of the total weight; otherwise, use provided capacity (at least 1)
            if capacity is None:
                cap: int = int(0.5 * sum(weights))
            else:
                cap = max(1, capacity)

            # Build the instance data
            instance_data: dict[str, int | list[int]] = {
                "capacity": cap,
                "num_items": size_int,
                "weights": weights,
                "values": values_arr,
            }
            instances[instance_name] = instance_data

            # Compute the optimal solution via DP
            optimal_value: int = dp_knapsack_01(weights, values_arr, cap)
            baseline[instance_name] = optimal_value

        return instances, baseline
