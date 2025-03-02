import numpy as np


def dp_knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    via dynamic programming.

    This uses a 1D DP array of size (capacity+1), running in O(n * capacity) time,
    where n = len(weights).

    :param weights: list of item weights
    :param values: list of item values
    :param capacity: total knapsack capacity
    :return: the maximum total value achievable
    """
    n = len(weights)
    dp = [0] * (capacity + 1)  # dp[c] = best possible value with capacity c

    for i in range(n):
        w = weights[i]
        v = values[i]
        # We iterate capacity in reverse to avoid overwriting data we still need
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)

    return dp[capacity]


class GetData:
    def __init__(self) -> None:
        self.datasets = {}

    def get_instances(
        self, size: str = "50", capacity: int = None
    ) -> tuple[dict, dict]:
        """
        Generate a set of 0-1 Knapsack instances with a given number of items.

        :param size: A string representing the number of items (e.g., '50', '100', '200', '500').
        :param capacity: (Optional) If provided, this integer will be used as a fixed knapsack capacity
                         for all instances. If None, the capacity is set to 50% of sum of the item weights.

        We generate random weights in [1,50] and values in [10,100].
        If capacity is None, the capacity is set to floor(0.5 * sum_of_weights).

        Returns:
            instances (dict): A dictionary of generated instances.
                              Each key is an instance name, and each value is another dict with:
                                - "capacity": the knapsack capacity (int)
                                - "num_items": number of items (int)
                                - "weights": list of item weights (list of int)
                                - "values": list of item values (list of int)
            baseline (dict): A dictionary mapping each instance name to the
                             **optimal** solution value (via DP).
        """
        size_int = int(size)
        instances = {}
        baseline = {}

        # Generate 3 instances for each 'size' for demonstration/testing
        for i in range(1, 4):
            instance_name = f"test_{i}"

            # Randomly generate weights and values
            weights = np.random.randint(1, 51, size_int).tolist()  # [1..50]
            values_arr = np.random.randint(10, 101, size_int).tolist()  # [10..100]

            # If user doesn't provide a capacity, use 50% of total weight
            if capacity is None:
                cap = int(0.5 * sum(weights))
            else:
                # Otherwise, use the provided capacity (must be > 0)
                cap = max(1, capacity)

            # Build the instance data
            instance_data = {
                "capacity": cap,
                "num_items": size_int,
                "weights": weights,
                "values": values_arr,
            }
            instances[instance_name] = instance_data

            # Compute the OPTIMAL baseline using DP for 0-1 Knapsack
            optimal_value = dp_knapsack_01(weights, values_arr, cap)
            baseline[instance_name] = optimal_value

        return instances, baseline
