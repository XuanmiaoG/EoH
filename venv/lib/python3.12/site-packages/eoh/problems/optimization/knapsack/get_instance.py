import numpy as np


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
            baseline (dict): A dictionary mapping each instance name to a baseline total value
                             computed by the greedy ratio approach (value/weight).
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

            # Compute baseline using the classic greedy ratio method (value/weight)
            items = [{"weight": w, "value": v} for w, v in zip(weights, values_arr)]
            items_sorted = sorted(
                items, key=lambda x: x["value"] / x["weight"], reverse=True
            )

            remaining = cap
            baseline_value = 0
            for item in items_sorted:
                if item["weight"] <= remaining:
                    baseline_value += item["value"]
                    remaining -= item["weight"]

            baseline[instance_name] = baseline_value

        return instances, baseline
