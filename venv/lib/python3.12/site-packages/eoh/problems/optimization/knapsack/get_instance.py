import numpy as np

class GetData():
    def __init__(self) -> None:
        self.datasets = {}

    def get_instances(self, size: str = '50', capacity: int = 100) -> (dict, dict):
        """
        Generate a set of knapsack instances with a given number of items.
        We choose four sizes (e.g. '50', '100', '200', '500').
        For each instance, items are generated with random weights (1–50) and values (10–100).
        The capacity is set to roughly 50% of the sum of weights.
        Also compute a baseline value using the classic greedy (value/weight) method.
        Returns a tuple: (instances, lb) where lb is a dict mapping instance name to its baseline total value.
        """
        size_int = int(size)
        instances = {}
        lb = {}
        # Here we generate 3 instances per size for evaluation.
        for i in range(1, 4):
            instance_name = f"test_{i}"
            weights = np.random.randint(1, 51, size_int).tolist()
            values_arr = np.random.randint(10, 101, size_int).tolist()
            cap = int(0.5 * sum(weights))
            instance_data = {
                "capacity": cap,
                "num_items": size_int,
                "weights": weights,
                "values": values_arr
            }
            instances[instance_name] = instance_data
            # Compute baseline using greedy by ratio (value/weight)
            items = [{'weight': w, 'value': v} for w, v in zip(weights, values_arr)]
            items_sorted = sorted(items, key=lambda x: x['value'] / x['weight'], reverse=True)
            remaining = cap
            baseline_value = 0
            for item in items_sorted:
                if item['weight'] <= remaining:
                    baseline_value += item['value']
                    remaining -= item['weight']
            lb[instance_name] = baseline_value
        return instances, lb
