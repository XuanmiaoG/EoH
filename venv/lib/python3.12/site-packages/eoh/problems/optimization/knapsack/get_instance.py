import numpy as np

class GetData():
    def __init__(self) -> None:
        self.datasets = {}

    def get_instances(self, size: str = '50', capacity: int = 100) -> (dict, dict):
        """
        Return hard-coded knapsack instances.
        This version only supports size '50'.
        Each instance contains 50 items with fixed weights and values.
        The knapsack capacity is set to 50% of the total weight.
        A baseline total value is computed via a greedy strategy (by value/weight ratio).
        
        Returns:
            instances: dict mapping instance name to its instance data.
            lb: dict mapping instance name to its baseline total value.
        """
        if size != '50':
            raise ValueError("Hard-coded instances only available for size '50'")
        
        instances = {}
        lb = {}
        
        # ----- Instance 1 -----
        weights1 = [10, 20, 15, 12, 18, 25, 30, 8, 9, 11,
                    13, 17, 22, 5, 6, 7, 12, 14, 15, 18,
                    20, 22, 24, 10, 8, 11, 15, 12, 14, 9,
                    13, 16, 19, 21, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        values1  = [50, 60, 55, 40, 80, 90, 100, 30, 35, 45,
                    50, 65, 70, 20, 25, 30, 45, 55, 60, 75,
                    80, 85, 90, 50, 45, 40, 55, 50, 60, 35,
                    50, 65, 70, 75, 30, 35, 40, 45, 50, 55,
                    60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
        cap1 = int(0.5 * sum(weights1))
        instance1 = {
            "capacity": cap1,
            "num_items": 50,
            "weights": weights1,
            "values": values1
        }
        instances["test_1"] = instance1
        lb["test_1"] = self.compute_baseline(instance1)
        
        # ----- Instance 2 -----
        weights2 = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                    10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                    11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                    31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
                    50, 52, 54, 56, 58, 60, 62, 64, 66, 68]
        values2  = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
                    50, 45, 40, 35, 30, 25, 20, 15, 10, 5,
                    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                    105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
                    155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
        cap2 = int(0.5 * sum(weights2))
        instance2 = {
            "capacity": cap2,
            "num_items": 50,
            "weights": weights2,
            "values": values2
        }
        instances["test_2"] = instance2
        lb["test_2"] = self.compute_baseline(instance2)
        
        # ----- Instance 3 -----
        weights3 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                    4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
                    44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
                    84, 88, 92, 96, 100, 104, 108, 112, 116, 120]
        values3  = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                    75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
                    20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                    70, 75, 80, 85, 90, 95, 100, 105, 110, 115,
                    120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
        cap3 = int(0.5 * sum(weights3))
        instance3 = {
            "capacity": cap3,
            "num_items": 50,
            "weights": weights3,
            "values": values3
        }
        instances["test_3"] = instance3
        lb["test_3"] = self.compute_baseline(instance3)
        
        return instances, lb

    def compute_baseline(self, instance):
        # Compute a baseline value using the classic greedy method by value-to-weight ratio.
        weights = instance["weights"]
        values_arr = instance["values"]
        cap = instance["capacity"]
        items = [{'weight': w, 'value': v} for w, v in zip(weights, values_arr)]
        items_sorted = sorted(items, key=lambda x: x['value'] / x['weight'], reverse=True)
        remaining = cap
        baseline_value = 0
        for item in items_sorted:
            if item['weight'] <= remaining:
                baseline_value += item['value']
                remaining -= item['weight']
        return baseline_value
