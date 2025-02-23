import numpy as np
import importlib
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys

class KNAPSACK():
    def __init__(self):
        # Here we load a default instance set (e.g. with 100 items)
        getdata = GetData()
        self.instances, self.lb = getdata.get_instances("100")
        self.prompts = GetPrompts()

    def greedy_knapsack(self, items, capacity, alg):
        remaining_capacity = capacity
        total_value = 0
        available = items.copy()
        while True:
            feasible = [item for item in available if item['weight'] <= remaining_capacity]
            if not feasible:
                break
            scores = [alg.score(item['weight'], item['value'], remaining_capacity) for item in feasible]
            idx = np.argmax(scores)
            chosen = feasible[idx]
            total_value += chosen['value']
            remaining_capacity -= chosen['weight']
            available.remove(chosen)
        return total_value

    def evaluateGreedy(self, alg) -> float:
        values = []
        for name, instance in self.instances.items():
            capacity = instance['capacity']
            weights = instance['weights']
            values_arr = instance['values']
            items = [{'weight': w, 'value': v} for w, v in zip(weights, values_arr)]
            candidate_value = self.greedy_knapsack(items, capacity, alg)
            values.append(candidate_value)
        avg_value = np.mean(values)
        baseline_avg = np.mean([self.lb[name] for name in self.instances.keys()])
        fitness = (avg_value - baseline_avg) / baseline_avg
        return fitness

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module
                fitness = self.evaluateGreedy(heuristic_module)
                return fitness
        except Exception as e:
            return None

if __name__ == "__main__":
    # Example usage:
    knap = KnapsackProblem()
    heuristic_module = importlib.import_module("heuristic")
    heuristic = importlib.reload(heuristic_module)
    fitness = knap.evaluateGreedy(heuristic)
    print("Knapsack Evaluation Fitness:", fitness)
