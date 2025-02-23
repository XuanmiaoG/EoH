import numpy as np

class Evaluation():
    def __init__(self):
        self.instances, self.lb = None, None

    def greedy_knapsack(self, items, capacity, alg):
        """
        Given a list of items (each a dict with keys 'weight' and 'value'),
        and a knapsack capacity, use the candidate heuristic (alg.score) to select items greedily.
        """
        remaining_capacity = capacity
        total_value = 0
        available = items.copy()
        while True:
            # Filter items that can fit
            feasible = [item for item in available if item['weight'] <= remaining_capacity]
            if not feasible:
                break
            # Compute scores for each feasible item
            scores = np.array([alg.score(item['weight'], item['value'], remaining_capacity) for item in feasible])
            idx = np.argmax(scores)
            chosen = feasible[idx]
            total_value += chosen['value']
            remaining_capacity -= chosen['weight']
            available.remove(chosen)
        return total_value

    def evaluateGreedy(self, instances: dict, alg) -> float:
        """
        For each knapsack instance, run the greedy_knapsack algorithm using the candidate heuristic.
        Each instance is a dict with keys: 'capacity', 'num_items', 'weights', 'values'.
        A baseline value (lb) is assumed to be provided in self.lb (e.g. by a standard greedy algorithm).
        The fitness is defined as the relative improvement of the candidate over the baseline.
        """
        values = []
        for name, instance in instances.items():
            capacity = instance['capacity']
            weights = instance['weights']
            values_arr = instance['values']
            # Create a list of items
            items = [{'weight': w, 'value': v} for w, v in zip(weights, values_arr)]
            candidate_value = self.greedy_knapsack(items, capacity, alg)
            values.append(candidate_value)
        avg_value = np.mean(values)
        baseline_avg = np.mean([self.lb[name] for name in instances.keys()])
        fitness = (avg_value - baseline_avg) / baseline_avg
        return fitness
