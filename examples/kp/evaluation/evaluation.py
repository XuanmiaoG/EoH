import numpy as np


class Evaluation:
    def __init__(self):
        # You can store instances and baseline (lb) here if needed.
        self.instances, self.lb = None, None

    def get_feasible_indices(
        self, items: list[tuple[float, float]], remaining_capacity: float
    ) -> list[int]:
        """
        Returns the indices of items that can still fit within remaining_capacity.

        :param items: A list of (weight, value) pairs (for items that are still available).
        :param remaining_capacity: The knapsack capacity still left.
        :return: List of indices for feasible items.
        """
        feasible = [i for i, (w, v) in enumerate(items) if w <= remaining_capacity]
        return feasible

    def greedy_knapsack(
        self, items: list[tuple[float, float]], capacity: float, alg
    ) -> float:
        """
        Perform a greedy selection of items into a single knapsack based on `alg.score()`.

        :param items: List of (weight, value) pairs for each item.
        :param capacity: Total capacity of the knapsack (must be >= 0).
        :param alg: An object or module with a method:
                     `score(weight, value, remaining_capacity) -> float`
        :return: The total value of items selected by the greedy approach.
        """
        remaining_capacity = capacity
        total_value = 0.0

        # Copy the items so we can delete as we pick them (to enforce 0-1 usage).
        available_items = items.copy()

        while True:
            # 1. Find which items can still fit.
            feasible_indices = self.get_feasible_indices(
                available_items, remaining_capacity
            )
            if not feasible_indices:
                # No items can fit anymore, so we stop.
                break

            # 2. Compute scores for feasible items.
            scores = []
            for idx in feasible_indices:
                w, v = available_items[idx]
                # Call the user-provided scoring function:
                scores.append(alg.score(w, v, remaining_capacity))

            # 3. Pick the item with the highest score.
            best_idx = feasible_indices[np.argmax(scores)]
            chosen_weight, chosen_value = available_items[best_idx]

            # 4. Update total value and reduce capacity.
            total_value += chosen_value
            remaining_capacity -= chosen_weight

            # 5. Remove the chosen item from the pool (enforcing 0-1 usage).
            del available_items[best_idx]

        return total_value

    def evaluateGreedy(self, instances: dict, alg) -> float:
        """
        Evaluate the heuristic (alg.score) on a batch of knapsack instances.

        :param instances: A dict of instances, where each instance has:
                {
                  "capacity": <float/int>,
                  "weights":  <list of weights>,
                  "values":   <list of values>,
                }
        :param alg: Object with `score(weight, value, remaining_capacity) -> float`
        :return: The average total value across all instances (or any other desired metric).
        """
        all_values = []

        for name, instance in instances.items():
            capacity = instance["capacity"]
            weights = instance["weights"]
            values = instance["values"]

            # Combine weights and values into pairs.
            items = list(zip(weights, values))

            # Run the “greedy knapsack” and track its total value.
            total_val = self.greedy_knapsack(items, capacity, alg)
            all_values.append(total_val)

        # Example metric: simply the average total value across instances.
        return np.mean(all_values)
