import numpy as np
import types
import warnings
import sys

from .get_instance import GetData
from .prompts import GetPrompts


class KNAPSACK:
    """
    A class for evaluating candidate heuristic functions (score functions) on the 0-1 Knapsack Problem.
    Each instance of the problem is defined by:
        - A list of items, each having 'weight' (int > 0) and 'value' (int > 0).
        - A knapsack capacity (int > 0).

    Our approach is to repeatedly pick from the set of unselected (available) items,
    using a user-provided scoring function `score(weight, value, remaining_capacity)`.
    We always select the item with the highest score among those that fit (weight <= remaining_capacity).
    Once an item is chosen, it is removed from the available set to satisfy the '0-1' property
    (i.e., no item can be used more than once), and the capacity is decreased accordingly.
    The process stops when no feasible item can be chosen.
    """

    def __init__(self):
        """
        Constructor loads default instances using GetData() and also
        retrieves the baseline (lb) of each instance for comparison.
        """
        getdata = GetData()
        # Here we load instances with 50 items as an example.
        self.instances, self.high_bound = getdata.get_instances("500")
        self.prompts = GetPrompts()

    def greedy_knapsack(self, items, capacity, score_func):
        """
        Perform a greedy selection of items (0-1 Knapsack) based on a provided scoring function.

        :param items: A list of dicts, each with keys:
                        - 'weight': int, the weight of the item
                        - 'value':  int, the value of the item
        :param capacity: int, the maximum weight capacity of the knapsack
        :param score_func: A function of signature score(weight, value, remaining_capacity) -> float

        :return: total_value (float), the sum of values of the chosen items.
        """
        remaining_capacity = capacity
        total_value = 0.0

        # Copy to avoid modifying the original list
        available = items.copy()

        while True:
            # Filter out items that can still fit
            feasible = [
                item for item in available if item["weight"] <= remaining_capacity
            ]
            if not feasible:
                # No item fits -> stop
                break

            # Compute the score for each feasible item
            scores = []
            for item in feasible:
                # Call the user's scoring function
                result = score_func(item["weight"], item["value"], remaining_capacity)

                # If result is a NumPy array, convert to scalar
                if isinstance(result, np.ndarray):
                    result = result.item()

                # Convert to float explicitly
                scores.append(float(result))

            # Choose the item with the highest score
            idx = np.argmax(scores)
            chosen_item = feasible[idx]

            # Update total value and capacity
            total_value += chosen_item["value"]
            remaining_capacity -= chosen_item["weight"]

            # Remove the chosen item from the pool (0-1 constraint)
            available.remove(chosen_item)

        return total_value

    def evaluateGreedy(self, alg) -> float:
        """
        Evaluate the given candidate 'score' function (defined inside alg) on each loaded knapsack instance.

        :param alg: A module-like object expected to define a function 'score(weight, value, remaining_capacity)'.
                    This function should return a numeric score guiding greedy selection.

        :return: fitness (float), the relative improvement over a baseline.
                 Specifically, if 'avg_value' is the average total value achieved by the candidate,
                 and 'baseline_avg' is the average total value of our known baseline,
                 fitness = (avg_value - baseline_avg) / baseline_avg.
        """
        # Retrieve the candidate score function
        score_func = getattr(alg, "score", None)
        if score_func is None:
            raise ValueError("Candidate module does not define a 'score' function.")

        # If it's a numba-jitted function, get the python version (optional step for debugging/flexibility)
        if hasattr(score_func, "py_func"):
            score_func = score_func.py_func

        # Run the candidate's greedy approach on each instance
        values_obtained = []
        for name, instance in self.instances.items():
            capacity = instance["capacity"]
            weights = instance["weights"]
            values_arr = instance["values"]

            # Build a list of item dicts
            items = [{"weight": w, "value": v} for w, v in zip(weights, values_arr)]

            candidate_value = self.greedy_knapsack(items, capacity, score_func)
            values_obtained.append(candidate_value)

        # Compute average result vs. baseline
        avg_value = np.mean(values_obtained)
        if avg_value == 0:
            print("Warning: Candidate returned zero total value on all instances.")
        baseline_avg = np.mean(
            [self.high_bound[name] for name in self.instances.keys()]
        )
        fitness = (baseline_avg - avg_value) / baseline_avg
        return fitness

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Print the code string for debugging
                print("Evaluating function source:")
                print(code_string)

                # Create a new, empty module to hold the candidate's code
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Print the function object
                print(
                    "Score function object:", getattr(heuristic_module, "score", None)
                )

                # Evaluate the module's score function via greedy_knapsack
                fitness = self.evaluateGreedy(heuristic_module)
                return fitness
        except Exception as e:
            print("Error:", str(e))
            return None
