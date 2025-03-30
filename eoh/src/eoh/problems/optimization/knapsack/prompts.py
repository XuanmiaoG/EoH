class GetPrompts:
    def __init__(self):
        # Task description referencing the 0-1 Knapsack Problem.
        self.prompt_task = (
            "In the 0-1 Knapsack Problem, we are given a set of n items, where each item i has "
            "an integer weight w_i > 0 and an integer value v_i > 0. We also have a knapsack with "
            "capacity W > 0. Our objective is to select a subset of these items such that the total "
            "weight does not exceed W and the total value is maximized. I need help designing a novel "
            "score function for a greedy approach. In each step, given an item's weight, value, and "
            "the remaining capacity of the knapsack, the function should output a numeric score. The "
            "item with the highest score among those that can fit will be selected. The final goal is "
            "to maximize the total value of the selected items."
        )

        # The name of the function to be defined.
        self.prompt_func_name = "score"

        # The inputs of the function.
        self.prompt_func_inputs = ["weight", "value", "remaining_capacity"]

        # The output of the function.
        self.prompt_func_outputs = ["score"]

        # Explanation of inputs/outputs.
        self.prompt_inout_inf = (
            "'weight' and 'value' represent the weight and value of an item, respectively, while "
            "'remaining_capacity' is the current available capacity in the knapsack. The output, "
            "must be a single numeric value (specifically a float) indicating the desirability "
            "of selecting that item."
        )

        # Additional implementation details.
        self.prompt_other_inf = (
            "Note that all inputs and outputs are numbers. Include the following import at the top of your code: "
            "'import numpy as np'. The function should always return a float value. If the item does not fit "
            "in the knapsack (i.e., weight > remaining_capacity), the function must return a large negative "
            "float value (for example, -1e9). The function must not return any callable or function object, "
            "and all helper functions (if any) should be placed immediately above the 'score' function definition. "
            "Ensure that your function returns a float number and that all parentheses are properly closed."
            """Sample code: 
import numpy as np

def calculate_adjusted_score(value, weight, remaining_capacity):
    if weight > remaining_capacity:
        return -1e9
    adjustment = np.log(remaining_capacity + 1)
    base_score = (value / weight) ** 2
    return np.exp(base_score) * adjustment


def score(weight, value, remaining_capacity):
    score = calculate_adjusted_score(value, weight, remaining_capacity)
    return score
"""
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
