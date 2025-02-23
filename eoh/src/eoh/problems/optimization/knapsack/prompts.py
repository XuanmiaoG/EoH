class GetPrompts():
    def __init__(self):
        self.prompt_task = ("I need help designing a novel score function for the knapsack problem. "
                            "In each step, given an item’s weight and value and the remaining capacity of the knapsack, "
                            "the function should output a score. The item with the highest score among those that can fit "
                            "will be selected. The final goal is to maximize the total value of items in the knapsack.")
        self.prompt_func_name = "score"
        self.prompt_func_inputs = ["weight", "value", "remaining_capacity"]
        self.prompt_func_outputs = ["score"]
        self.prompt_inout_inf = ("'weight' and 'value' represent the item's weight and value, and "
                                 "'remaining_capacity' is the current available capacity. The output 'score' "
                                 "is a numeric value indicating the desirability of including the item.")
        self.prompt_other_inf = ("Note that all inputs and outputs are numbers. Include 'import numpy as np' "
                                 "and 'from numba import jit' at the top. Place '@jit(nopython=True)' just above the "
                                 "function definition for performance.")

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
