class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "I need help designing a novel score function for the knapsack problem. "
            "In each step, given an item’s weight and value along with the remaining capacity "
            "of the knapsack, the function should output a numeric score. The item with the highest "
            "score among those that can fit (i.e., whose weight is less than or equal to the remaining capacity) "
            "will be selected. The final goal is to maximize the total value of items in the knapsack."
        )
        self.prompt_func_name = "score"
        self.prompt_func_inputs = ["weight", "value", "remaining_capacity"]
        self.prompt_func_outputs = ["score"]
        self.prompt_inout_inf = (
            "'weight' and 'value' represent the weight and value of an item, respectively, "
            "and 'remaining_capacity' is the current available capacity in the knapsack. "
            "The output 'score' must be a single numeric value (for example, a float) that indicates "
            "the desirability of selecting that item."
        )
        self.prompt_other_inf = (
            "Note that all inputs and outputs are numbers. Include the following imports at the top of your code: "
            "'import numpy as np' and 'from numba import jit'. Place '@jit(nopython=True)' immediately "
            "above the function definition. Ensure that your function returns a numeric value (not a callable) "
            "and that all parentheses are properly closed."
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