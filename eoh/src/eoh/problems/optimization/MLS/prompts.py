class GetPrompts:
    """
    Provides prompt/task information for the multi-facility location mechanism design problem,
    reflecting the design in the paper. It defines a function named `get_locations(samples)`,
    precisely matching the example heuristic from the paper, returning facility positions on [0,1].
    """

    def __init__(self):
        """
        We store the key pieces of text (the 'prompt') that describe:
          (1) The overall task.
          (2) The expected function name (here, `get_locations`).
          (3) The function's inputs.
          (4) The function's output.
          (5) Explanations of inputs/outputs.
          (6) Other guidance or sample code (replaced with the paper snippet).
        """

        # (1) TASK DESCRIPTION (mention the function name get_locations):
        self.prompt_task = (
            "We have a multi-facility location design problem on the real interval [0,1], "
            "with n agents each having a single-peaked preference. Specifically, agent i has a peak p_i, "
            "and the cost from a facility at x is |x - p_i|. We want to place facilities to "
            "minimize the total weighted cost, i.e., the sum over all agents of (weight_i * distance "
            "to the nearest facility). The final function should be named `get_locations(samples)` and "
            "should return a Python list of facility positions in [0,1], exactly as in the paper's example. "
            "In particular, please replicate the same code content shown in the example heuristic."
        )

        # (2) FUNCTION NAME:
        self.prompt_func_name = "get_locations"

        # (3) FUNCTION INPUTS:
        self.prompt_func_inputs = ["samples"]

        # (4) FUNCTION OUTPUT:
        self.prompt_func_outputs = ["locations"]

        # (5) EXPLANATION OF INPUTS/OUTPUTS:
        self.prompt_inout_inf = (
            "Inputs:\n"
            "  - `samples`: a list of location samples (agent peaks), shape (n,) in [0,1].\n"
            "    Internally, a list of fixed weights is also used or assumed, as in the paper's snippet.\n"
            "\n"
            "Output:\n"
            "  - A Python list containing the facility locations in [0,1]."
        )

        # (6) OTHER IMPLEMENTATION DETAILS + SAMPLE CODE
        self.prompt_other_inf = (
            "Below is the exact code snippet from the paper's example heuristic:\n\n"
            "def get_locations(samples):\n"
            "    '''\n"
            "    Determines the optimal locations from a given list of location samples.\n\n"
            "    Args:\n"
            "        samples (list): A one-dimensional list containing the location samples.\n"
            "        weights (list): A list of fixed weights assigned to the samples: [5,1,1,1,1]\n\n"
            "    Returns:\n"
            "        list: A one-dimensional list of the optimal locations, each in [0,1].\n"
            "    '''\n"
            "    weights = [5] + [1] * (len(samples) - 1)\n"
            "    weighted_samples = list(zip(samples, weights))\n"
            "\n"
            "    # Step 1: Cluster the samples into two groups based on proximity\n"
            "    weighted_samples.sort()\n"
            "    mid_index = len(weighted_samples) // 2\n"
            "    group1 = weighted_samples[:mid_index]\n"
            "    group2 = weighted_samples[mid_index:]\n\n"
            "    # Step 2: Calculate the weighted median for each group\n"
            "    def weighted_median(group):\n"
            "        total_weight = sum(w for _, w in group)\n"
            "        cum_weight = 0\n"
            "        for loc, w in group:\n"
            "            cum_weight += w\n"
            "            if cum_weight >= total_weight / 2:\n"
            "                return loc\n"
            "\n"
            "    facility1 = weighted_median(group1)\n"
            "    facility2 = weighted_median(group2)\n"
            "    locations = [facility1, facility2]\n"
            "    return locations\n"
        )

    def get_task(self) -> str:
        return self.prompt_task

    def get_func_name(self) -> str:
        return self.prompt_func_name

    def get_func_inputs(self) -> list:
        return self.prompt_func_inputs

    def get_func_outputs(self) -> list:
        return self.prompt_func_outputs

    def get_inout_inf(self) -> str:
        return self.prompt_inout_inf

    def get_other_inf(self) -> str:
        return self.prompt_other_inf
