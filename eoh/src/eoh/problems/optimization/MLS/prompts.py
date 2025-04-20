class GetPrompts:
    """
    Provides prompt/task information for the multi-facility location mechanism design problem,
    matching the paper's weighted median heuristic for k=2 and generalizing to k facilities.
    """

    def __init__(self):
        # (1) TASK DESCRIPTION
        self.prompt_task = (
            "We have n agents with single-peaked preferences on [0,1] and weights gamma_i. "
            "Define a function get_locations(peaks, weights, k) that places k facilities to minimize the "
            "sum of weighted distances to the nearest facility (social cost)."
        )

        # (2) FUNCTION NAME
        self.prompt_func_name = "get_locations"

        # (3) FUNCTION INPUTS
        self.prompt_func_inputs = [
            "peaks",    # list of agent peak locations, shape (n,)
            "weights",  # list of agent weights, shape (n,)
            "k"         # number of facilities to place
        ]

        # (4) FUNCTION OUTPUT
        self.prompt_func_outputs = [
            "locations"  # list of k facility positions in [0,1]
        ]

        # (5) INPUT/OUTPUT EXPLANATION
        self.prompt_inout_inf = (
            "Inputs:\n"
            "  - peaks: a one-dimensional list of n agent peaks in [0,1].\n"
            "  - weights: a one-dimensional list of n agent weights.\n"
            "  - k: the number of facilities to locate.\n"
            "Output:\n"
            "  - locations: a list of length k with each facility position in [0,1]."
            "You should define the list locations and asign it the k facility positions.\n"
        )

        # (6) SIMPLE CODE EXAMPLE
        self.prompt_other_inf = (
            "Here's a minimal example for k=2 using weighted medians:\n"
            "```python\n"
            "def get_locations(peaks, weights, k):\n"
            "    \"\"\"\n"
            "    Strategy: pair peaks with weights, split, then per-group weighted median.\n"
            "    \"\"\"\n"
            "    pairs = sorted(zip(peaks, weights))\n"
            "    half = len(pairs) // 2\n"
            "    def wmedian(group):\n"
            "        total = sum(w for _, w in group)\n"
            "        cum = 0\n"
            "        for p, w in group: return p if (cum := cum + w) >= total/2 else None\n"
            "    return [wmedian(pairs[:half]), wmedian(pairs[half:])]\n"
            "```"
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
