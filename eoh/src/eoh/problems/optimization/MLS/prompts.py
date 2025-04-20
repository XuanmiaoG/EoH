class GetPrompts:
    """
    Provides prompt/task information for the multi-facility location mechanism design problem,
    reflecting the design in the paper. It defines a function named `place_facilities(peaks, weights, k)`
    that the LLM should generate or improve, returning the facility positions on [0,1].
    """

    def __init__(self):
        """
        We store the key pieces of text (the 'prompt') that describe:
          (1) The overall task.
          (2) The expected function name.
          (3) The function's inputs.
          (4) The function's output.
          (5) Explanations of inputs/outputs.
          (6) Other guidance or sample code.
        """

        # (1) TASK DESCRIPTION:
        self.prompt_task = (
            "We have a multi-facility location design problem on the real interval [0,1], "
            "with n agents each having a single-peaked preference. Specifically, agent i has a peak p_i, "
            "and the cost from a facility at x is |x - p_i|. We want to place K facilities to "
            "minimize the total weighted cost, i.e., the sum over all agents of (weight_i * distance "
            "to the nearest facility). You may implement any baseline (e.g., percentile, dictator, constant) "
            "or a novel heuristic. The final function should be named `place_facilities(peaks, weights, k)` "
            "and return a NumPy array of size k specifying where to place the facilities."
        )

        # (2) FUNCTION NAME:
        self.prompt_func_name = "place_facilities"

        # (3) FUNCTION INPUTS:
        #   - "peaks": array of agent peaks p_i in [0,1], shape (n,)
        #   - "weights": array of agent weights, shape (n,), possibly all 1 if unweighted
        #   - "k": number of facilities to place
        self.prompt_func_inputs = ["peaks", "weights", "k"]

        # (4) FUNCTION OUTPUT:
        #   Return a 1D NumPy array of length k, each entry in [0,1].
        self.prompt_func_outputs = ["facilities_positions"]

        # (5) EXPLANATION OF INPUTS/OUTPUTS:
        self.prompt_inout_inf = (
            "Inputs:\n"
            "  - `peaks`: agent peaks p_i, shape (n,) in [0,1]\n"
            "  - `weights`: agent weights, shape (n,). For unweighted, they can be 1s.\n"
            "  - `k`: the number of facilities to place.\n\n"
            "Output:\n"
            "  - A NumPy array of length k specifying the facility locations (each in [0,1])."
        )

        # (6) OTHER IMPLEMENTATION DETAILS + SAMPLE CODE:
        self.prompt_other_inf = (
            "Implementation Tips:\n"
            "  - The objective is to minimize sum_i [weights_i * min_j |peaks_i - facility_j|].\n"
            "  - You can search over candidate positions, or apply percentile/dictator rules. "
            "    For instance:\n"
            "      (a) Best Percentile: place each facility at certain quantiles of peaks.\n"
            "      (b) Best Dictator: pick the location of one agent's peak.\n"
            "      (c) Best Constant: search a grid in [0,1], if k=1.\n"
            "  - For general K>1, you can do clustering, or repeated grid search, etc.\n\n"
            "Below is a short sample code for the 'Best Constant' approach (k=1) and a simple fallback:\n\n"
            "import numpy as np\n\n"
            "def place_facilities(peaks, weights, k):\n"
            "    # If K=1, we do an explicit search over a small grid to find the best single constant.\n"
            "    if k == 1:\n"
            "        best_pos = 0.0\n"
            "        best_cost = float('inf')\n"
            "        for c in np.linspace(0, 1, 101):\n"
            "            cost_c = np.sum(weights * np.abs(peaks - c))\n"
            "            if cost_c < best_cost:\n"
            "                best_cost = cost_c\n"
            "                best_pos = c\n"
            "        return np.array([best_pos])\n\n"
            "    # Otherwise, a fallback: place them at equally spaced quantiles.\n"
            "    sorted_peaks = np.sort(peaks)\n"
            "    facilities = []\n"
            "    for i in range(k):\n"
            "        frac = (i + 1) / (k + 1)\n"
            "        idx = int(frac * (len(peaks) - 1))\n"
            "        facilities.append(sorted_peaks[idx])\n"
            "    return np.array(facilities)\n"
        )

    def get_task(self) -> str:
        """
        Returns a short textual description of the multi-facility location problem
        and what is expected (the user must define `place_facilities`).
        """
        return self.prompt_task

    def get_func_name(self) -> str:
        """
        Returns the required function name: 'place_facilities'.
        """
        return self.prompt_func_name

    def get_func_inputs(self) -> list:
        """
        Returns the list of parameter names: ['peaks', 'weights', 'k'].
        """
        return self.prompt_func_inputs

    def get_func_outputs(self) -> list:
        """
        Returns the list of output variable names, i.e. ['facilities_positions'].
        """
        return self.prompt_func_outputs

    def get_inout_inf(self) -> str:
        """
        Returns an explanation describing how the function inputs/outputs are structured.
        """
        return self.prompt_inout_inf

    def get_other_inf(self) -> str:
        """
        Returns additional guidance, including a sample baseline code snippet
        that enumerates a small grid (for k=1) or uses quantiles (for k>1).
        """
        return self.prompt_other_inf
