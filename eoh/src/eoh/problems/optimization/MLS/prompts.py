class GetPrompts:
    """
    Provides prompt/task information for the Multi-Facility Location Mechanism Design problem,
    reflecting baseline methods (Percentile, Dictator, Constant) mentioned in the literature.
    """

    def __init__(self):
        """
        The structure and method names remain consistent with prior examples:
          - get_task()
          - get_func_name()
          - get_func_inputs()
          - get_func_outputs()
          - get_inout_inf()
          - get_other_inf()
        """
        # (1) TASK DESCRIPTION
        self.prompt_task = (
            "In multi-facility location mechanism design, we have n agents with single-peaked "
            "preferences on [0,1]. The cost for agent i from a facility at x is |x - p_i|, where p_i "
            "is that agent’s peak location. We want to place K facilities to minimize the total "
            "weighted cost (sum of each agent’s distance to the nearest facility, potentially "
            "weighted by gamma_i). Our baseline approaches often include:\n"
            "  (a) Best Percentile Rule: pick one or more quantiles of the distribution of peaks.\n"
            "  (b) Best Dictator Rule: fix the facility location(s) at a particular agent’s peak.\n"
            "  (c) Best Constant Rule: search over a fixed set of candidate positions in [0,1].\n\n"
            "The user should define a function that implements any such baseline (or a new approach) "
            "and returns the facility positions for a given set of agent peaks and weights."
        )

        # (2) FUNCTION NAME
        # In many papers, the user must define place_facilities(peaks, weights, k).
        self.prompt_func_name = "place_facilities"

        # (3) FUNCTION INPUTS
        self.prompt_func_inputs = [
            "peaks",    # array of agent peaks, shape (n,)
            "weights",  # array of agent weights, shape (n,) or uniform if not used
            "k"         # number of facilities to place
        ]

        # (4) FUNCTION OUTPUT
        # The function should return an array of length k specifying the facility locations.
        self.prompt_func_outputs = [
            "facilities_positions"  # shape (k,)
        ]

        # (5) INPUTS/OUTPUTS EXPLANATION
        self.prompt_inout_inf = (
            "Input:\n"
            "  - 'peaks': agent peak locations on [0,1], shape (n,)\n"
            "  - 'weights': agent weights, shape (n,). If not relevant, they could be uniform.\n"
            "  - 'k': the desired number of facilities.\n\n"
            "Output:\n"
            "  - A NumPy array of length k specifying each facility’s position on [0,1]."
        )

        # (6) OTHER IMPLEMENTATION DETAILS + SAMPLE CODE
        self.prompt_other_inf = (
            "Implementation tips:\n"
            "  - You can enumerate or search parameters to find the best constant, percentile, or dictator.\n"
            "  - Or place them by a direct formula (e.g., equally spaced quantiles).\n"
            "  - Return an array of facility positions, each in [0,1].\n\n"

            "Below is a sample code snippet that illustrates how you might implement a 'Best Constant' baseline "
            "by enumerating a small set of points in [0,1]. For more advanced baselines, you can adapt similarly.\n\n"

            "import numpy as np\n\n"
            "def place_facilities(peaks, weights, k):\n"
            "    # Example: 'Best Constant' if k=1, enumerating a grid in [0,1].\n"
            "    # For multiple k, we might do multi-dimensional grid or a cluster approach.\n\n"
            "    if k != 1:\n"
            "        # Simple fallback: put them at equally spaced quantiles\n"
            "        sorted_peaks = np.sort(peaks)\n"
            "        facilities = []\n"
            "        for i in range(k):\n"
            "            frac = (i + 1) / (k + 1)\n"
            "            idx = int(frac * (len(peaks) - 1))\n"
            "            facilities.append(sorted_peaks[idx])\n"
            "        return np.array(facilities)\n\n"
            "    # If k=1, do an explicit search for best constant\n"
            "    best_pos = 0.0\n"
            "    best_cost = float('inf')\n"
            "    # e.g., search 101 points from 0.0 to 1.0\n"
            "    for c in np.linspace(0, 1, 101):\n"
            "        cost_c = np.sum(weights * np.abs(peaks - c))\n"
            "        if cost_c < best_cost:\n"
            "            best_cost = cost_c\n"
            "            best_pos = c\n"
            "    # Return the best single facility\n"
            "    return np.array([best_pos])"
        )

    def get_task(self):
        """Returns the multi-facility location baseline task description."""
        return self.prompt_task

    def get_func_name(self):
        """Returns the expected function name, e.g., 'place_facilities'."""
        return self.prompt_func_name

    def get_func_inputs(self):
        """Returns the list of input parameters the user must define."""
        return self.prompt_func_inputs

    def get_func_outputs(self):
        """Returns the expected output variable names."""
        return self.prompt_func_outputs

    def get_inout_inf(self):
        """Returns the explanation of function inputs and outputs."""
        return self.prompt_inout_inf

    def get_other_inf(self):
        """
        Returns additional guidance and sample code, including how to implement a baseline
        approach (e.g., Best Constant or a fallback quantile approach).
        """
        return self.prompt_other_inf
