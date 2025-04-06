class GetPrompts:
    def __init__(self):
        """
        This class encapsulates the prompt construction logic for the Multi-Facility Location
        Mechanism Design problem with single-peaked preferences on [0,1]. We only provide
        technical/mathematical details about the problem setting, inputs, and outputs,
        without extraneous general discussions.
        """

        # Task description with problem-specific technical details.
        self.prompt_task = (
            "We consider a multi-facility location design problem on the interval [0,1], where "
            "we have n agents. Each agent i has a single-peaked cost function u_i(x) = |x - p_i|, "
            "where p_i is the agent's peak in [0,1]. We aim to place K facilities (K <= n) in [0,1] "
            "to serve the agents. Each agent uses the nearest facility and incurs that distance as cost. "
            "Our objective is to minimize the (weighted) sum of all agents' costs. Formally, given "
            "peaks p = (p_1, p_2, ..., p_n) and K, we want to find x = (x_1, ..., x_K) so as to "
            "minimize sum_i (gamma_i * min_k |p_i - x_k|). If there are weights gamma_i, they indicate "
            "the relative importance of agent i. The function we define must output a single numeric "
            "score for each potential facility position in [0,1]. A higher score indicates a more "
            "desirable position to place a facility (before finalizing all K positions). The approach "
            "should enable a greedy strategy that selects positions one by one if needed. Our ultimate "
            "goal is to place the K facilities to achieve low total cost."
        )

        # The name of the function to be defined.
        self.prompt_func_name = "position_score"

        # The inputs of the function.
        # We assume an incremental facility placement scenario, where we have:
        #   - The array of peaks (p) of length n
        #   - The current index for the next facility k_idx (0 <= k_idx < K)
        #   - Optionally an array fac_so_far for already-chosen facility locations
        self.prompt_func_inputs = ["p", "k_idx", "fac_so_far"]

        # The output of the function.
        # We output a single float representing the desirability (score) of placing
        # the (k_idx+1)-th facility at some position x in [0,1].
        # In the actual code, x could be varied or discretized, etc.
        self.prompt_func_outputs = ["score"]

        # Explanation of inputs/outputs in a purely technical manner.
        self.prompt_inout_inf = (
            "Input:\n"
            " - p: a one-dimensional list or array of agent peaks, each in [0,1], length n.\n"
            " - k_idx: an integer index (0 <= k_idx < K) indicating which facility we are currently placing.\n"
            " - fac_so_far: a list containing the locations of the facilities already placed, if any.\n\n"
            "Output:\n"
            " - score: a numeric float evaluating the desirability of placing the next facility "
            "   at a certain location x in [0,1], based on the agent peaks p, any previously placed "
            "   facilities fac_so_far, and k_idx."
        )

        # Additional implementation constraints or instructions.
        # Provide a sample snippet to illustrate usage.
        self.prompt_other_inf = (
            "We assume import of numpy as np at the start of the code (import numpy as np). "
            "The function must return a float value for each candidate location x in [0,1]. "
            "If needed, we can evaluate different x's by calling this function multiple times. "
            "Internally, we may incorporate weights gamma_i if relevant. The function must not "
            "return a callable or function object but only a float for each x. If x is out of [0,1], "
            "we can penalize it heavily by returning a large positive cost or negative score.\n\n"
            "Here is a minimal sample snippet:\n\n"
            "import numpy as np\n\n"
            "def potential_gain(p, x, fac_so_far):\n"
            "    # Example calculation: measure how much total distance cost might be reduced\n"
            "    # if we add a facility at x. (This is a simple placeholder formula.)\n"
            "    cost_before = sum(min(abs(pi - f) for f in fac_so_far) if fac_so_far else 1.0 for pi in p)\n"
            "    cost_after  = sum(min(abs(pi - f) for f in fac_so_far + [x]) for pi in p)\n"
            "    return cost_before - cost_after\n\n"
            "def position_score(p, k_idx, fac_so_far):\n"
            "    # Try a naive approach: we use potential_gain.\n"
            "    # For each agent's peak, if x is out of [0,1], return a negative.\n"
            "    # Otherwise, return potential_gain.\n"
            "    x = 0.5  # placeholder, or we iterate externally\n"
            "    if x < 0 or x > 1:\n"
            "        return -1e9\n"
            "    return potential_gain(p, x, fac_so_far)\n"
        )

    def get_task(self):
        """
        Returns the prompt_task string, describing the multi-facility location problem.
        """
        return self.prompt_task

    def get_func_name(self):
        """
        Returns the name of the function, e.g., 'position_score'.
        """
        return self.prompt_func_name

    def get_func_inputs(self):
        """
        Returns a list of the function's inputs:
        ['p', 'k_idx', 'fac_so_far'].
        """
        return self.prompt_func_inputs

    def get_func_outputs(self):
        """
        Returns a list of the function's outputs: ['score'].
        """
        return self.prompt_func_outputs

    def get_inout_inf(self):
        """
        Returns a string explaining the inputs/outputs in a purely technical manner.
        """
        return self.prompt_inout_inf

    def get_other_inf(self):
        """
        Returns additional constraints or sample code snippet for reference.
        """
        return self.prompt_other_inf
