import numpy as np
import types
import warnings
import sys

# "get_instance.py" presumably defines:
#     - GetData
#     - calc_fitness( peaks, misreports, weights, place_func, k, epsilon=0.01 )
#       which now requires a 'k' parameter.
#     - place_facilities_quantiles (optional baseline)
from .get_instance import (
    GetData,
    calc_fitness,
    place_facilities_quantiles,  # optional
)

from .prompts import GetPrompts


class MLS:
    """
    Multi-Facility Location Mechanism Design 'Runner'.

    Loads train/test data from GetData, then provides evaluate(...) to measure how well
    a user-defined 'get_locations(...)' function performs.  The evaluation is the average
    'fitness', i.e., sum of (weighted) social cost + any strategyproofness penalty,
    following the paper's approach.

    Key notes:
      - Single-peaked preferences on [0,1].
      - We pass 'k' to calc_fitness(...) if it requires a facility count.
      - The user code must define `get_locations(samples) -> [locations]`.
      - By default, we assume two facilities (k=2), or you can adjust it as needed.
    """

    def __init__(self, k=2, epsilon=0.01):
        """
        Initialize the runner. We store 'k' (number of facilities) and 'epsilon'
        (the strategyproofness threshold).
        """
        self.k = k
        self.epsilon = epsilon

        # Load data from get_instance.py
        getdata = GetData()
        train_data, test_data = getdata.get_instances()

        # Merge train+test
        self.instances = {}
        self.instances.update(train_data)
        self.instances.update(test_data)

        # Load prompt info if needed
        try:
            self.prompts = GetPrompts()
        except Exception:
            self.prompts = None

    def evaluateHeuristic(self, alg) -> float:
        """
        Evaluate a user-supplied mechanism on all loaded samples.

        The mechanism 'alg' must define `get_locations(samples)`.
        We'll compute the average fitness across all data.
        """
        place_func = getattr(alg, "get_locations", None)
        if place_func is None:
            raise ValueError(
                "No 'get_locations(samples)' function found in the user code."
            )

        total_fitness = 0.0
        total_samples = 0

        for key, val in self.instances.items():
            if "peaks" not in val:
                continue

            peaks_arr = val["peaks"]
            misr_arr = val.get("misreports", None)
            weights_info = val.get("weights", None)

            if len(peaks_arr.shape) != 2:
                warnings.warn(f"'peaks' for {key} is not 2D. Skipping.")
                continue

            num_samples, n_agents = peaks_arr.shape

            for i in range(num_samples):
                peaks_i = peaks_arr[i]

                if weights_info is not None:
                    if len(weights_info.shape) == 1:
                        weights_i = weights_info
                    elif len(weights_info.shape) == 2:
                        weights_i = weights_info[i]
                    else:
                        warnings.warn(
                            f"Unexpected 'weights' shape {weights_info.shape} in {key}."
                        )
                        weights_i = np.ones(n_agents, dtype=float)
                else:
                    weights_i = np.ones(n_agents, dtype=float)

                misreports_i = None
                if misr_arr is not None and len(misr_arr.shape) == 3:
                    if (
                        misr_arr.shape[0] == num_samples
                        and misr_arr.shape[1] == n_agents
                    ):
                        misreports_i = misr_arr[i]
                    else:
                        warnings.warn(
                            f"misreports shape mismatch {misr_arr.shape} for {key}, skipping misreports."
                        )

                # Pass k=self.k explicitly to calc_fitness so it won't complain
                fitness_i = calc_fitness(
                    peaks_i,
                    misreports_i,
                    weights_i,
                    place_func,
                    k=self.k,
                    epsilon=self.epsilon,
                )
                total_fitness += fitness_i
                total_samples += 1

        if total_samples == 0:
            warnings.warn("No valid samples found. Returning fitness=9999.")
            return 9999.0

        return total_fitness / total_samples

    def evaluate(self, code_string: str) -> float:
        """
        Evaluate a user-supplied code snippet that defines 'get_locations(samples)'.
        We'll compile and run it, returning the average fitness (lower is better).
        """
        try:
            print("Evaluating user-supplied multi-facility code:\n", code_string)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a fresh module
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateHeuristic(heuristic_module)
                return fitness

        except Exception as e:
            print("Error in evaluate:", e)
            return None
