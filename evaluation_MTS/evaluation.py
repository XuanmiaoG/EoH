import numpy as np
import types
import warnings
import sys

# Import necessary components from get_instance module
from get_instance import (
    GetData,
    calc_fitness,
)


class MLS_Test:
    """
    An evaluation class for multi-facility location problems using only test data.

    Workflow:
      1. Load only the test data through GetData.
      2. The method evaluateHeuristic requires that the candidate module defines
         place_facilities(peaks, weights, k).
         This is called for all test instances to compute the fitness (weighted social cost + penalty).
      3. The method evaluate takes a user-provided code string, dynamically loads it,
         and calls evaluateHeuristic to obtain the average fitness (lower is better).
    """

    def __init__(self, epsilon: float = 0.01, k: int = 2) -> None:
        """
        Constructor for MLS_Test.

        Args:
            epsilon: The maximum regret threshold above which a penalty is added.
            k: The number of facilities to be placed (can be overridden per instance if desired).
        """
        self.epsilon = epsilon
        self.k = k
        data_handler = GetData()
        # Retrieve only the test data
        _, test_data = data_handler.get_instances()
        self.instances = test_data

    def evaluateHeuristic(self, alg: object) -> float:
        """
        Evaluates the candidate mechanism by invoking the place_facilities(...) function
        on all test data samples. Returns the average fitness (lower is better).

        Args:
            alg: A module-like object that must provide the function
                 place_facilities(peaks, weights, k).

        Returns:
            The average fitness across all test samples.
        """
        # Obtain place_facilities function from the candidate module
        place_func = getattr(alg, "place_facilities", None)
        if place_func is None:
            raise ValueError(
                "The candidate module does not define 'place_facilities(peaks, weights, k)'."
            )

        total_fitness = 0.0
        total_samples = 0

        # Iterate over all test data instances
        for key, val in self.instances.items():
            # If 'peaks' is not available, skip
            if "peaks" not in val:
                continue

            peaks_arr = val["peaks"]  # Shape: (num_samples, n_agents)
            misr_arr = val.get("misreports", None)
            weights_info = val.get("weights", None)
            k_local = self.k

            # Check that peaks has two dimensions
            if len(peaks_arr.shape) != 2:
                print(
                    f"Warning: in instance {key}, 'peaks' is not 2-dimensional. Skipping."
                )
                continue

            num_samples, n_agents = peaks_arr.shape

            # Compute fitness for each sample in this instance
            for i in range(num_samples):
                peaks_i = peaks_arr[i]

                # Process weight data, which may be 1D or 2D
                if weights_info is not None:
                    if len(weights_info.shape) == 1:
                        weights_i = weights_info
                    elif len(weights_info.shape) == 2:
                        weights_i = weights_info[i]
                    else:
                        print(
                            f"Warning: in instance {key}, 'weights' has unexpected shape {weights_info.shape}, using uniform weights."
                        )
                        weights_i = np.ones(n_agents, dtype=float)
                else:
                    # Default to uniform weights if none are provided
                    weights_i = np.ones(n_agents, dtype=float)

                # If misreports exist and are 3D, extract the portion for this sample
                misreports_i = None
                if misr_arr is not None and len(misr_arr.shape) == 3:
                    if (
                        misr_arr.shape[0] == num_samples
                        and misr_arr.shape[1] == n_agents
                    ):
                        misreports_i = misr_arr[i]
                    else:
                        print(
                            f"Warning: in instance {key}, misreports has mismatch shape {misr_arr.shape}."
                        )

                # Calculate fitness for this sample
                fitness_i = calc_fitness(
                    peaks_i,
                    misreports_i,
                    weights_i,
                    place_func,
                    k_local,
                    epsilon=self.epsilon,
                )
                total_fitness += fitness_i
                total_samples += 1

        # If no valid samples were found, return a large fitness value
        if total_samples == 0:
            print("Warning: no valid samples found, returning 9999.0.")
            return 9999.0

        return total_fitness / total_samples

    def evaluate(self, code_string: str) -> float or None:
        """
        Evaluates user-provided code that must define the following function:

            def place_facilities(peaks, weights, k):
                ...
                return np.array([...])

        The code is dynamically loaded, then evaluateHeuristic is called to compute
        the average fitness on the test data.

        Args:
            code_string: String containing user-provided Python code.

        Returns:
            The average fitness (float) if evaluation is successful, otherwise None in case of error.
        """
        try:
            print(
                "Evaluating user-provided multi-facility location code:\n", code_string
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Dynamically create a module for the user's code
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Invoke evaluateHeuristic on the loaded module
                fitness = self.evaluateHeuristic(heuristic_module)
                return fitness

        except Exception as e:
            print("An error occurred during evaluation:", e)
            return None
