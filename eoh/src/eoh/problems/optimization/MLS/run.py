import numpy as np
import types
import warnings
import sys

# We import the same modules from get_instance:
from .get_instance import (
    GetData,
    calc_fitness,
    place_facilities_quantiles,  # optional if you want direct usage
)
from .prompts import GetPrompts  # if you have a prompts.py file


class MLS:
    """
    A class analogous to your knapsack 'evaluate' snippet,
    but for multi-facility location. We'll load data from GetData,
    then define an evaluate(...) method that expects a user-defined
    place_facilities(...) function, measure the fitness, and return it.
    """

    def __init__(self, epsilon=0.01, k=2):
        """
        We load train/test data, store them in self.instances, and keep
        an epsilon threshold for strategy-proofness checking. Also store k if you want
        a default # of facilities.
        """
        self.epsilon = epsilon
        self.k = k
        getdata = GetData()
        train_data, test_data = getdata.get_instances()

        # Merge or keep them separate
        self.instances = {}
        self.instances.update(train_data)
        self.instances.update(test_data)

        self.prompts = GetPrompts()

    def evaluateHeuristic(self, alg) -> float:
        """
        Evaluate the user-supplied place_facilities(...) code on all loaded instances.

        alg must define:
          def place_facilities(peaks, weights, k):
              # returns an array of length k
        We'll compute the average fitness across all samples in self.instances.
        """
        place_func = getattr(alg, "place_facilities", None)
        if place_func is None:
            raise ValueError(
                "Candidate module does not define a 'place_facilities(peaks,weights,k)' function."
            )

        total_fitness = 0.0
        total_samples = 0

        for key, val in self.instances.items():
            if 'peaks' not in val:
                continue
            peaks_arr = val['peaks']  # shape (num_samples, n)
            misr_arr = val.get('misreports', None)
            weights_info = val.get('weights', None)

            # If the instance has 'k' stored, you can override self.k
            #  k_local = val['k'] if 'k' in val else self.k
            k_local = self.k

            # Basic shape checks:
            if len(peaks_arr.shape) != 2:
                print(f"Warning: 'peaks' shape not 2D for {key}")
                continue

            num_samples, n_agents = peaks_arr.shape

            for i in range(num_samples):
                peaks_i = peaks_arr[i]

                # If weights are shape (n_agents,)
                # or if shape (num_samples, n_agents), take row i
                if weights_info is not None:
                    if len(weights_info.shape) == 1:
                        weights_i = weights_info
                    elif len(weights_info.shape) == 2:
                        weights_i = weights_info[i]
                    else:
                        print(f"Warning: unexpected weights shape {weights_info.shape} in {key}")
                        weights_i = np.ones(n_agents, dtype=float)
                else:
                    weights_i = np.ones(n_agents, dtype=float)

                # If misreports is shape (num_samples, n_agents, M)
                misreports_i = None
                if misr_arr is not None and len(misr_arr.shape) == 3:
                    if misr_arr.shape[0] == num_samples and misr_arr.shape[1] == n_agents:
                        misreports_i = misr_arr[i]
                    else:
                        print(f"Warning: misreports shape mismatch {misr_arr.shape} in {key}")

                fitness_i = calc_fitness(
                    peaks_i, misreports_i, weights_i,
                    place_func,
                    k_local,
                    epsilon=self.epsilon
                )
                total_fitness += fitness_i
                total_samples += 1

        if total_samples == 0:
            print("Warning: no valid samples found. Returning 9999.")
            return 9999.0

        return total_fitness / total_samples

    def evaluate(self, code_string):
        """
        This is analogous to the knapsack's evaluate(...) method.
        We expect code_string to define a function:

            def place_facilities(peaks, weights, k):
                ...
                return np.array([...])  # length k

        We'll parse that code, call evaluateHeuristic(...), and return a single float fitness.
        Lower is better => 0 means no cost/penalty, etc.
        """
        try:
            print("Evaluating user-supplied multi-facility code:\n", code_string)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateHeuristic(heuristic_module)
                return fitness

        except Exception as e:
            print("Error in evaluate:", e)
            return None


