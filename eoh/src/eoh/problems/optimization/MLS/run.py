import types
import warnings
import sys
import numpy as np

from .get_instance import GetData, calc_fitness
from .prompts import GetPrompts


class MLS:
    """
    Runner for multi‑facility location mechanisms.
    Loads data via GetData, then evaluateHeuristic returns
    exactly the average fitness (social cost + penalty) per Equation (1).
    """

    def __init__(self, k: int = 2, epsilon: float = 0.01):
        self.k = k
        self.epsilon = epsilon

        gd = GetData()
        train, test = gd.get_instances()
        self.instances = {}
        self.instances.update(train)
        self.instances.update(test)

        try:
            self.prompts = GetPrompts()
        except ImportError:
            self.prompts = None

    def evaluateHeuristic(self, alg_module: types.ModuleType) -> float:
        """
        Runs `calc_fitness(...)` on every sample and returns
        the average fitness (cost + penalty) across all samples.
        """
        place_func = getattr(alg_module, "get_locations", None)
        if place_func is None:
            raise ValueError("Module must define get_locations(samples).")

        total_fit = 0.0
        count = 0

        for key, val in self.instances.items():
            peaks = val.get("peaks")
            if peaks is None or peaks.ndim != 2:
                continue

            misreports = val.get("misreports", None)
            weights = val.get("weights", None)
            num_samples, n_agents = peaks.shape

            for i in range(num_samples):
                p = peaks[i]
                # extract weights for this sample
                if weights is None:
                    w = np.ones(n_agents, dtype=float)
                elif weights.ndim == 1:
                    w = weights
                else:
                    w = weights[i]
                # extract misreports for this sample
                m = None
                if isinstance(misreports, np.ndarray) and misreports.ndim == 3:
                    if misreports.shape[0] == num_samples:
                        m = misreports[i]

                # this is exactly social cost + penalty
                fit_i = calc_fitness(p, m, w, place_func, k=self.k, epsilon=self.epsilon)
                total_fit += fit_i
                count += 1

        if count == 0:
            warnings.warn("No valid samples; returning large fitness.")
            return float("inf")

        return total_fit / count

    def evaluate(self, code_string: str) -> float | None:
        """
        Compile user code (must define get_locations) and return its average fitness.
        """
        try:
            print("Evaluating user code:\n", code_string)
            mod = types.ModuleType("user_heuristic")
            exec(code_string, mod.__dict__)
            sys.modules[mod.__name__] = mod
            return self.evaluateHeuristic(mod)
        except Exception as e:
            print("Evaluation error:", e)
            return None
