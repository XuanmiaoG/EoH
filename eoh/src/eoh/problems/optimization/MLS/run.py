"""
run.py

This module is responsible for:
1) Loading facility-location data (peaks, misreports, weights).
2) Evaluating any user-defined mechanism (heuristic) by:
   - Computing its empirical social cost (per Equation (1) in the paper),
   - Adding a penalty term (Equation (2)) if the mechanism is not empirically strategyproof.
3) Returning a single scalar value (the average fitness) for each evaluated mechanism.

Reference to Paper Sections:
- Section 2 (Preliminaries): Describes the data structure (peaks, misreports, weights).
- Section 3.3 (Fitness Evaluation): Explains how we compute the sum of agent distances to
  the nearest facility (social cost) and enforce strategyproofness via a regret penalty.
"""

import types
import warnings
import sys
import numpy as np
from typing import Optional

# The classes below are assumed to be local or from the same package.
#   - GetData: loads the dataset (agent peaks, optional misreports, and weights).
#   - calc_fitness: calculates the social cost + regret penalty from a single sample.
#   - GetPrompts: (optional) a helper for prompt-based generation, not shown here.
from .get_instance import GetData, calc_fitness
from .prompts import GetPrompts


class MLS:
    """
    Multi-Facility Location Scoring (MLS)

    This class provides methods to:
    (1) Load and store facility-location data.
    (2) Evaluate a user-provided mechanism via its empirical fitness (Equation (1) in the paper).
    (3) Return the average fitness (lower is better).
    """

    def __init__(self, k: int = 2, epsilon: float = 0.01) -> None:
        """
        Constructor for MLS.

        Args:
            k (int): The number of facilities to be located. Defaults to 2.
            epsilon (float): The threshold (ε) used in Equation (2) of the paper
                             to decide whether to penalize a mechanism for incurring regret.
        """
        self.k: int = k
        self.epsilon: float = epsilon

        # Load training/test data from an assumed 'GetData' class.
        gd = GetData()
        train, test = gd.get_instances()

        # We store both training data and test data in a single dictionary (self.instances).
        # Each key in self.instances should map to:
        #   { "peaks": 2D array, "misreports": 3D array, "weights": 1D or 2D array }
        self.instances = {}
        self.instances.update(train)
        # self.instances.update(test)  # Optionally include test data if desired

        # If there is a prompt-generation helper, we store it here. Not mandatory for scoring.
        try:
            self.prompts = GetPrompts()
        except ImportError:
            self.prompts = None

    def evaluateHeuristic(self, alg_module: types.ModuleType) -> float:
        """
        evaluateHeuristic

        Core routine for evaluating a user-defined facility-location mechanism.

        The user-provided mechanism must define a function:
            def get_locations(samples: list[float]) -> list[float]:
        that returns a list of facility locations.

        We compute the average fitness over all samples in self.instances,
        following Equation (1) + (2) from the paper.

        Args:
            alg_module (types.ModuleType): A Python module object containing
                                           the user-defined `get_locations` function.

        Returns:
            float: The average (social cost + penalty) across all loaded samples.
        """
        # Must have a function named 'get_locations'
        place_func = getattr(alg_module, "get_locations", None)
        if place_func is None:
            raise ValueError("Module must define get_locations(samples).")

        total_fit: float = 0.0
        count: int = 0

        # Iterate over each problem setting in self.instances
        for key, val in self.instances.items():
            peaks = val.get("peaks")
            if peaks is None or peaks.ndim != 2:
                # No valid data
                continue

            misreports = val.get("misreports", None)
            weights = val.get("weights", None)
            num_samples, n_agents = peaks.shape

            # For each sample in this problem set
            for i in range(num_samples):
                # Extract the i-th row of agent peaks (one scenario)
                p = peaks[i]

                # If 'weights' is a 1D array, we assume the same weights for all samples.
                # If 'weights' is a 2D array, we pick the i-th row.
                if weights is None:
                    w = np.ones(n_agents, dtype=float)
                elif weights.ndim == 1:
                    w = weights
                else:
                    w = weights[i]

                # If we have misreports, they are typically shaped (num_samples, n_agents, M)
                #   where M is the number of misreports per agent. We pick misreports for row i.
                m = None
                if isinstance(misreports, np.ndarray) and misreports.ndim == 3:
                    if misreports.shape[0] == num_samples:
                        m = misreports[i]

                # calc_fitness is a function that implements the logic from the paper:
                #   1) Evaluate the total distance to the nearest facility (social cost).
                #   2) Evaluate the regret if any agent can gain by misreporting.
                #   3) Impose a penalty if regret exceeds self.epsilon.
                fit_i = calc_fitness(
                    peaks_row=p,
                    misreports_row=m,
                    weights_row=w,
                    mechanism_func=place_func,
                    k=self.k,
                    epsilon=self.epsilon,
                )
                total_fit += fit_i
                count += 1

        if count == 0:
            warnings.warn("No valid samples; returning a large fitness value.")
            return float("inf")

        # Return the average fitness across all samples
        return total_fit / count

    def evaluate(self, code_string: str) -> Optional[float]:
        """
        Compiles and evaluates a user-provided mechanism code.

        Args:
            code_string (str): A Python code snippet that must define a
                               function get_locations(samples) -> list[float].

        Returns:
            Optional[float]: The average fitness for the user mechanism if valid,
                             or None if there is an error (e.g., compile-time error).
        """
        try:
            print("Evaluating user code:\n", code_string)

            # Create a new module object to hold the user code.
            mod = types.ModuleType("user_heuristic")
            exec(
                code_string, mod.__dict__
            )  # Execute user code in the module's namespace

            # Insert this dynamic module into sys.modules to ensure it can be imported if needed
            sys.modules[mod.__name__] = mod

            # Now evaluate the user's heuristic using the method above
            return self.evaluateHeuristic(mod)

        except Exception as e:
            print("Evaluation error:", e)
            return None
