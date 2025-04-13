import os
import pickle
import numpy as np


def place_facilities_quantiles(peaks, weights, k=2):
    """
    A simple baseline for demonstration: place k facilities
    at evenly spaced quantiles of the agent peaks (ignoring weights).
    """
    # Ensure peaks is float
    peaks = peaks.astype(float, copy=False)
    sorted_peaks = np.sort(peaks)
    n = len(peaks)
    facilities = []
    for i in range(k):
        frac = (i + 1) / (k + 1)  # e.g., 1/3, 2/3 for k=2
        idx = int(frac * (n - 1))
        facilities.append(sorted_peaks[idx])
    return np.array(facilities, dtype=float)


def calc_weighted_social_cost(peaks, facilities, weights):
    """
    Sum of distances from each agent to its nearest facility, weighted by weights[i].
    """
    total = 0.0
    for i, p in enumerate(peaks):
        dists = np.abs(facilities - p)
        total += weights[i] * dists.min()
    return total


def calc_max_regret(peaks, misreports, weights, place_func, k):
    """
    Given:
      - peaks: shape (n,)
      - misreports: shape depends on your data structure, e.g. (n, 10) or (n*10, n).
      - place_func: user or baseline function that places facilities
      - k: number of facilities
    Returns the maximum difference (original_cost - new_cost) an agent can achieve by misreporting.
    """
    n = len(peaks)

    # Convert all to float
    peaks = peaks.astype(float, copy=False)
    weights = weights.astype(float, copy=False)

    fac_original = place_func(peaks, weights, k)
    original_costs = np.zeros(n, dtype=float)
    for i in range(n):
        original_costs[i] = weights[i] * np.abs(fac_original - peaks[i]).min()

    max_r = 0.0
    if misreports is None:
        return max_r

    # Example: misreports is shape (n, 10), i.e. for agent i we have 10 alt. peaks
    if len(misreports.shape) == 2 and misreports.shape[0] == n:
        misreports = misreports.astype(float, copy=False)
        for i in range(n):
            cost_i_orig = original_costs[i]
            for rep_idx in range(misreports.shape[1]):
                new_peaks = np.copy(peaks)
                new_peaks[i] = misreports[i, rep_idx]
                fac_new = place_func(new_peaks, weights, k)
                cost_i_new = weights[i] * np.abs(fac_new - new_peaks[i]).min()

                regret = cost_i_orig - cost_i_new
                if regret > max_r:
                    max_r = regret

    return max_r


def calc_fitness(peaks, misreports, weights, place_func, k, epsilon=0.01):
    """
    Computes the fitness = (weighted social cost) + penalty, where
      penalty = 1 if max_regret > epsilon, else 0.
    """
    # Convert to float
    peaks = peaks.astype(float, copy=False)
    weights = weights.astype(float, copy=False)

    # Place facilities
    facilities = place_func(peaks, weights, k)

    # Weighted social cost
    cost = calc_weighted_social_cost(peaks, facilities, weights)

    # Max regret
    max_r = (
        calc_max_regret(peaks, misreports, weights, place_func, k)
        if misreports is not None
        else 0.0
    )
    penalty = 1.0 if max_r > epsilon else 0.0
    return cost + penalty


class GetData:
    """
    Loads train/test data from your data/ .pkl files (all_data_train.pkl / all_data_test.pkl),
    or any other directory. Also provides a baseline function.

    We store each problem instance in a dictionary with keys:
       ('distribution', n) -> {
          'peaks': shape (num_samples, n),
          'weights': shape (n,) or (num_samples, n),
          'misreports': depends on your structure,
          'k': optional integer
       }
    or similarly.
    """

    def __init__(
        self,
        data_train_path="eoh/src/eoh/problems/optimization/MLS/data/all_data_train.pkl",
        data_test_path="eoh/src/eoh/problems/optimization/MLS/data/all_data_test.pkl",
    ):
        self.train_path = data_train_path
        self.test_path = data_test_path
        self.train_data = {}
        self.test_data = {}
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train data not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found: {self.test_path}")

        with open(self.train_path, "rb") as f:
            self.train_data = pickle.load(f)

        with open(self.test_path, "rb") as f:
            self.test_data = pickle.load(f)

        # -- Convert loaded data to float dtype, plus do basic shape checks
        self._convert_to_float_and_check(self.train_data)
        self._convert_to_float_and_check(self.test_data)

    def _convert_to_float_and_check(self, dataset):
        """Convert 'peaks','weights','misreports' to float and do minimal shape checks."""
        for key, val in dataset.items():
            # Convert peaks
            if "peaks" in val and val["peaks"] is not None:
                val["peaks"] = val["peaks"].astype(float)
            # Convert weights
            if "weights" in val and val["weights"] is not None:
                val["weights"] = val["weights"].astype(float)
            # Convert misreports
            if "misreports" in val and val["misreports"] is not None:
                val["misreports"] = val["misreports"].astype(float)

            # (Optional) Basic shape check: if we have n from the shape of peaks,
            # ensure weights/misreports line up in their last dimension
            if "peaks" in val and val["peaks"] is not None:
                peaks_arr = val["peaks"]
                n = peaks_arr.shape[1]  # second dimension is # of agents
                # check weights shape
                if "weights" in val and val["weights"] is not None:
                    wshape = val["weights"].shape
                    if len(wshape) == 1:
                        if wshape[0] != n:
                            print(
                                f"Warning: mismatch in n for {key}: weights={wshape}, peaks n={n}"
                            )
                    elif len(wshape) == 2:
                        if wshape[1] != n:
                            print(
                                f"Warning: mismatch in n for {key}: weights shape={wshape}, peaks n={n}"
                            )
                # check misreports shape if you want to ensure final dimension matches n
                # ...

    def get_instances(self):
        """
        Return (train_data, test_data).
        Each is a dict of problem instances described above.
        """
        return self.train_data, self.test_data

    def baseline_evaluate(self, data_dict, k=2, epsilon=0.01):
        """
        Example of using the baseline place_facilities_quantiles as a reference.
        We compute the average fitness across all samples in 'data_dict'.
        """
        total = 0.0
        count = 0
        for key, val in data_dict.items():
            if "peaks" not in val:
                continue
            peaks_arr = val["peaks"]
            misr_arr = val["misreports"] if "misreports" in val else None
            # weights could be shape (n,) or (num_samples,n)
            weights_info = val.get("weights", None)

            num_samples = peaks_arr.shape[0]
            n_agents = peaks_arr.shape[1]
            for i in range(num_samples):
                peaks_i = peaks_arr[i]
                # handle weights
                if weights_info is not None:
                    if len(weights_info.shape) == 1:
                        weights_i = weights_info
                    else:
                        weights_i = weights_info[i]
                else:
                    # uniform
                    weights_i = np.ones(n_agents, dtype=float)

                # handle misreports
                if misr_arr is not None:
                    if len(misr_arr.shape) == 3 and misr_arr.shape[0] == num_samples:
                        misreports_i = misr_arr[i]  # shape (n, 10) or so
                    else:
                        misreports_i = None
                else:
                    misreports_i = None

                fit_i = calc_fitness(
                    peaks_i,
                    misreports_i,
                    weights_i,
                    place_facilities_quantiles,
                    k,
                    epsilon=epsilon,
                )
                total += fit_i
                count += 1

        return total / count if count > 0 else 0.0
