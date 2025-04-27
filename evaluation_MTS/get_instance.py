from __future__ import annotations
import os
import pickle
import warnings
from typing import Dict, Tuple, Any, Optional

import numpy as np

# --- Configuration ---
# Assumes data is in a 'data' subdirectory relative to this file's location.
# Adjust if your data is elsewhere.
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(CURRENT_DIR, "data")
TRAIN_DATA_FILE: str = os.path.join(DATA_DIR, "all_data_train.pkl")
TEST_DATA_FILE: str = os.path.join(DATA_DIR, "all_data_test.pkl")

# Keys expected to be found in the data dictionaries
PEAK_DATA_KEY: str = "peaks"  # Expected key for peak data in the pkl dict
WEIGHTS_DATA_KEY: str = "weights"  # Expected key for weights data (optional)
MISREPORT_DATA_KEY: str = "misreports"  # Expected key for misreport data

# !!! Crucial: Set this based on your misreport data generation !!!
# How many misreport scenarios exist for each original sample?
# If your shape was (1000, 5) and flattened is (10000, 5), this should be 10.
NUM_MISREPORTS_PER_AGENT_OR_SCENARIO: int = 10


def calc_weighted_social_cost(
    agent_peaks: np.ndarray, facility_locations: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculates the (weighted) social cost for a single instance.

    Args:
      agent_peaks: A NumPy array of shape (n,) representing the agents' peaks.
      facility_locations: A NumPy array/list of shape (k,) representing facility locations.
      weights: A NumPy array of shape (n,) representing the weights/importance of each agent.

    Returns:
      The weighted social cost (float). Returns float('inf') if facility_locations or agent_peaks is invalid.
    """
    if (
        not isinstance(facility_locations, (list, np.ndarray))
        or len(facility_locations) == 0
    ):
        return float("inf")
    if not isinstance(agent_peaks, (list, np.ndarray)) or len(agent_peaks) == 0:
        return float("inf")

    agent_peaks_arr = np.array(agent_peaks, dtype=float)
    facility_locations_arr = np.array(facility_locations, dtype=float)
    weights_arr = np.array(weights, dtype=float)

    # Normalize weights so they sum to 1 if we intend to average
    total_w = np.sum(weights_arr)
    if total_w <= 0:
        # If sum of weights is zero or negative, treat cost as invalid
        return float("inf")
    normalized_weights = weights_arr / total_w

    # Compute distance from each agent to its nearest facility
    # agent_peaks_arr: shape (n,)
    # facility_locations_arr: shape (k,)
    # distances: shape (n, k)
    distances = np.abs(agent_peaks_arr[:, np.newaxis] - facility_locations_arr)
    min_distances = np.min(distances, axis=1)

    # Weighted average cost
    weighted_cost = np.sum(min_distances * normalized_weights)
    return float(weighted_cost)


def calc_max_regret(
    peaks: np.ndarray,
    misreports: Optional[np.ndarray],
    weights: np.ndarray,
    place_func: callable,
    k: int,
) -> float:
    """
    Calculates the maximum regret for a single instance.

    Args:
      peaks: True peaks for this instance, shape (n,).
      misreports: Misreports for this instance (if any), shape (n, M) expected.
                  If None, we treat the max regret as 0.
      weights: Weights for each agent, shape (n,).
      place_func: A callable that places k facilities given (peaks, weights, k).
      k: Number of facilities.

    Returns:
      The maximum regret (float). If shape checks fail, returns 0.0 or warns.
    """
    n: int = len(peaks)
    if misreports is None:
        return 0.0

    # We expect misreports to be shape (n, M)
    if len(misreports.shape) != 2 or misreports.shape[0] != n:
        warnings.warn(
            f"Unexpected misreports shape: {misreports.shape} (expected ({n}, M)). "
            "Skipping regret computation by returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    # Place facilities using the original peaks
    fac_original = place_func(peaks, weights, k)
    if not isinstance(fac_original, (list, np.ndarray)) or len(fac_original) == 0:
        warnings.warn(
            "Invalid facility placement for original peaks. Regret=0.", stacklevel=2
        )
        return 0.0
    fac_original_arr = np.array(fac_original, dtype=float)

    # Original cost for each agent
    dist_original = np.abs(peaks[:, np.newaxis] - fac_original_arr)  # shape (n, k)
    original_costs_per_agent = np.min(dist_original, axis=1)

    max_r: float = 0.0
    # Check each agent's misreports
    for i in range(n):
        cost_i_orig = original_costs_per_agent[i]
        true_peak_i = peaks[i]

        # Loop over each possible misreport for agent i
        for rep_idx in range(misreports.shape[1]):
            new_peaks = np.copy(peaks)
            new_peaks[i] = misreports[i, rep_idx]  # single agent misreport

            # New facility placement
            fac_new = place_func(new_peaks, weights, k)
            if not isinstance(fac_new, (list, np.ndarray)) or len(fac_new) == 0:
                # If the mechanism fails, assume no gain
                continue

            fac_new_arr = np.array(fac_new, dtype=float)
            # Agent i's new cost using its true peak
            cost_i_new = np.min(np.abs(true_peak_i - fac_new_arr))
            # Regret = original cost - new cost
            regret_i = cost_i_orig - cost_i_new
            if regret_i > max_r:
                max_r = regret_i

    return float(max_r)


def calc_fitness(
    peaks: np.ndarray,
    misreports: Optional[np.ndarray],
    weights: np.ndarray,
    place_func: callable,
    k: int,
    epsilon: float = 0.01,
) -> float:
    """
    Computes the fitness of a mechanism on a single instance, defined as:
      fitness = weighted social cost + penalty,
      where penalty = 1 if max_regret > epsilon, else 0.

    Args:
      peaks: True peaks for this instance, shape (n,).
      misreports: Misreports for this instance, shape (n, M). Can be None.
      weights: Weights of each agent, shape (n,).
      place_func: A callable that outputs facility locations given (peaks, weights, k).
      k: Number of facilities to place.
      epsilon: A small threshold for deciding if the mechanism triggers penalty.

    Returns:
      The fitness value (float). High values are "worse".
      If facility placement fails, returns 9999.0 as an indicator.
    """
    fac_locs = place_func(peaks, weights, k)
    if not isinstance(fac_locs, (list, np.ndarray)) or len(fac_locs) == 0:
        warnings.warn(
            "Mechanism returned invalid facility locations. Fitness=9999.", stacklevel=2
        )
        return 9999.0

    fac_locs_arr = np.array(fac_locs, dtype=float)
    cost = calc_weighted_social_cost(peaks, fac_locs_arr, weights)
    if np.isnan(cost) or np.isinf(cost):
        warnings.warn("Cost is NaN or Inf. Fitness=9999.", stacklevel=2)
        return 9999.0

    max_r = calc_max_regret(peaks, misreports, weights, place_func, k)
    penalty = 1.0 if max_r > epsilon else 0.0
    return float(cost + penalty)


def place_facilities_quantiles(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """
    Example baseline: place k facilities at k evenly spaced quantiles of the agent_peaks.
    Ignores weights.
    """
    n_agents: int = len(agent_peaks)
    if n_agents == 0 or k <= 0:
        return np.array([], dtype=float)
    if k >= n_agents:
        unique_peaks = np.unique(agent_peaks)
        return unique_peaks[:k]

    sorted_peaks = np.sort(agent_peaks)
    indices = np.linspace(0, 1, k + 2)[1:-1]
    # If your NumPy is older and lacks 'method="linear"', remove the argument
    locations = np.quantile(sorted_peaks, indices, method="linear")
    return locations.astype(float)


def dictatorial_rule(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int, dictator_index: int
) -> np.ndarray:
    """
    Example baseline: place all k facilities at the location of the chosen 'dictator' agent.
    """
    n_agents: int = len(agent_peaks)
    if n_agents == 0 or k <= 0:
        return np.array([], dtype=float)
    if dictator_index < 0 or dictator_index >= n_agents:
        warnings.warn("Invalid dictator index. Using median instead.", stacklevel=2)
        median_loc = float(np.median(agent_peaks))
        return np.array([median_loc] * k, dtype=float)
    return np.array([agent_peaks[dictator_index]] * k, dtype=float)


def best_dictatorial_rule(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """
    Example baseline: choose the best dictator among all agents
    to minimize the social cost for this single instance.
    """
    n_agents: int = len(agent_peaks)
    if n_agents == 0 or k <= 0:
        return np.array([], dtype=float)

    best_cost: float = float("inf")
    best_locs: Any = None
    total_w = np.sum(weights)
    if total_w <= 0:
        warnings.warn("Sum of weights <= 0. Returning empty facilities.")
        return np.array([], dtype=float)

    # For consistent comparison, normalize weights for cost calculation
    normalized_weights = weights / total_w

    for i in range(n_agents):
        candidate_locs = dictatorial_rule(agent_peaks, weights, k, i)
        c = calc_weighted_social_cost(agent_peaks, candidate_locs, normalized_weights)
        if c < best_cost:
            best_cost = c
            best_locs = candidate_locs

    if not isinstance(best_locs, (list, np.ndarray)) or len(best_locs) == 0:
        # fallback to median-based if something went wrong
        med = float(np.median(agent_peaks))
        return np.array([med] * k, dtype=float)
    return best_locs.astype(float)


def constant_rule(agent_peaks: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """
    Example baseline: place k facilities at fixed, evenly spaced locations in [0,1].
    """
    if k <= 0:
        return np.array([], dtype=float)
    positions = [(i + 1.0) / (k + 1.0) for i in range(k)]
    return np.array(positions, dtype=float)


class GetData:
    """
    Loads train/test data from .pkl files specified by paths.
    Handles a nested dictionary structure: { (dist, n_agents): {...}, ... }
    Each dictionary sub-entry is expected to contain:
      - 'peaks': np.ndarray of shape (num_samples, n_agents)
      - 'misreports': (optional) np.ndarray of shape (num_samples, n_agents, M)
      - 'weights': (not read from .pkl here, but assigned based on n_agents)
      - any other fields as desired.

    Example of the final structure after loading:
      self.train_data[(dist, n)] = {
        'peaks': np.ndarray of shape (num_samples, n_agents),
        'weights': np.ndarray of shape (n_agents,),
        'misreports': np.ndarray of shape (num_samples, n_agents, M) or None,
        ...
      }
    """

    def __init__(
        self,
        data_train_path: str = TRAIN_DATA_FILE,
        data_test_path: str = TEST_DATA_FILE,
        num_misreports_per_item: int = NUM_MISREPORTS_PER_AGENT_OR_SCENARIO,
    ) -> None:
        """
        Args:
          data_train_path: File path for the training set .pkl
          data_test_path: File path for the testing set .pkl
          num_misreports_per_item: The number of misreports per sample you expect.
        """
        self.train_path: str = data_train_path
        self.test_path: str = data_test_path
        self.num_misreports: int = num_misreports_per_item

        self.train_data: Dict[Any, Dict[str, np.ndarray]] = {}
        self.test_data: Dict[Any, Dict[str, np.ndarray]] = {}

        self.load_data()

    def load_data(self) -> None:
        """
        Loads and preprocesses the data from the specified pickle files,
        storing them in self.train_data and self.test_data.
        """
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train data not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found: {self.test_path}")

        print(f"Loading train data from: {self.train_path}")
        with open(self.train_path, "rb") as f:
            self.train_data = pickle.load(f)
        print("Train data loaded.")

        print(f"Loading test data from: {self.test_path}")
        with open(self.test_path, "rb") as f:
            self.test_data = pickle.load(f)
        print("Test data loaded.")

        print("Preprocessing training data...")
        self._preprocess_dataset(self.train_data)
        print("Preprocessing test data...")
        self._preprocess_dataset(self.test_data)
        print("Data preprocessing complete.")

    def _preprocess_dataset(self, dataset: Dict[Any, Dict[str, Any]]) -> None:
        """
        Internal method to preprocess the dataset dictionary:
          - Convert 'peaks' to float np.ndarray of shape (num_samples, n_agents).
          - Assign 'weights' based on n_agents, e.g. 1st agent weight=5 for n=5.
          - Reshape 'misreports' from flatten if needed.
          - Remove invalid entries.
        """
        keys_to_remove = []
        for key, val in dataset.items():
            if not isinstance(val, dict):
                warnings.warn(
                    f"Value for key {key} is not a dict; removing from dataset.",
                    stacklevel=2,
                )
                keys_to_remove.append(key)
                continue

            valid_instance = True

            # 1) Convert peaks
            if PEAK_DATA_KEY in val and val[PEAK_DATA_KEY] is not None:
                try:
                    peaks_array = np.array(val[PEAK_DATA_KEY], dtype=float)
                    if peaks_array.ndim != 2:
                        warnings.warn(
                            f"Peaks for {key} are not 2D (found shape {peaks_array.shape}). Removing key.",
                            stacklevel=2,
                        )
                        valid_instance = False
                    else:
                        val[PEAK_DATA_KEY] = peaks_array
                except Exception as e:
                    warnings.warn(
                        f"Failed to convert peaks for {key} to float array: {e}",
                        stacklevel=2,
                    )
                    valid_instance = False
            else:
                warnings.warn(
                    f"No valid '{PEAK_DATA_KEY}' in {key}; removing key.", stacklevel=2
                )
                valid_instance = False

            if not valid_instance:
                keys_to_remove.append(key)
                continue

            num_samples, n_agents = val[PEAK_DATA_KEY].shape

            # 2) Assign weights based on n_agents
            if n_agents == 5:
                # E.g., 1st agent weight=5, rest=1
                val[WEIGHTS_DATA_KEY] = np.array(
                    [5] + [1] * (n_agents - 1), dtype=float
                )
            elif n_agents == 10:
                # E.g., first 2 agents weight=5, rest=1
                val[WEIGHTS_DATA_KEY] = np.array(
                    [5, 5] + [1] * (n_agents - 2), dtype=float
                )
            else:
                # Default: all weights=1
                val[WEIGHTS_DATA_KEY] = np.ones(n_agents, dtype=float)

            # 3) Convert and reshape misreports
            misr_raw = val.get(MISREPORT_DATA_KEY, None)
            if misr_raw is not None:
                try:
                    misr_arr = np.array(misr_raw, dtype=float)
                    expected_flat = num_samples * self.num_misreports
                    target_shape = (num_samples, n_agents, self.num_misreports)

                    if misr_arr.shape == target_shape:
                        val[MISREPORT_DATA_KEY] = misr_arr
                    elif misr_arr.shape == (expected_flat, n_agents):
                        # Reshape from (num_samples*M, n_agents) -> (num_samples, n_agents, M)
                        intermediate = misr_arr.reshape(
                            (num_samples, self.num_misreports, n_agents)
                        )
                        val[MISREPORT_DATA_KEY] = intermediate.transpose(0, 2, 1)
                    else:
                        warnings.warn(
                            f"Misreports for {key} have shape {misr_arr.shape}, "
                            f"expected {target_shape} or ({expected_flat}, {n_agents}). Setting to None.",
                            stacklevel=2,
                        )
                        val[MISREPORT_DATA_KEY] = None
                except Exception as e:
                    warnings.warn(
                        f"Failed to process misreports for {key}: {e}. Setting to None.",
                        stacklevel=2,
                    )
                    val[MISREPORT_DATA_KEY] = None
            else:
                val[MISREPORT_DATA_KEY] = None

        # 4) Remove invalid items
        for bad_key in keys_to_remove:
            del dataset[bad_key]

    def get_instances(
        self,
    ) -> Tuple[Dict[Any, Dict[str, np.ndarray]], Dict[Any, Dict[str, np.ndarray]]]:
        """
        Returns the processed (train_data, test_data).

        Additionally, here we confirm that each valid misreports array in both
        train_data and test_data has the last dimension matching self.num_misreports.
        """
        # --- Confirm misreport shapes for the training set ---
        for key, val in self.train_data.items():
            misreports = val.get(MISREPORT_DATA_KEY, None)
            if misreports is not None:
                # Expect shape = (num_samples, n_agents, self.num_misreports)
                if misreports.ndim == 3 and misreports.shape[2] != self.num_misreports:
                    warnings.warn(
                        f"[Train] Key={key}: Found misreports.shape={misreports.shape} "
                        f"but expected 3rd dim={self.num_misreports}.",
                        stacklevel=2,
                    )

        # --- Confirm misreport shapes for the testing set ---
        for key, val in self.test_data.items():
            misreports = val.get(MISREPORT_DATA_KEY, None)
            if misreports is not None:
                # Expect shape = (num_samples, n_agents, self.num_misreports)
                if misreports.ndim == 3 and misreports.shape[2] != self.num_misreports:
                    warnings.warn(
                        f"[Test] Key={key}: Found misreports.shape={misreports.shape} "
                        f"but expected 3rd dim={self.num_misreports}.",
                        stacklevel=2,
                    )

        return self.train_data, self.test_data
