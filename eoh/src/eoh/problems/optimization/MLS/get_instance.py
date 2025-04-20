import os
import pickle
import numpy as np
import warnings

# --- Configuration ---
# Assumes data is in a 'data' subdirectory relative to this file's location
# Adjust if your data is elsewhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "all_data_train.pkl")
TEST_DATA_FILE = os.path.join(DATA_DIR, "all_data_test.pkl")

PEAK_DATA_KEY: str = "peaks"  # Expected key for peak data in the pkl dict
WEIGHTS_DATA_KEY: str = "weights"  # Expected key for weights data (optional)
MISREPORT_DATA_KEY: str = "misreports"  # Expected key for misreport data

# !!! Crucial: Set this based on your misreport data generation !!!
# How many misreport scenarios exist for each original sample?
# If original shape was (1000, 5) and flattened is (10000, 5), this should be 10.
NUM_MISREPORTS_PER_AGENT_OR_SCENARIO: int = 10


# --- Helper Functions ---


def calc_weighted_social_cost(
    agent_peaks: np.ndarray, facility_locations: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculates the weighted social cost for one instance.
    Assumes weights are normalized or represent the desired relative importance.
    """
    if (
        not isinstance(facility_locations, (list, np.ndarray))
        or len(facility_locations) == 0
    ):
        return float("inf")  # Invalid locations
    if not isinstance(agent_peaks, (list, np.ndarray)) or len(agent_peaks) == 0:
        return float("inf")  # Invalid peaks

    agent_peaks_arr = np.array(agent_peaks, dtype=float)
    facility_locations_arr = np.array(facility_locations, dtype=float)
    weights_arr = np.array(weights, dtype=float)

    # Ensure weights sum to 1 for averaging, or adjust calculation if they represent absolute importance
    normalized_weights = weights_arr / np.sum(weights_arr)

    # Calculate distance from each agent to their nearest facility
    # Expand dims for broadcasting: agent_peaks (n, 1), facility_locations (k,) -> distances (n, k)
    distances = np.abs(agent_peaks_arr[:, np.newaxis] - facility_locations_arr)
    min_distances = np.min(distances, axis=1)  # Shape (n,)

    # Calculate weighted average cost
    weighted_cost = np.sum(min_distances * normalized_weights)
    return float(weighted_cost)


def calc_max_regret(
    peaks: np.ndarray,
    misreports: np.ndarray | None,
    weights: np.ndarray,
    place_func,
    k: int,
) -> float:
    """
    Calculates the maximum regret for a single instance.

    Args:
      peaks: True peaks for the instance, shape (n,).
      misreports: Misreports for this instance, expected shape (n, M).
          If None, we return 0.0.
      weights: Agent weights, shape (n,).
      place_func: A callable place_func(peaks, weights, k) -> np.ndarray
      k: Number of facilities.

    Returns:
      The maximum regret for this instance (float).
    """
    n: int = len(peaks)
    if misreports is None:
        return 0.0

    # Ensure expected shape (n, M)
    if not (len(misreports.shape) == 2 and misreports.shape[0] == n):
        warnings.warn(
            f"Unexpected misreports shape {misreports.shape} in calc_max_regret. Expected ({n}, M). Skipping regret.",
            stacklevel=2,
        )
        return 0.0

    # Ensure float types
    peaks_arr = peaks.astype(float, copy=False)
    weights_arr = weights.astype(float, copy=False)
    misreports_arr = misreports.astype(float, copy=False)

    # Place facilities for original peaks
    fac_original = place_func(peaks_arr, weights_arr, k)
    if not isinstance(fac_original, (list, np.ndarray)) or len(fac_original) == 0:
        warnings.warn(
            "Original placement function returned invalid facilities. Regret=0.",
            stacklevel=2,
        )
        return 0.0
    fac_original_arr = np.array(fac_original, dtype=float)

    # Cost for each agent (with original placements, based on true peaks)
    original_costs_per_agent = np.min(
        np.abs(peaks_arr[:, np.newaxis] - fac_original_arr), axis=1
    )

    max_r: float = 0.0

    # Iterate through each agent misreporting
    for i in range(n):
        cost_i_orig = original_costs_per_agent[i]
        agent_true_peak = peaks_arr[i]

        # Iterate through each misreport for agent i
        for rep_idx in range(misreports_arr.shape[1]):  # shape (n, M)
            new_peaks = np.copy(peaks_arr)
            new_peaks[i] = misreports_arr[i, rep_idx]  # Agent i misreports

            # Get new facility locations based on the misreport
            fac_new = place_func(new_peaks, weights_arr, k)
            if not isinstance(fac_new, (list, np.ndarray)) or len(fac_new) == 0:
                # Mechanism failed with this misreport, assume no gain
                continue
            fac_new_arr = np.array(fac_new, dtype=float)

            # Calculate agent i's cost with the *new* facilities but using their *true* peak
            cost_i_new = np.min(np.abs(agent_true_peak - fac_new_arr))

            # Regret = gain = original_cost - new_cost
            regret = cost_i_orig - cost_i_new
            if regret > max_r:
                max_r = regret

    return float(max_r)


def calc_fitness(
    peaks: np.ndarray,
    misreports: np.ndarray | None,
    weights: np.ndarray,
    place_func,
    k: int,
    epsilon: float = 0.01,
) -> float:
    """
    Computes the fitness = (weighted social cost) + penalty, where
      penalty = 1 if max_regret > epsilon, else 0.
    """
    # Ensure float types
    peaks_arr = peaks.astype(float, copy=False)
    weights_arr = weights.astype(float, copy=False)

    # Place facilities using the provided function
    facilities = place_func(peaks_arr, weights_arr, k)
    if not isinstance(facilities, (list, np.ndarray)) or len(facilities) == 0:
        warnings.warn(
            "Placement function returned invalid facilities. Returning high fitness.",
            stacklevel=2,
        )
        return 9999.0
    facilities_arr = np.array(facilities, dtype=float)

    # Calculate Weighted social cost
    cost = calc_weighted_social_cost(peaks_arr, facilities_arr, weights_arr)

    # Calculate Max regret
    max_r = calc_max_regret(peaks_arr, misreports, weights_arr, place_func, k)
    penalty = 1.0 if max_r > epsilon else 0.0

    # Check for NaN or Inf in cost
    if np.isnan(cost) or np.isinf(cost):
        warnings.warn(
            f"NaN or Inf cost detected for peaks: {peaks_arr}, facilities: {facilities_arr}. Returning high fitness.",
            stacklevel=2,
        )
        return 9999.0

    return float(cost + penalty)


# --- Data Loading Class ---


class GetData:
    """
    Loads train/test data from .pkl files specified by paths.
    Handles nested dictionary structure: { (dist, n_agents) : {'peaks': ndarray, 'misreports': ndarray, ...}, ...}
    Attempts to reshape misreports if they appear flattened.
    Provides baseline evaluation methods.
    """

    def __init__(
        self,
        data_train_path: str = TRAIN_DATA_FILE,
        data_test_path: str = TEST_DATA_FILE,
        num_misreports_per_item: int = NUM_MISREPORTS_PER_AGENT_OR_SCENARIO,
    ):
        self.train_path = data_train_path
        self.test_path = data_test_path
        self.num_misreports = num_misreports_per_item
        self.train_data: dict = {}
        self.test_data: dict = {}
        self.load_data()

    def load_data(self) -> None:
        """Loads and preprocesses data from the specified pickle files."""
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

        # Preprocess
        print("Preprocessing training data...")
        self._preprocess_dataset(self.train_data)
        print("Preprocessing test data...")
        self._preprocess_dataset(self.test_data)
        print("Data preprocessing complete.")

    def _preprocess_dataset(self, dataset: dict) -> None:
        """
        Convert types and attempt to reshape misreports for a dataset dict.
        We no longer read weights from the pickle; 而是根据 n_agents 直接设置。
        """
        keys_to_remove = []
        for key, val in dataset.items():
            if not isinstance(val, dict):
                warnings.warn(f"Value for key {key} is not a dict, skipping.")
                keys_to_remove.append(key)
                continue

            valid_instance = True

            # 1. Convert peaks
            if PEAK_DATA_KEY in val and val[PEAK_DATA_KEY] is not None:
                try:
                    peaks_array = np.array(val[PEAK_DATA_KEY], dtype=float)
                    if peaks_array.ndim != 2:
                        warnings.warn(f"Peaks for {key} are not 2D. Skipping key.")
                        valid_instance = False
                    else:
                        val[PEAK_DATA_KEY] = peaks_array
                except Exception as e:
                    warnings.warn(
                        f"Could not convert peaks for {key} to float array: {e}. Skipping key."
                    )
                    valid_instance = False
            else:
                warnings.warn(
                    f"'{PEAK_DATA_KEY}' not found or is None for key {key}. Skipping key."
                )
                valid_instance = False

            if not valid_instance:
                keys_to_remove.append(key)
                continue

            num_samples, n_agents = val[PEAK_DATA_KEY].shape

            # 2. 根据 n_agents 设置 agent weights
            if n_agents == 5:
                # n=5 时，第一个 agent 权重 5，其余 1
                val[WEIGHTS_DATA_KEY] = np.array(
                    [5] + [1] * (n_agents - 1), dtype=float
                )
            elif n_agents == 10:
                # n=10 时，前两个 agent 权重 5，其余 1
                val[WEIGHTS_DATA_KEY] = np.array(
                    [5, 5] + [1] * (n_agents - 2), dtype=float
                )
            else:
                # 其他情况都用均匀权重 1
                val[WEIGHTS_DATA_KEY] = np.ones(n_agents, dtype=float)

            # 3. Convert and reshape misreports
            if MISREPORT_DATA_KEY in val and val[MISREPORT_DATA_KEY] is not None:
                try:
                    misreports_raw = np.array(val[MISREPORT_DATA_KEY], dtype=float)
                    expected_flat = num_samples * self.num_misreports
                    target_shape = (num_samples, n_agents, self.num_misreports)

                    if misreports_raw.shape == target_shape:
                        val[MISREPORT_DATA_KEY] = misreports_raw
                    elif misreports_raw.shape == (expected_flat, n_agents):
                        # Reshape from (num_samples * M, n_agents) -> (num_samples, n_agents, M)
                        intermediate = misreports_raw.reshape(
                            (num_samples, self.num_misreports, n_agents)
                        )
                        val[MISREPORT_DATA_KEY] = intermediate.transpose(0, 2, 1)
                    else:
                        warnings.warn(
                            f"Misreports for {key} have unexpected shape {misreports_raw.shape}. Setting to None."
                        )
                        val[MISREPORT_DATA_KEY] = None

                except Exception as e:
                    warnings.warn(
                        f"Could not process misreports for {key}: {e}. Setting to None."
                    )
                    val[MISREPORT_DATA_KEY] = None
            else:
                val[MISREPORT_DATA_KEY] = None

        # 4. 删除所有无效的实例
        for bad_key in keys_to_remove:
            del dataset[bad_key]

    def get_instances(self) -> tuple[dict, dict]:
        """
        Return (train_data, test_data).
        Each is a dict mapping (dist, n_agents) -> {
            'peaks': np.ndarray,
            'weights': np.ndarray,
            'misreports': np.ndarray | None,
            ...
        }.
        """
        return self.train_data, self.test_data
