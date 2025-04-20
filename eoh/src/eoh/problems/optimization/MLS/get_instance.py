import os
import pickle
import numpy as np
import warnings

# --- Configuration ---
# Assumes data is in a 'data' subdirectory relative to this file's location
# Adjust if your data is elsewhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'all_data_train.pkl')
TEST_DATA_FILE = os.path.join(DATA_DIR, 'all_data_test.pkl')

PEAK_DATA_KEY = 'peaks'         # Expected key for peak data in the pkl dict
WEIGHTS_DATA_KEY = 'weights'     # Expected key for weights data (optional)
MISREPORT_DATA_KEY = 'misreports' # Expected key for misreport data

# !!! Crucial: Set this based on your misreport data generation !!!
# How many misreport scenarios exist for each original sample?
# If original shape was (1000, 5) and flattened is (10000, 5), this should be 10.
NUM_MISREPORTS_PER_AGENT_OR_SCENARIO = 10


# --- Helper Functions ---

def calc_weighted_social_cost(agent_peaks, facility_locations, weights):
    """
    Calculates the weighted social cost for one instance.
    Assumes weights are normalized or represent the desired relative importance.
    """
    if not isinstance(facility_locations, (list, np.ndarray)) or len(facility_locations) == 0:
        return np.inf # Invalid locations
    if not isinstance(agent_peaks, (list, np.ndarray)) or len(agent_peaks) == 0:
         return np.inf # Invalid peaks

    agent_peaks = np.array(agent_peaks, dtype=float)
    facility_locations = np.array(facility_locations, dtype=float)
    weights = np.array(weights, dtype=float)

    # Ensure weights sum to 1 for averaging, or adjust calculation if they represent absolute importance
    normalized_weights = weights / np.sum(weights)

    # Calculate distance from each agent to their nearest facility
    # Expand dims for broadcasting: agent_peaks (n, 1), facility_locations (k,) -> distances (n, k)
    distances = np.abs(agent_peaks[:, np.newaxis] - facility_locations)
    min_distances = np.min(distances, axis=1) # Shape (n,)

    # Calculate weighted average cost
    weighted_cost = np.sum(min_distances * normalized_weights)
    return weighted_cost


def calc_max_regret(peaks, misreports, weights, place_func, k):
    """
    Calculates the maximum regret for a single instance.

    Args:
      peaks (np.ndarray): True peaks for the instance, shape (n,).
      misreports (np.ndarray | None): Misreports for this instance.
          Expected shape is (n, num_misreports_per_agent) after potential reshaping.
          If None, regret is 0.
      weights (np.ndarray): Agent weights, shape (n,).
      place_func (callable): The mechanism function place_func(peaks, weights, k).
      k (int): Number of facilities.

    Returns:
      float: Maximum regret for this instance.
    """
    n = len(peaks)
    if misreports is None:
        return 0.0

    # Ensure expected shape (n, num_misreports)
    if not (len(misreports.shape) == 2 and misreports.shape[0] == n):
         warnings.warn(f"Unexpected misreports shape {misreports.shape} in calc_max_regret. Expected ({n}, M). Skipping regret.", stacklevel=2)
         return 0.0

    # Ensure float types
    peaks = peaks.astype(float, copy=False)
    weights = weights.astype(float, copy=False)
    misreports = misreports.astype(float, copy=False)
    normalized_weights = weights / np.sum(weights) # Use normalized weights for cost comparison

    # Calculate cost with original peaks
    fac_original = place_func(peaks, weights, k)
    if not isinstance(fac_original, (list, np.ndarray)) or len(fac_original) == 0:
        warnings.warn("Original placement function returned invalid facilities. Regret=0.", stacklevel=2)
        return 0.0
    fac_original = np.array(fac_original, dtype=float)
    original_costs_per_agent = np.min(np.abs(peaks[:, np.newaxis] - fac_original), axis=1) # Cost for each agent

    max_r = 0.0

    # Iterate through each agent misreporting
    for i in range(n):
        cost_i_orig = original_costs_per_agent[i]
        agent_true_peak = peaks[i]

        # Iterate through each misreport for agent i
        for rep_idx in range(misreports.shape[1]): # Iterate through M misreports
            new_peaks = np.copy(peaks)
            new_peaks[i] = misreports[i, rep_idx] # Agent i misreports

            # Get new facility locations based on the misreport
            fac_new = place_func(new_peaks, weights, k)
            if not isinstance(fac_new, (list, np.ndarray)) or len(fac_new) == 0:
                # Mechanism failed with this misreport, assume no gain
                continue
            fac_new = np.array(fac_new, dtype=float)

            # Calculate agent i's cost with the *new* facilities but using their *true* peak
            cost_i_new = np.min(np.abs(agent_true_peak - fac_new))

            # Regret = gain = original_cost - new_cost
            # Note: We compare unweighted costs for the agent's gain
            regret = cost_i_orig - cost_i_new
            if regret > max_r:
                max_r = regret

    return max_r


def calc_fitness(peaks, misreports, weights, place_func, k, epsilon=0.01):
    """
    Computes the fitness = (weighted social cost) + penalty, where
      penalty = 1 if max_regret > epsilon, else 0.
    """
    # Ensure float types
    peaks = peaks.astype(float, copy=False)
    weights = weights.astype(float, copy=False)

    # Place facilities using the provided function
    facilities = place_func(peaks, weights, k)
    if not isinstance(facilities, (list, np.ndarray)) or len(facilities) == 0:
         warnings.warn("Placement function returned invalid facilities. Returning high fitness.", stacklevel=2)
         return 9999.0 # Return high fitness if placement fails
    facilities = np.array(facilities, dtype=float)

    # Calculate Weighted social cost
    cost = calc_weighted_social_cost(peaks, facilities, weights)

    # Calculate Max regret
    # Pass the original misreports structure for this instance
    max_r = calc_max_regret(peaks, misreports, weights, place_func, k)
    penalty = 1.0 if max_r > epsilon else 0.0

    # Check for NaN or Inf in cost (can happen with extreme inputs/outputs)
    if np.isnan(cost) or np.isinf(cost):
        warnings.warn(f"NaN or Inf cost detected for peaks: {peaks}, facilities: {facilities}. Returning high fitness.", stacklevel=2)
        return 9999.0

    return cost + penalty


# --- Baseline Mechanism Implementations ---

def place_facilities_quantiles(agent_peaks, weights, k):
    """Baseline: Place k facilities at evenly spaced quantiles (ignoring weights)."""
    n_agents = len(agent_peaks)
    if n_agents == 0: return np.array([], dtype=float)
    sorted_peaks = np.sort(agent_peaks.astype(float))

    if k == 0: return np.array([], dtype=float)
    if k >= n_agents: # Place facility at each unique peak up to k
         unique_sorted = np.unique(sorted_peaks)
         return unique_sorted[:k]

    # Calculate positions based on quantiles
    indices = np.linspace(0, 1, k + 2)[1:-1] # e.g., [1/(k+1), 2/(k+1), ..., k/(k+1)]
    locations = np.quantile(sorted_peaks, indices, method='linear' if hasattr(np, 'quantile') and 'method' in np.quantile.__code__.co_varnames else 'linear') # 'linear' interpolation is common
    return np.array(locations, dtype=float)


def dictatorial_rule(agent_peaks, weights, k, dictator_index):
    """Baseline: Places all K facilities at the dictator's peak."""
    n_agents = len(agent_peaks)
    if n_agents == 0: return np.array([0.5] * k, dtype=float) # Default if no agents
    if dictator_index < 0 or dictator_index >= n_agents:
        warnings.warn(f"Invalid dictator index {dictator_index}. Using median.", stacklevel=2)
        return np.array([np.median(agent_peaks)] * k, dtype=float) # Fallback
    return np.array([agent_peaks[dictator_index]] * k, dtype=float)


def best_dictatorial_rule(agent_peaks, weights, k):
    """Baseline: Finds the best dictator for a specific instance."""
    n_agents = len(agent_peaks)
    if n_agents == 0: return np.array([0.5] * k, dtype=float) # Default if no agents

    best_cost = np.inf
    best_locations = []
    normalized_weights = weights / np.sum(weights) # Use normalized for cost comparison

    for i in range(n_agents):
        locations = dictatorial_rule(agent_peaks, weights, k, i)
        cost = calc_weighted_social_cost(agent_peaks, locations, normalized_weights)
        if cost < best_cost:
            best_cost = cost
            best_locations = locations

    # Handle case where no valid dictator found (shouldn't happen if peaks are valid)
    if not isinstance(best_locations, (list, np.ndarray)) or len(best_locations) == 0:
         median_loc = np.median(agent_peaks)
         best_locations = np.array([median_loc] * k, dtype=float)

    return np.array(best_locations, dtype=float)


def constant_rule(agent_peaks, weights, k):
    """Baseline: Uses fixed, evenly spaced locations."""
    if k == 0: return np.array([], dtype=float)
    # Simple fixed locations: 1/(K+1), 2/(K+1), ..., K/(K+1)
    locations = [ (i + 1.0) / (k + 1.0) for i in range(k) ]
    return np.array(locations, dtype=float)


# --- Data Loading Class ---

class GetData:
    """
    Loads train/test data from .pkl files specified by paths.
    Handles nested dictionary structure {'peaks': ndarray, 'misreports': ndarray, ...}
    Attempts to reshape misreports if they appear flattened.
    Provides baseline evaluation methods.
    """

    def __init__(
        self,
        data_train_path=TRAIN_DATA_FILE,
        data_test_path=TEST_DATA_FILE,
        num_misreports_per_item=NUM_MISREPORTS_PER_AGENT_OR_SCENARIO # Pass the constant
    ):
        self.train_path = data_train_path
        self.test_path = data_test_path
        self.num_misreports = num_misreports_per_item
        self.train_data = {}
        self.test_data = {}
        self.load_data()

    def load_data(self):
        """Loads and preprocesses data."""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train data not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found: {self.test_path}")

        print(f"Loading train data from: {self.train_path}")
        with open(self.train_path, 'rb') as f:
            self.train_data = pickle.load(f)
        print("Train data loaded.")

        print(f"Loading test data from: {self.test_path}")
        with open(self.test_path, 'rb') as f:
            self.test_data = pickle.load(f)
        print("Test data loaded.")

        # Preprocess (convert types, reshape misreports)
        print("Preprocessing training data...")
        self._preprocess_dataset(self.train_data)
        print("Preprocessing test data...")
        self._preprocess_dataset(self.test_data)
        print("Data preprocessing complete.")


    def _preprocess_dataset(self, dataset):
        """Convert types and attempt to reshape misreports for a dataset dict.
        We no longer read weights from the pickle; 而是根据 n_agents 直接设置。"""
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
                    val[PEAK_DATA_KEY] = np.array(val[PEAK_DATA_KEY], dtype=float)
                    if val[PEAK_DATA_KEY].ndim != 2:
                        warnings.warn(f"Peaks for {key} are not 2D shape. Skipping key.")
                        valid_instance = False
                except Exception as e:
                    warnings.warn(f"Could not convert peaks for {key} to float array: {e}. Skipping key.")
                    valid_instance = False
            else:
                warnings.warn(f"'{PEAK_DATA_KEY}' not found or is None for key {key}. Skipping key.")
                valid_instance = False

            if not valid_instance:
                keys_to_remove.append(key)
                continue

            num_samples, n_agents = val[PEAK_DATA_KEY].shape

            # 2. 直接按论文设置 agent weights
            if n_agents == 5:
                # n=5 时，第一个 agent 权重 5，其余 1
                val[WEIGHTS_DATA_KEY] = np.array([5] + [1] * (n_agents - 1), dtype=float)
            elif n_agents == 10:
                # n=10 时，前两个 agent 权重 5，其余 1
                val[WEIGHTS_DATA_KEY] = np.array([5, 5] + [1] * (n_agents - 2), dtype=float)
            else:
                # 其他情况都用均匀权重 1
                val[WEIGHTS_DATA_KEY] = np.ones(n_agents, dtype=float)

            # 3. Convert and reshape misreports (optional)
            if MISREPORT_DATA_KEY in val and val[MISREPORT_DATA_KEY] is not None:
                try:
                    misreports_raw = np.array(val[MISREPORT_DATA_KEY], dtype=float)
                    expected_flat = num_samples * self.num_misreports
                    target_shape = (num_samples, n_agents, self.num_misreports)

                    if misreports_raw.shape == target_shape:
                        val[MISREPORT_DATA_KEY] = misreports_raw
                    elif misreports_raw.shape == (expected_flat, n_agents):
                        intermediate = misreports_raw.reshape((num_samples, self.num_misreports, n_agents))
                        val[MISREPORT_DATA_KEY] = intermediate.transpose(0, 2, 1)
                    else:
                        warnings.warn(f"Misreports for {key} have unexpected shape {misreports_raw.shape}. Setting to None.")
                        val[MISREPORT_DATA_KEY] = None

                except Exception as e:
                    warnings.warn(f"Could not process misreports for {key}: {e}. Setting to None.")
                    val[MISREPORT_DATA_KEY] = None
            else:
                val[MISREPORT_DATA_KEY] = None

        # 4. 删除所有无效的实例
        for bad_key in keys_to_remove:
            del dataset[bad_key]




    def get_instances(self):
        """
        Return (train_data, test_data).
        Each is a dict mapping (dist, n_agents) -> {'peaks': ..., 'weights': ..., 'misreports': ...}.
        """
        return self.train_data, self.test_data











