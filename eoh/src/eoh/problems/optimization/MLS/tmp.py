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
def _evaluate_single_baseline(self, place_func, data_dict, k, epsilon):
        """Helper to evaluate one baseline function over a data dictionary."""
        total_fitness = 0.0
        total_cost = 0.0
        total_regret = 0.0
        count = 0
        skipped_instances = 0

        for key, val in data_dict.items():
            # Use preprocessed data
            peaks_arr = val.get(PEAK_DATA_KEY)
            misr_arr = val.get(MISREPORT_DATA_KEY) # Should be shape (samples, agents, misreports_per_agent) or None
            weights_info = val.get(WEIGHTS_DATA_KEY)

            if peaks_arr is None: continue # Skip if peaks are missing

            num_samples, n_agents = peaks_arr.shape
            k_local = val.get('k', k) # Allow overriding k per instance if needed

            for i in range(num_samples):
                peaks_i = peaks_arr[i]

                # Determine weights for this instance
                if weights_info is not None:
                    if len(weights_info.shape) == 1 and weights_info.shape[0] == n_agents:
                        weights_i = weights_info
                    elif len(weights_info.shape) == 2 and weights_info.shape[0] == num_samples and weights_info.shape[1] == n_agents:
                        weights_i = weights_info[i]
                    else: # Fallback to uniform if shape is wrong
                        weights_i = np.ones(n_agents, dtype=float)
                else: # Default to uniform
                    weights_i = np.ones(n_agents, dtype=float)

                # Get misreports for this instance (should be shape (n_agents, M) or None)
                misreports_i = None
                if misr_arr is not None:
                    # Check if misr_arr has the expected 3D shape after preprocessing
                    if len(misr_arr.shape) == 3 and misr_arr.shape[0] == num_samples and misr_arr.shape[1] == n_agents:
                         misreports_i = misr_arr[i] # Shape (n_agents, M)
                    # Add handling here if another valid shape is possible after preprocessing
                    # else: # Already warned during preprocessing if shape was wrong

                # Calculate fitness using the baseline function
                try:
                    # Need to pass correctly shaped misreports_i to calc_fitness
                    fitness_i = calc_fitness(
                        peaks_i, misreports_i, weights_i,
                        place_func, # Use the passed baseline function
                        k_local, epsilon=epsilon
                    )
                    # Also calculate cost and regret separately for reporting
                    facilities_i = place_func(peaks_i, weights_i, k_local)
                    cost_i = calc_weighted_social_cost(peaks_i, facilities_i, weights_i)
                    regret_i = calc_max_regret(peaks_i, misreports_i, weights_i, place_func, k_local)

                    if np.isnan(fitness_i) or np.isinf(fitness_i):
                         skipped_instances += 1
                         continue # Skip if fitness calculation failed

                    total_fitness += fitness_i
                    total_cost += cost_i
                    total_regret += regret_i
                    count += 1

                except Exception as e:
                     warnings.warn(f"Error evaluating baseline {place_func.__name__} on instance {i} of key {key}: {e}", stacklevel=2)
                     skipped_instances += 1


        if skipped_instances > 0:
             print(f"Warning: Skipped {skipped_instances} instances during baseline evaluation for {place_func.__name__} due to errors.")

        avg_fitness = total_fitness / count if count > 0 else np.inf
        avg_cost = total_cost / count if count > 0 else np.inf
        avg_regret = total_regret / count if count > 0 else np.inf

        return avg_fitness, avg_cost, avg_regret


    def evaluate_all_baselines(self, k=None, epsilon=0.01):
        """Evaluates all defined baselines on the loaded test data."""
        results = {}
        if k is None:
            k = 2 # Default K if not specified
        print(f"\n--- Evaluating Baselines (k={k}, epsilon={epsilon}) ---")

        # Evaluate Quantile Baseline
        fitness, cost, regret = self._evaluate_single_baseline(
            place_facilities_quantiles, self.test_data, k, epsilon
        )
        results['Quantile'] = {'fitness': fitness, 'cost': cost, 'regret': regret}
        print(f"Quantile Baseline Avg Fitness: {fitness:.6f} (Cost: {cost:.6f}, Regret: {regret:.6f})")

        # Evaluate Best Dictator Baseline
        fitness, cost, regret = self._evaluate_single_baseline(
            best_dictatorial_rule, self.test_data, k, epsilon
        )
        results['BestDictator'] = {'fitness': fitness, 'cost': cost, 'regret': regret}
        print(f"Best Dictator Baseline Avg Fitness: {fitness:.6f} (Cost: {cost:.6f}, Regret: {regret:.6f})")

        # Evaluate Constant Baseline
        fitness, cost, regret = self._evaluate_single_baseline(
            constant_rule, self.test_data, k, epsilon
        )
        results['Constant'] = {'fitness': fitness, 'cost': cost, 'regret': regret}
        print(f"Constant Baseline Avg Fitness: {fitness:.6f} (Cost: {cost:.6f}, Regret: {regret:.6f})")

        print("--- Baseline Evaluation Complete ---")
        return results
# --- Baseline Mechanism Implementations ---


def place_facilities_quantiles(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """Baseline: Place k facilities at evenly spaced quantiles (ignoring weights)."""
    n_agents: int = len(agent_peaks)
    if n_agents == 0:
        return np.array([], dtype=float)
    sorted_peaks = np.sort(agent_peaks.astype(float))

    if k == 0:
        return np.array([], dtype=float)
    if k >= n_agents:  # Place facility at each unique peak up to k
        unique_sorted = np.unique(sorted_peaks)
        return unique_sorted[:k]

    # Calculate positions based on quantiles
    indices = np.linspace(0, 1, k + 2)[1:-1]  # e.g. [1/(k+1), ..., k/(k+1)]
    # If np.quantile in your version doesn't have the 'method' argument, just remove it.
    locations = np.quantile(sorted_peaks, indices, method="linear")
    return np.array(locations, dtype=float)


def dictatorial_rule(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int, dictator_index: int
) -> np.ndarray:
    """Baseline: Places all K facilities at the dictator's peak."""
    n_agents: int = len(agent_peaks)
    if n_agents == 0:
        return np.array([0.5] * k, dtype=float)  # Default if no agents
    if dictator_index < 0 or dictator_index >= n_agents:
        warnings.warn(
            f"Invalid dictator index {dictator_index}. Using median.", stacklevel=2
        )
        return np.array([np.median(agent_peaks)] * k, dtype=float)
    return np.array([agent_peaks[dictator_index]] * k, dtype=float)


def best_dictatorial_rule(
    agent_peaks: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """Baseline: Finds the best dictator for a specific instance."""
    n_agents: int = len(agent_peaks)
    if n_agents == 0:
        return np.array([0.5] * k, dtype=float)  # Default if no agents

    best_cost: float = float("inf")
    best_locations: np.ndarray | list[float] = []
    # We'll do cost comparisons with normalized weights
    normalized_weights = weights / np.sum(weights)

    for i in range(n_agents):
        locations = dictatorial_rule(agent_peaks, weights, k, i)
        cost = calc_weighted_social_cost(agent_peaks, locations, normalized_weights)
        if cost < best_cost:
            best_cost = cost
            best_locations = locations

    if not isinstance(best_locations, (list, np.ndarray)) or len(best_locations) == 0:
        median_loc = float(np.median(agent_peaks))
        best_locations = np.array([median_loc] * k, dtype=float)

    return np.array(best_locations, dtype=float)


def constant_rule(agent_peaks: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Baseline: Uses fixed, evenly spaced locations."""
    if k == 0:
        return np.array([], dtype=float)
    # Simple fixed locations: 1/(K+1), 2/(K+1), ..., K/(K+1)
    locations = [(i + 1.0) / (k + 1.0) for i in range(k)]
    return np.array(locations, dtype=float)