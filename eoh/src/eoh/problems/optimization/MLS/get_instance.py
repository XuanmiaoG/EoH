import numpy as np


class GetData:
    def __init__(
        self,
        distributions=("uniform", "normal", "beta1", "beta2"),
        n_values=(5, 10, 25),
        k_values=(1, 2, 3, 4),
        samples_per_setting=10,
        seed=0,
    ):
        """
        Initializes a dataset for a multi-facility location problem
        with single-peaked agents.

        Args:
            distributions: Tuple/list of distributions to sample agent peaks from.
                Allowed values (example): "uniform", "normal" (with mean=0.5, std=1),
                or "beta1"/"beta2" (with alpha/beta = 1 or 9, etc.).
            n_values: Tuple/list of number of agents per instance (e.g., 5, 10).
            k_values: Tuple/list of number of facilities to place.
            samples_per_setting: How many random problem instances to generate per setting.
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.datasets = {}
        self.generate_random_datasets(
            distributions, n_values, k_values, samples_per_setting
        )

    def generate_random_datasets(
        self, distributions, n_values, k_values, samples_per_setting
    ):
        """
        Generates random problem instances for multi-facility location,
        storing them under self.datasets in a nested dictionary structure.

        For each combination of (distribution, n, k), we create some number of
        randomly generated samples, each stored as an entry "test_0", "test_1", etc.
        """
        for dist_name in distributions:
            for n in n_values:
                for k in k_values:
                    # Skip cases where k ≥ n (not interesting for facility location)
                    if k >= n:
                        continue
                    
                    # Construct a key in the same style as "Weibull 5k",
                    # but now for facility location:
                    dataset_key = f"{dist_name}_n{n}_k{k}"
                    self.datasets[dataset_key] = {}

                    for sample_id in range(samples_per_setting):
                        instance_name = f"test_{sample_id}"
                        # Generate peaks according to dist_name
                        peaks = self.generate_peaks(dist_name=dist_name, n=n)

                        # Assign weights based on n
                        if n == 5:
                            weights = np.array([5, 1, 1, 1, 1], dtype=float)
                        elif n == 10:
                            weights = np.array([5, 5, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
                        else:
                            # For n=25 or other values, create a simple pattern
                            # 20% of agents have weight 5, rest have weight 1
                            high_weight_count = max(1, int(0.2 * n))
                            weights = np.ones(n, dtype=float)
                            weights[:high_weight_count] = 5.0
                            
                        # Generate misreports for strategyproofness testing
                        misreports = self.generate_misreports(peaks)

                        instance_data = {
                            "n": n,
                            "k": k,
                            "peaks": peaks,
                            "weights": weights,
                            "misreports": misreports,
                        }
                        self.datasets[dataset_key][instance_name] = instance_data

    def generate_peaks(self, dist_name, n):
        """
        Generate n agent peaks in [0,1], according to the requested distribution.
        """
        if dist_name == "uniform":
            return self.rng.random(n)  # uniform [0,1]

        elif dist_name == "normal":
            # Normal with mean 0.5, std=1; then clipped to [0,1]
            raw = self.rng.normal(loc=0.5, scale=0.2, size=n)
            return np.clip(raw, 0.0, 1.0)

        elif dist_name == "beta1":
            # alpha=1, beta=9 (skewed to the left)
            raw = self.rng.beta(a=1.0, b=9.0, size=n)
            return raw

        elif dist_name == "beta2":
            # alpha=9, beta=1 (skewed to the right)
            raw = self.rng.beta(a=9.0, b=1.0, size=n)
            return raw

        else:
            # Default fallback: uniform
            return self.rng.random(n)

    def generate_misreports(self, peaks):
        """
        Generate reasonable misreports for each agent to test strategyproofness.
        """
        n = len(peaks)
        misreports_per_agent = 10
        misreports = np.zeros((n, misreports_per_agent))
        
        for i in range(n):
            for j in range(misreports_per_agent):
                strategy = j % 5  # 5 different misreporting strategies
                
                if strategy == 0:
                    # Random misreport in [0,1]
                    misreports[i, j] = self.rng.random()
                elif strategy == 1:
                    # Small deviation from true peak
                    noise = self.rng.normal(0, 0.05)  # Small Gaussian noise
                    misreports[i, j] = np.clip(peaks[i] + noise, 0.0, 1.0)
                elif strategy == 2:
                    # Move away from true peak (to the right)
                    shift = self.rng.uniform(0.1, 0.3)
                    misreports[i, j] = np.clip(peaks[i] + shift, 0.0, 1.0)
                elif strategy == 3:
                    # Move away from true peak (to the left)
                    shift = self.rng.uniform(0.1, 0.3)
                    misreports[i, j] = np.clip(peaks[i] - shift, 0.0, 1.0)
                else:
                    # Extreme report (0 or 1)
                    misreports[i, j] = 0.0 if self.rng.random() < 0.5 else 1.0
        
        return misreports

    def calculate_lower_bound(self, peaks, weights, k):
        """
        Calculate a simple lower bound for the weighted social cost.
        This is based on the best possible facility placement for an equal distribution.
        """
        # Sort peaks for easier calculation
        sorted_peaks = np.sort(peaks)
        n = len(peaks)
        
        # In the best case, facilities are optimally placed to minimize cost
        # For k=1, the weighted median is optimal
        if k == 1:
            # Approximate weighted median
            cumulative_weights = np.cumsum(weights) / np.sum(weights)
            median_idx = np.argmin(np.abs(cumulative_weights - 0.5))
            return np.sum(weights * np.abs(peaks - sorted_peaks[median_idx])) / np.sum(weights)
        
        # For k>1, a simple lower bound is to divide agents into k equal groups
        # and place a facility at the center of each group
        else:
            # Divide sorted peaks into k segments
            segment_size = n // k
            
            # Calculate cost assuming perfect division of agents
            total_cost = 0
            for i in range(k):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < k-1 else n
                
                if start_idx < end_idx:
                    # Place facility at the mean of this segment
                    segment_peaks = sorted_peaks[start_idx:end_idx]
                    facility_loc = np.mean(segment_peaks)
                    
                    # Calculate cost for this segment
                    segment_costs = np.abs(segment_peaks - facility_loc)
                    segment_weights = weights[start_idx:end_idx] if len(weights) == n else np.ones(len(segment_peaks))
                    
                    total_cost += np.sum(segment_weights * segment_costs)
            
            return total_cost / np.sum(weights)

    def dataset_lower_bound(self, instances):
        """
        Computes the mean of the lower bound over all instances in a dataset.
        """
        bounds = []
        for instance_name in instances:
            data = instances[instance_name]
            peaks = data["peaks"]
            weights = data["weights"]
            k = data["k"]
            bounds.append(self.calculate_lower_bound(peaks, weights, k))
        return np.mean(bounds)

    def get_instances(self):
        """
        Returns (all_datasets, summary_bounds) where summary_bounds provides
        a lower bound estimate for each dataset configuration.
        """
        summary_bounds = {}
        for dataset_key, instances in self.datasets.items():
            summary_bounds[dataset_key] = self.dataset_lower_bound(instances)

        return self.datasets, summary_bounds