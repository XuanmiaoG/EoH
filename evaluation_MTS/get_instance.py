import numpy as np


class GetData:
    def __init__(
        self,
        distributions=("uniform", "normal"),
        n_values=(5, 10),
        k_values=(1, 2),
        samples_per_setting=5,
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
                    # Construct a key in the same style as "Weibull 5k",
                    # but now for facility location:
                    dataset_key = f"{dist_name}_n{n}_k{k}"
                    self.datasets[dataset_key] = {}

                    for sample_id in range(samples_per_setting):
                        instance_name = f"test_{sample_id}"
                        # Generate peaks according to dist_name
                        peaks = self.generate_peaks(dist_name=dist_name, n=n)

                        # Example: Assign uniform weights or random weights
                        # (You can adapt as needed)
                        weights = np.ones(n, dtype=float)  # e.g. unweighted
                        # or random in {1,...,5}
                        # weights = self.rng.integers(1, 6, size=n)

                        instance_data = {
                            "n": n,
                            "k": k,
                            "peaks": peaks,
                            "weights": weights,
                            # Optionally store other info (e.g. number misreports, etc.)
                            "misreports_per_agent": 10,  # example placeholder
                        }
                        self.datasets[dataset_key][instance_name] = instance_data

    def generate_peaks(self, dist_name, n):
        """
        Generate n agent peaks in [0,1], according to the requested distribution.
        Adjust or extend as needed for beta, etc.
        """
        if dist_name == "uniform":
            return self.rng.random(n)  # uniform [0,1]

        elif dist_name == "normal":
            # Example: normal with mean 0.5, std=1; then clipped to [0,1]
            raw = self.rng.normal(loc=0.5, scale=1.0, size=n)
            return np.clip(raw, 0.0, 1.0)

        elif dist_name == "beta1":
            # alpha=1, beta=9
            raw = self.rng.beta(a=1.0, b=9.0, size=n)
            return raw

        elif dist_name == "beta2":
            # alpha=9, beta=1
            raw = self.rng.beta(a=9.0, b=1.0, size=n)
            return raw

        else:
            # Default fallback: uniform
            return self.rng.random(n)

    def dummy_lower_bound(self, peaks, weights, k):
        """
        Example placeholder: compute a trivial 'lower bound' for the cost or facility usage.
        E.g., sum of all weights / k. Adjust to something meaningful if you like.
        """
        total_weight = np.sum(weights)
        return total_weight / max(k, 1)

    def dataset_lower_bound(self, instances):
        """
        Computes the mean of the dummy lower bound over all instances in a dataset.
        You can adapt this to something more appropriate for facility location if you like.
        """
        bounds = []
        for instance_name in instances:
            data = instances[instance_name]
            peaks = data["peaks"]
            weights = data["weights"]
            k = data["k"]
            bounds.append(self.dummy_lower_bound(peaks, weights, k))
        return np.mean(bounds)

    def get_instances(self):
        """
        Mimics the same final call as in your original bin-packing code:
        returns (all_datasets, some_summary) where some_summary can be
        your “lower bound” measure or anything else.

        In the original code, opt_num_bins was the average L1 bound
        for each dataset. We replicate that style here.
        """
        summary_bounds = {}
        for dataset_key, instances in self.datasets.items():
            # e.g. “uniform_n5_k2”
            summary_bounds[dataset_key] = self.dataset_lower_bound(instances)

        return self.datasets, summary_bounds


# Example usage:
if __name__ == "__main__":
    gd = GetData(
        distributions=["uniform", "normal", "beta1", "beta2"],
        n_values=[5, 10],
        k_values=[1, 2],
        samples_per_setting=3,
        seed=42,
    )
    all_data, summary = gd.get_instances()
    print("Dataset keys:", list(all_data.keys()))
    print("Summary lower-bounds:", summary)
