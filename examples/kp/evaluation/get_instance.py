from __future__ import annotations  # Enables postponed evaluation of type annotations
import numpy as np
import os


def dp_knapsack_01(weights: list[float], values: list[float], capacity: float) -> float:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    via dynamic programming, using a 1D DP array.

    :param weights: list of item weights
    :param values: list of item values
    :param capacity: total knapsack capacity
    :return: the maximum total value achievable
    """
    n: int = len(weights)
    
    # Convert floats to integers by scaling for DP
    scale_factor = 10000
    int_weights = [int(w * scale_factor) for w in weights]
    int_capacity = int(capacity * scale_factor)
    
    dp: list[float] = [0] * (int_capacity + 1)

    for i in range(n):
        w_i: int = int_weights[i]
        v_i: float = values[i]
        # Iterate capacity in reverse to avoid using updated dp values prematurely
        for c in range(int_capacity, w_i - 1, -1):
            dp[c] = max(dp[c], dp[c - w_i] + v_i)

    return dp[int_capacity]


class GetData:
    """
    A class for loading knapsack instances from datasets and computing
    their optimal solutions using the dp_knapsack_01 function.
    """

    def __init__(self) -> None:
        """
        Constructor initializing an empty datasets dictionary.
        """
        self.datasets: dict[str, dict[str, int | list[int]]] = {}

    def get_instances(
        self,
        size: str = "100",
        capacity: float | None = None,
        count: int = 50,
    ) -> tuple[dict[str, dict[str, float | list[float]]], dict[str, float]]:
        """
        Get multiple 0-1 Knapsack instances with a given number of items,
        and compute the optimal solution values for each instance.

        Each instance includes:
            - capacity   (float)      : the knapsack capacity
            - num_items  (int)        : number of items in the instance
            - weights    (list[float]): list of item weights
            - values     (list[float]): list of item values

        :param size:
            A string representing the number of items (e.g., '50', '100', '200', '500').
        :param capacity:
            The knapsack capacity.
        :param count:
            Number of instances to load. Default is 50.

        :return:
            (instances, baseline) as a 2-tuple:
              - instances: dict of instances
              - baseline: dict mapping instance names to optimal values
        """
        # Set default capacity if not provided based on original .npy dataset characteristics
        if capacity is None:
            if size == "50":
                capacity = 12.5
            else:
                capacity = 25.0
        
        # Try to locate dataset files in the following order
        dataset_paths = [
            os.path.join("dataset", f"train{size}_dataset.npy"),
            os.path.join("dataset", f"val{size}_dataset.npy"),
            os.path.join("dataset", f"test{size}_dataset.npy")
        ]
        
        # Check if any dataset file exists
        data = None
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    break
                except Exception:
                    pass
        
        # If no dataset exists or couldn't be loaded, fall back to the original random generation
        if data is None:
            # Set a fixed random seed for reproducibility
            np.random.seed(2025)

            size_int: int = int(size)
            instances: dict[str, dict[str, float | list[float]]] = {}
            baseline: dict[str, float] = {}

            for i in range(1, count + 1):
                instance_name: str = f"test_{i}"

                # Generate random weights in [1..100] and values in [10..200]
                weights: list[float] = np.random.randint(1, 101, size_int).tolist()
                values_arr: list[float] = np.random.randint(10, 201, size_int).tolist()

                # Build the instance data
                instance_data: dict[str, float | list[float] | int] = {
                    "capacity": capacity,
                    "num_items": size_int,
                    "weights": weights,
                    "values": values_arr,
                }
                instances[instance_name] = instance_data

                # Compute the optimal solution via DP
                optimal_value: float = dp_knapsack_01(weights, values_arr, capacity)
                baseline[instance_name] = optimal_value

            return instances, baseline
        
        # Process the loaded dataset
        size_int: int = int(size)
        instances: dict[str, dict[str, float | list[float] | int]] = {}
        baseline: dict[str, float] = {}
        
        # Determine how many instances to load
        n_instances = min(data.shape[0], count)
        
        for i in range(n_instances):
            instance_name: str = f"test_{i+1}"
            
            # Extract weights and values
            weights: list[float] = data[i, :, 0].tolist()  # First column is weights
            values: list[float] = data[i, :, 1].tolist()   # Second column is values
            
            # Build the instance data
            instance_data: dict[str, float | list[float] | int] = {
                "capacity": capacity,
                "num_items": size_int,
                "weights": weights,
                "values": values,
            }
            instances[instance_name] = instance_data
            
            # Compute the optimal solution via DP
            try:
                optimal_value: float = dp_knapsack_01(weights, values, capacity)
                baseline[instance_name] = optimal_value
            except Exception as e:
                print(f"Error computing optimal for {instance_name}: {e}")
                # For very large instances, DP might be too slow or memory-intensive
                baseline[instance_name] = -1.0
        
        return instances, baseline


# Example usage
if __name__ == "__main__":
    # Create an instance of the GetData class
    data_loader = GetData()
    
    # Load instances with 100 items
    instances, optimal_values = data_loader.get_instances(
        size="100", 
        count=5  # Just load 5 for quick testing
    )
    
    # Print details of the first instance
    first_instance_name = next(iter(instances))
    first_instance = instances[first_instance_name]
    
    print(f"Instance: {first_instance_name}")
    print(f"  Capacity: {first_instance['capacity']}")
    print(f"  Number of items: {first_instance['num_items']}")
    print(f"  First 5 weights: {first_instance['weights'][:5]}")
    print(f"  First 5 values: {first_instance['values'][:5]}")
    print(f"  Optimal value: {optimal_values[first_instance_name]}")