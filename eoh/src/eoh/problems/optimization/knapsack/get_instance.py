from __future__ import annotations  # Enables postponed evaluation of type annotations
import numpy as np
import os


def dp_knapsack_01(weights: list[float], values: list[float], capacity: float) -> float:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    via dynamic programming, using a 1D DP array.

    Adapted to work with float weights and values from the existing dataset.

    Let:
      - n = number of items
      - W = capacity of the knapsack
      - w_i = weight of item i
      - v_i = value of item i

    We define the DP array as:
        dp[c] = maximum possible value achievable using capacity c

    Transition Formula (iterating in reverse order for capacity):
        dp[c] = max( dp[c], dp[c - w_i] + v_i ),  if c >= w_i

    The time complexity is O(n * W).

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
    A class for loading existing Knapsack datasets and computing 
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
        count: int = 64,
        mode: str = "train",
        dataset_path: str = "dataset",
    ) -> tuple[dict[str, dict[str, float | list[float]]], dict[str, float]]:
        """
        Load multiple 0-1 Knapsack instances from the existing dataset files,
        and compute the **optimal** solution values for each instance.

        Each instance includes:
            - capacity   (float)      : the knapsack capacity
            - num_items  (int)        : number of items in the instance
            - weights    (list[float]): list of item weights
            - values     (list[float]): list of item values

        :param size:
            A string representing the number of items (e.g., '50', '100', '200', '500').
        :param capacity:
            (Optional) If provided, this value will be used as the fixed knapsack capacity
            for all instances. If None, capacity is set based on problem size:
            - 12.5 for size 50
            - 25.0 for other sizes
        :param count:
            Number of instances to load. Default is 64.
        :param mode:
            Dataset mode: 'train', 'val', or 'test'. Default is 'train'.
        :param dataset_path:
            Path to the dataset directory. Default is 'dataset'.

        :return:
            (instances, baseline) as a 2-tuple:
              - instances (dict[str, dict[str, float | list[float]]]):
                  A dictionary of loaded instances.
                  Each key is an instance name, each value is a dictionary containing:
                    {
                      "capacity": float,
                      "num_items": int,
                      "weights": list[float],
                      "values": list[float]
                    }
              - baseline (dict[str, float]):
                  A dictionary mapping each instance name to the optimal solution value.
        """
        file_name = f"{mode}{size}_dataset.npy"
        file_path = os.path.join(dataset_path, file_name)
        
        # Load the dataset from .npy file
        try:
            data = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Set default capacity based on problem size
        if capacity is None:
            if size == "50":
                capacity = 12.5
            else:
                capacity = 25.0
        
        size_int: int = int(size)
        instances: dict[str, dict[str, float | list[float]]] = {}
        baseline: dict[str, float] = {}
        
        # Determine how many instances to load (min of available instances and requested count)
        n_instances = min(data.shape[0], count)
        
        for i in range(n_instances):
            instance_name: str = f"{mode}_{size}_{i+1}"
            
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
    
    # Load training instances with 100 items
    train_instances, train_optimal = data_loader.get_instances(
        size="100", 
        mode="train",
        count=5  # Just load 5 for quick testing
    )
    
    # Print details of the first instance
    first_instance_name = next(iter(train_instances))
    first_instance = train_instances[first_instance_name]
    
    print(f"Instance: {first_instance_name}")
    print(f"  Capacity: {first_instance['capacity']}")
    print(f"  Number of items: {first_instance['num_items']}")
    print(f"  First 5 weights: {first_instance['weights'][:5]}")
    print(f"  First 5 values: {first_instance['values'][:5]}")
    print(f"  Optimal value: {train_optimal[first_instance_name]}")