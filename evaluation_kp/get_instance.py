from __future__ import annotations
import numpy as np
import os


def dp_knapsack_01(weights: list[float], values: list[float], capacity: float) -> float:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    using a Branch and Bound approach (exact) for floating-point weights/capacities.

    This function signature remains the same, but internally it does NOT use
    classic DP. Instead, it uses a Branch and Bound strategy.

    Parameters
    ----------
    weights : list[float]
        A list of item weights (floating-point).
    values : list[float]
        A list of item values (floating-point).
    capacity : float
        The total knapsack capacity (floating-point).

    Returns
    -------
    float
        The maximum total value achievable under 0-1 constraints.
    """

    n: int = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0

    # Sort items in descending order of value/weight ratio for better bounding
    items_sorted: list[tuple[float, int]] = [
        (values[i] / weights[i], i) for i in range(n)
    ]
    items_sorted.sort(key=lambda x: x[0], reverse=True)

    best_value: float = 0.0

    def bound(idx: int, current_val: float, remaining_cap: float) -> float:
        """
        Estimate the upper bound (potential maximum value) from items_sorted[idx:]
        using a fractional knapsack assumption. This helps prune the search space.
        """
        total_val: float = current_val
        for j in range(idx, n):
            ratio_j, item_idx = items_sorted[j]
            w_j: float = weights[item_idx]
            v_j: float = values[item_idx]
            if w_j <= remaining_cap:
                remaining_cap -= w_j
                total_val += v_j
            else:
                # Take only the fraction we can fit
                total_val += ratio_j * remaining_cap
                break
        return total_val

    def backtrack(
        idx: int, current_val: float, remaining_cap: float, used_indices: set[int]
    ) -> None:
        nonlocal best_value

        if current_val > best_value:
            best_value = current_val

        # Stop if we've exhausted items or capacity
        if idx >= n or remaining_cap <= 1e-9:
            return

        # Upper bound check
        est: float = bound(idx, current_val, remaining_cap)
        if est <= best_value:
            return

        ratio_i, real_idx = items_sorted[idx]
        w_i: float = weights[real_idx]
        v_i: float = values[real_idx]

        # Branch 1: do not take this item
        backtrack(idx + 1, current_val, remaining_cap, used_indices)

        # Branch 2: take this item if it fits
        if w_i <= remaining_cap:
            used_indices.add(real_idx)
            backtrack(idx + 1, current_val + v_i, remaining_cap - w_i, used_indices)
            used_indices.remove(real_idx)

    backtrack(0, 0.0, capacity, set())
    return best_value


class GetData:
    """
    A class for loading 0-1 Knapsack datasets and computing
    their optimal solutions using dp_knapsack_01.
    """

    def __init__(self) -> None:
        """
        Constructor that initializes an empty dictionary for datasets.
        """
        self.datasets: dict[str, dict[str, float | list[float]]] = {}

    def get_instances(
        self,
        size: str = "100",
        capacity: float | None = None,
        mode: str = "test",
        dataset_path: str = "evaluation_kp/dataset",
    ) -> tuple[dict[str, dict[str, float | list[float]]], dict[str, float]]:
        """
        Load multiple 0-1 Knapsack instances from a .npy file and compute
        the optimal solution for each instance using dp_knapsack_01.

        Each instance includes:
            - "capacity"   (float)      : the knapsack capacity
            - "num_items"  (int)        : the number of items in the instance
            - "weights"    (list[float]): a list of item weights
            - "values"     (list[float]): a list of item values

        Parameters
        ----------
        size : str
            A string indicating the number of items (e.g., "50", "100", "200", "500").
        capacity : float, optional
            If provided, this float is used as the knapsack capacity for all instances.
            If None, a default capacity is chosen based on 'size' (12.5 for size '50', else 25.0).
        mode : str
            One of 'train', 'val', or 'test'. This determines which .npy file to load,
            e.g., f"{mode}{size}_dataset.npy".
        dataset_path : str
            Path to the dataset directory. Default is "evaluation_kp/dataset".

        Returns
        -------
        tuple
            A 2-tuple containing:
              - instances : dict[str, dict[str, float | list[float]]]
                  {
                    "instance_name": {
                      "capacity": float,
                      "num_items": int,
                      "weights": list[float],
                      "values": list[float]
                    }
                  }
              - baseline  : dict[str, float]
                  {
                    "instance_name": optimal_value (float)
                  }
        """
        file_name: str = f"{mode}{size}_dataset.npy"
        file_path: str = os.path.join(dataset_path, file_name)

        # Load the .npy file
        try:
            data: np.ndarray = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Set default capacity if not provided
        if capacity is None:
            if size == "50":
                capacity = 12.5
            else:
                capacity = 25.0

        size_int: int = int(size)
        instances: dict[str, dict[str, float | list[float]]] = {}
        baseline: dict[str, float] = {}

        # data.shape should be (num_instances, size_int, 2)
        num_instances_in_file: int = data.shape[0]
        n_instances: int = num_instances_in_file

        for i in range(n_instances):
            instance_name: str = f"{mode}_{size}_{i+1}"

            # data[i] has shape (size_int, 2):
            #   column 0 => weights
            #   column 1 => values
            weights_arr: list[float] = data[i, :, 0].tolist()
            values_arr: list[float] = data[i, :, 1].tolist()

            instance_data: dict[str, float | list[float]] = {
                "capacity": capacity,
                "num_items": size_int,
                "weights": weights_arr,
                "values": values_arr,
            }
            instances[instance_name] = instance_data

            # Compute the optimal solution via Branch and Bound
            try:
                opt_val: float = dp_knapsack_01(weights_arr, values_arr, capacity)
                baseline[instance_name] = opt_val
            except Exception as exc:
                print(f"[ERROR] {instance_name}: {exc}")
                baseline[instance_name] = -1.0

        return instances, baseline


if __name__ == "__main__":
    # Example usage: load "train100_dataset.npy" from "evaluation_kp/dataset" and compute solutions
    data_loader = GetData()
    instances_dict, baseline_dict = data_loader.get_instances(
        size="100",
        capacity=None,  # default capacity = 25.0 for size != "50"
        mode="train",
        dataset_path="evaluation_kp/dataset",
    )

    # Print info about the first loaded instance
    if instances_dict:
        first_instance_name = next(iter(instances_dict))
        inst_data = instances_dict[first_instance_name]
        print(f"\nInstance: {first_instance_name}")
        print(f"  capacity: {inst_data['capacity']}")
        print(f"  num_items: {inst_data['num_items']}")
        print(f"  first 5 weights: {inst_data['weights'][:5]}")
        print(f"  first 5 values: {inst_data['values'][:5]}")
        print(f"  optimal_value: {baseline_dict[first_instance_name]}")
    else:
        print("No instances loaded.")
