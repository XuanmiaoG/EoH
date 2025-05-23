from __future__ import annotations
import numpy as np
import os
import bisect


def dp_knapsack_01(weights: list[float], values: list[float], capacity: float) -> float:
    """
    Compute the optimal (maximum) total value for the 0-1 Knapsack problem
    using a Branch and Bound approach (exact) for floating-point weights/capacities.
    We have improved it by:
      - avoiding divide-by-zero if any weight == 0;
      - speeding up bounding with prefix sums + binary search.

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

    items_with_ratio = []
    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        if w_i > 1e-14:
            ratio = v_i / w_i
        else:
            ratio = float("inf") if (v_i > 0) else 0.0
        items_with_ratio.append((ratio, i))

    items_with_ratio.sort(key=lambda x: x[0], reverse=True)

    sorted_w = []
    sorted_v = []
    for rat, idx in items_with_ratio:
        sorted_w.append(weights[idx])
        sorted_v.append(values[idx])

    prefix_w = [0.0] * (n + 1)
    prefix_v = [0.0] * (n + 1)
    for i in range(n):
        prefix_w[i + 1] = prefix_w[i] + sorted_w[i]
        prefix_v[i + 1] = prefix_v[i] + sorted_v[i]

    best_value: float = 0.0

    def bound(idx: int, current_val: float, remaining_cap: float) -> float:
        total_val = current_val
        if idx >= n:
            return total_val

        needed = prefix_w[n] - prefix_w[idx]
        if needed <= remaining_cap:
            return total_val + (prefix_v[n] - prefix_v[idx])

        limit = prefix_w[idx] + remaining_cap
        mid = bisect.bisect_right(prefix_w, limit)
        if mid > n:
            mid = n
        full_idx = max(idx, mid - 1)

        total_val += prefix_v[full_idx] - prefix_v[idx]

        if full_idx < n:
            leftover = limit - prefix_w[full_idx]
            ratio = items_with_ratio[full_idx][0]
            total_val += ratio * leftover

        return total_val

    def backtrack(
        idx: int, current_val: float, remaining_cap: float, used_indices: set[int]
    ) -> None:
        nonlocal best_value
        if current_val > best_value:
            best_value = current_val

        if idx >= n or remaining_cap <= 1e-9:
            return

        est = bound(idx, current_val, remaining_cap)
        if est <= best_value:
            return

        backtrack(idx + 1, current_val, remaining_cap, used_indices)

        w_i = sorted_w[idx]
        v_i = sorted_v[idx]
        if w_i <= remaining_cap + 1e-14:
            used_indices.add(idx)
            backtrack(idx + 1, current_val + v_i, remaining_cap - w_i, used_indices)
            used_indices.remove(idx)

    backtrack(0, 0.0, capacity, set())
    return best_value


class GetData:
    """
    A class for loading 0-1 Knapsack datasets and computing
    the optimal solutions using dp_knapsack_01, with caching of baseline solutions.
    """

    def __init__(self) -> None:
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
        the optimal solution for each instance using dp_knapsack_01. If a baseline
        cache exists for the same (mode, size), load it and skip recomputing. Then
        save updated baseline solutions to disk.

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

        try:
            data: np.ndarray = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if capacity is None:
            if size == "50":
                capacity = 12.5
            else:
                capacity = 25.0

        size_int: int = int(size)
        instances: dict[str, dict[str, float | list[float]]] = {}
        baseline: dict[str, float] = {}

        num_instances_in_file: int = data.shape[0]
        n_instances: int = num_instances_in_file

        baseline_filename: str = f"baseline_{mode}_{size}.npy"
        baseline_filepath: str = os.path.join(dataset_path, baseline_filename)

        cached_baseline = {}
        if os.path.exists(baseline_filepath):
            try:
                loaded = np.load(baseline_filepath, allow_pickle=True)
                if isinstance(loaded, np.ndarray):
                    # Typically this will be a 0D array containing the dict
                    # so we can do loaded.item() safely if shape=()
                    if loaded.shape == ():
                        possible_dict = loaded.item()
                        if isinstance(possible_dict, dict):
                            cached_baseline = possible_dict
            except Exception as e:
                print(
                    f"[WARNING] Could not load baseline cache: {baseline_filepath}, error={e}"
                )

        for i in range(n_instances):
            instance_name: str = f"{mode}_{size}_{i+1}"

            weights_arr: list[float] = data[i, :, 0].tolist()
            values_arr: list[float] = data[i, :, 1].tolist()

            instance_data: dict[str, float | list[float]] = {
                "capacity": capacity,
                "num_items": size_int,
                "weights": weights_arr,
                "values": values_arr,
            }
            instances[instance_name] = instance_data

            if instance_name in cached_baseline:
                baseline[instance_name] = cached_baseline[instance_name]
            else:
                try:
                    opt_val: float = dp_knapsack_01(weights_arr, values_arr, capacity)
                    baseline[instance_name] = opt_val
                    cached_baseline[instance_name] = opt_val
                except Exception as exc:
                    print(f"[ERROR] {instance_name}: {exc}")
                    baseline[instance_name] = -1.0
                    cached_baseline[instance_name] = -1.0

        try:
            # store updated baseline
            np.save(
                baseline_filepath,
                np.array(cached_baseline, dtype=object),
                allow_pickle=True,
            )
        except Exception as e:
            print(
                f"[WARNING] Could not save baseline cache: {baseline_filepath}, error={e}"
            )

        return instances, baseline


if __name__ == "__main__":
    data_loader = GetData()
    instances_dict, baseline_dict = data_loader.get_instances(
        size="100",
        capacity=None,
        mode="train",
        dataset_path="evaluation_kp/dataset",
    )

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
