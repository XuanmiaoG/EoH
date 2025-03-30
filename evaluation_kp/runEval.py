import importlib
from types import ModuleType
from typing import Any

from get_instance import GetData  # Module to load knapsack instances.
# Module providing evaluation routines for 0-1 knapsack.
from evaluation import Evaluation


def main() -> None:
    """
    Main function to run evaluation on 0-1 knapsack instances.

    It pairs each capacity with a corresponding dataset size (1-1 matching).
    For each (capacity, size) pair, it dynamically loads the heuristic module,
    evaluates each instance in the dataset using a greedy evaluation routine,
    and writes the results to 'results.txt'.

    Assumptions:
      - `GetData().get_instances(...)` returns:
            (instances, lb)
        where:
          - instances is a dict[str, Any], mapping each instance name to its data:
                {
                  "capacity": float,
                  "weights": list[float],
                  "values": list[float],
                  ...
                }
          - lb is a dict[str, float], mapping each instance name to the optimal (or best known) total value.

      - The 'heuristic' module is expected to define a `score(weight, value, remaining_capacity) -> float`.
      - `evaluateGreedy(...)` takes a dict of instances and the heuristic module,
        then returns the total value achieved by greedy selection.
    """

    # Create an Evaluation instance for running evaluations.
    eva: Evaluation = Evaluation()

    # Example: define a list of knapsack capacities (paired 1-to-1 with size_list).
    capacity_list: list[float] = [12.5, 25, 25]
    # Example: define a list of dataset sizes (paired 1-to-1 with capacity_list).
    size_list: list[str] = ["50", "100", "200"]

    # Open the output file in write mode.
    with open("results.txt", "w") as file:
        # Use zip to pair each capacity with the corresponding size
        for capacity, size in zip(capacity_list, size_list):
            total_gap: float = 0
            total_size: int = 0

            # Instantiate GetData to load the appropriate dataset.
            getdata: GetData = GetData()
            # print work directory
            import os
            print(f"Current working directory: {os.getcwd()}")

            # Retrieve instances and their known-optimal values (lb).
            instances, lb = getdata.get_instances(size=size, capacity=capacity)

            # Dynamically import (and reload) the heuristic module.
            heuristic_module: ModuleType = importlib.import_module("heuristic")
            heuristic_module = importlib.reload(heuristic_module)

            # Evaluate each instance in the dataset.
            for name, dataset in instances.items():
                # Wrap the single instance into a dictionary before passing to evaluateGreedy.
                total_value: float = eva.evaluateGreedy(
                    {name: dataset}, heuristic_module)

                # `lb[name]` is presumably the optimal or best-known total value.
                # Compute the relative gap = (Optimal - Achieved) / Optimal
                optimal_val = lb[name]
                if optimal_val != 0:
                    gap: float = (optimal_val - total_value) / optimal_val
                else:
                    gap = 0.0

                # Format the result as a string.
                result: str = (
                    f"Instance: {name}, Capacity: {capacity}, Size: {size}, "
                    f"Achieved: {total_value:.2f}, Optimal: {optimal_val:.2f}, "
                    f"Gap: {gap * 100:.2f}%"
                )

                total_gap += gap
                total_size += 1

                print(result)
                # Write the result to the output file.
                file.write(result + "\n")

            # Calculate and print the average gap for this (capacity, size) pair.
            avg_gap: float = total_gap / total_size if total_size > 0 else 0.0
            msg = f"Average gap for capacity {capacity}, size {size}: {avg_gap * 100:.2f}%"
            print(msg)
            file.write(msg + "\n\n")


if __name__ == "__main__":
    main()
