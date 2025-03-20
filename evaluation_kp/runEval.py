import importlib
from types import ModuleType
from typing import Any

from get_instance import GetData  # Module to load knapsack instances.
from evaluation import (
    Evaluation,
)  # Module providing evaluation routines for 0-1 knapsack.


def main() -> None:
    """
    Main function to run evaluation on 0-1 knapsack instances.

    It iterates through various capacities and dataset sizes, dynamically loads
    the heuristic module, evaluates each instance using a greedy evaluation routine,
    and writes the results to 'results.txt'.

    Assumptions:
      - `GetData().get_instances(...)` returns:
            (instances, lb)
        where:
          - instances is a dict[str, Any], mapping each instance name to its data:
                {
                  "capacity": int,
                  "weights": list[int],
                  "values": list[int],
                  ...
                }
          - lb is a dict[str, float], mapping each instance name to the optimal (or best known) total value.
    """
    # Create an Evaluation instance for running evaluations.
    eva: Evaluation = Evaluation()

    # Example: define a list of knapsack capacities.
    capacity_list: list[int] = [100, 300, 500]
    # Example: define a list of dataset sizes.
    size_list: list[str] = ["50", "100", "200"]

    # Open the output file in write mode.
    with open("results.txt", "w") as file:
        # Iterate over each capacity.
        for capacity in capacity_list:
            # Iterate over each dataset size.
            for size in size_list:
                total_gap: float = 0
                total_size: int = 0
                # Instantiate GetData to load the appropriate dataset.
                getdata: GetData = GetData()
                # Retrieve instances and their known-optimal values (lb).
                #   - `instances` is a dict of: { instance_name: { "capacity": ..., "weights": [...], "values": [...] } }
                #   - `lb` is a dict of: { instance_name: best_known_value }
                instances, lb = getdata.get_instances(size=size, capacity=capacity)

                # Dynamically import the heuristic module and reload it.
                # The 'heuristic' module is expected to define a `score(weight, value, remaining_capacity) -> float`.
                heuristic_module: ModuleType = importlib.import_module("heuristic")
                heuristic_module = importlib.reload(heuristic_module)

                # Evaluate each instance in the dataset.
                for name, dataset in instances.items():
                    # Wrap the single instance into a dictionary before passing to evaluateGreedy.
                    total_value: float = eva.evaluateGreedy(
                        {name: dataset}, heuristic_module
                    )
                    # `lb[name]` is presumably the optimal or best-known total value.
                    # Compute how far off (or above/below) we are from optimal:
                    # gap (relative) = (Optimal - Achieved) / Optimal
                    gap: float = (
                        (lb[name] - total_value) / lb[name] if lb[name] != 0 else 0
                    )

                    # Format the result as a string. You can multiply by 100 to express as a percentage.
                    result: str = (
                        f"Instance: {name}, Capacity: {capacity}, Size: {size}, "
                        f"Achieved: {total_value:.2f}, Optimal: {lb[name]:.2f}, "
                        f"Gap: {gap * 100:.2f}%"
                    )
                    total_gap += gap
                    total_size += 1
                    print(result)
                    # Write the result to the output file.
                    file.write(result + "\n")
                # Calculate the average gap for the current capacity and size.
                avg_gap: float = total_gap / total_size
                print(
                    f"Average gap for capacity {capacity}, size {size}: {avg_gap * 100:.2f}%"
                )


if __name__ == "__main__":
    main()
