from __future__ import annotations
import os
import importlib

from get_instance import GetData  # Module to load knapsack instances

# Module providing evaluation routines for 0-1 knapsack
from evaluation import Evaluation


def main() -> None:
    """
    Main function to run evaluation on 0-1 knapsack instances.

    It pairs each capacity with a corresponding dataset size (1-to-1 matching).
    For each (capacity, size) pair, it dynamically loads the heuristic module,
    evaluates the dataset using a greedy routine, and writes summarized results
    to 'results.txt'.

    Assumptions:
      - get_instances(...) returns (instances, lb), where:
          instances is a dict of { name: { "capacity": float, "weights": [...], "values": [...] } }
          lb is a dict of { name: float }, mapping each instance to its optimal value.
      - The 'heuristic' module provides score(weight, value, remaining_capacity) -> float.
      - evaluateGreedy(...) takes a dict of instances and the heuristic module,
        returning the total value achieved by the greedy selection.
    """

    evaluator: Evaluation = Evaluation()

    capacity_list: list[float] = [12.5, 25.0, 25.0]
    size_list: list[str] = ["50", "100", "200"]

    with open("results.txt", "w", encoding="utf-8") as file:
        for capacity, size in zip(capacity_list, size_list):
            print(f"Current working directory: {os.getcwd()}")

            # Load data
            getdata: GetData = GetData()
            instances, lb = getdata.get_instances(size=size, capacity=capacity)

            # Reload heuristic module
            heuristic_module = importlib.import_module("heuristic")
            heuristic_module = importlib.reload(heuristic_module)

            # Accumulate total achieved and total optimal
            total_achieved: float = 0.0
            total_optimal: float = 0.0
            count: int = 0

            # Evaluate each instance in a single pass
            for name, data in instances.items():
                # Evaluate greedy solution
                achieved_value: float = evaluator.evaluateGreedy(
                    {name: data}, heuristic_module
                )
                optimal_value: float = lb[name]

                total_achieved += achieved_value
                total_optimal += optimal_value
                count += 1

            if count > 0:
                avg_achieved: float = total_achieved / count
                avg_optimal: float = total_optimal / count
                if avg_optimal != 0:
                    avg_gap: float = (avg_optimal - avg_achieved) / avg_optimal
                else:
                    avg_gap = 0.0
            else:
                avg_achieved = 0.0
                avg_optimal = 0.0
                avg_gap = 0.0

            # Print only the aggregated results
            summary_line: str = (
                f"Capacity: {capacity}, Size: {size}, "
                f"Achieved: {avg_achieved:.4f}, Optimal: {avg_optimal:.4f}, "
                f"Gap: {avg_gap * 100:.4f}%"
            )
            print(summary_line)
            file.write(summary_line + "\n")


if __name__ == "__main__":
    main()
