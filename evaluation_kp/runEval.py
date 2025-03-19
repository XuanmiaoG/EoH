from __future__ import annotations  # Enables postponed evaluation of type annotations

"""
Evaluation code for EoH on Online Bin Packing.

More results may refer to:
Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang,
"Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model"
ICML 2024, https://arxiv.org/abs/2401.02051.
"""

import importlib
from types import ModuleType
from typing import Any

from get_instance import GetData  # Module to load bin packing instances.
from evaluation import Evaluation  # Module providing evaluation routines.


def main() -> None:
    """
    Main function to run evaluation on online bin packing instances.

    It iterates through various capacities and dataset sizes, dynamically loads
    the heuristic module, evaluates each instance using a greedy evaluation routine,
    and writes the results to 'results.txt'.
    """
    # Create an Evaluation instance for running evaluations.
    eva: Evaluation = Evaluation()

    # Define a list of bin capacities.
    capacity_list: list[int] = [100, 300, 500]
    # Define a list of dataset sizes.
    size_list: list[str] = ["1k", "5k", "10k"]

    # Open the output file in write mode.
    with open("results.txt", "w") as file:
        # Iterate over each capacity.
        for capacity in capacity_list:
            # Iterate over each dataset size.
            for size in size_list:
                # Instantiate GetData to load the appropriate dataset.
                getdata: GetData = GetData()
                # Retrieve instances and lower bounds.
                # Expected to return a tuple:
                #   - instances: dict[str, Any] where keys are instance names and values are datasets.
                #   - lb: dict[str, float] where keys are instance names and values are lower bounds.
                instances, lb = getdata.get_instances(
                    capacity, size
                )  # type: tuple[dict[str, Any], dict[str, float]]

                # Dynamically import the heuristic module and reload it.
                heuristic_module: ModuleType = importlib.import_module("heuristic")
                heuristic_module = importlib.reload(heuristic_module)

                # Evaluate each instance in the dataset.
                for name, dataset in instances.items():
                    # The evaluation method returns a negative value; we negate it to get the average number of bins.
                    avg_num_bins: float = -eva.evaluateGreedy(dataset, heuristic_module)
                    # Calculate the relative excess over the lower bound.
                    excess: float = (avg_num_bins - lb[name]) / lb[name]
                    # Format the result as a string.
                    result: str = f"{name}, {capacity}, Excess: {100 * excess:.2f}%"
                    print(result)
                    # Write the result to the output file.
                    file.write(result + "\n")


if __name__ == "__main__":
    main()
