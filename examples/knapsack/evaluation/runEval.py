import importlib
from get_instance import GetData
from evaluation import Evaluation

eva = Evaluation()

# Test instance sizes: '50', '100', '200', '500'
sizes = ['50', '100', '200', '500']
with open("results.txt", "w") as file:
    for size in sizes:
        instances, lb = GetData().get_instances(size)
        eva.instances = instances
        eva.lb = lb
        heuristic_module = importlib.import_module("heuristic")
        heuristic = importlib.reload(heuristic_module)
        fitness = eva.evaluateGreedy(instances, heuristic)
        result = f"Knapsack {size} items, Fitness: {100 * fitness:.2f}%"
        print(result)
        file.write(result + "\n")
