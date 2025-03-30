import numpy as np


def calculate_best_heuristic(value, weight, remaining_capacity):
    if weight > remaining_capacity:
        return -1e9
    ratio = value / weight
    capacity_factor = np.log((remaining_capacity / weight) + 1)
    return ratio * capacity_factor


def score(weight, value, remaining_capacity):
    score = float(calculate_best_heuristic(value, weight, remaining_capacity))
    return score
