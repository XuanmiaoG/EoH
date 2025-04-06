import numpy as np


def calculate_enhanced_score(value, weight, remaining_capacity):
    if weight > remaining_capacity:
        return -1e9
    value_density = value / weight
    capacity_ratio = remaining_capacity / weight
    return value_density * np.log(value + 1) * (1 + capacity_ratio)


def score(weight, value, remaining_capacity):
    score = calculate_enhanced_score(value, weight, remaining_capacity)
    return score
