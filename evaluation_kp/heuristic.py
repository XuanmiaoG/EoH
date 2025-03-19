import numpy as np


def calculate_adjusted_score(value, weight, remaining_capacity):
    if weight > remaining_capacity:
        return -1e9
    adjustment = np.log(remaining_capacity + 1)
    base_score = (value / weight) ** 2
    return np.exp(base_score) * adjustment


def score(weight, value, remaining_capacity):
    score = calculate_adjusted_score(value, weight, remaining_capacity)
    return score
