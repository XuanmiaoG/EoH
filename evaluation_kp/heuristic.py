import numpy as np


def calculate_dynamic_score(value, weight, remaining_capacity):
    if weight > remaining_capacity:
        return -1e9
    capacity_ratio = remaining_capacity / weight
    return value * capacity_ratio


def score(weight, value, remaining_capacity):
    score = calculate_dynamic_score(value, weight, remaining_capacity)
    return score
