import numpy as np


def place_facilities(peaks, weights, k):
    candidate_positions = np.linspace(0, 1, 101)
    facilities_positions = []
    for position in candidate_positions:
        cost = np.sum(weights * np.abs(peaks - position))
        facilities_positions.append((position, cost))
    facilities_positions.sort(key=lambda x: x[1])
    best_positions = [fac_position[0] for fac_position in facilities_positions[:k]]

    return facilities_positions
