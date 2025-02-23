import numpy as np
from numba import jit

@jit(nopython=True)
def score(weight, value, remaining_capacity):
    """
    A simple example heuristic for knapsack:
    If the item does not fit, return a very low score.
    Otherwise, return the value-to-weight ratio.
    (You can modify this to add further complexity.)
    """
    if weight > remaining_capacity:
        return -1e9
    return value / weight
