import numpy as np
import warnings
import types
import sys

# from func_timeout import func_set_timeout  # if you wish to use timeouts

# Local imports (adjust if needed)
from .get_instance import GetData
from .prompts import GetPrompts

class MLS:
    def __init__(self):

        # 1) Basic parameters (you can adapt these as necessary)
        self.n_instances = 3  # how many instances we want to evaluate
        self.max_nodes = 10  # up to how many nodes in each instance
        self.running_time = 10  # placeholder for any time budget
        self.neighbor_size = 5  # how many neighbors to consider if doing partial search

        # 2) Get the prompt manager
        self.prompts = GetPrompts()

        # 3) Get the dataset
        #    For demonstration, we request dataset from GetData
        #    that might produce 1D peaks in [0,1]. We'll interpret them as 2D with y=0.
        #    distributions, n_values, etc. were set in GetData, but we can also re-init if we want.
        get_data = GetData(
            distributions=["uniform", "normal", "beta1", "beta2"],
            n_values=[5, 10],  # could be changed
            k_values=[1, 2],  # not really used in TSP, but is part of the original code
            samples_per_setting=3,
            seed=42,
        )
        self.all_data, _ = get_data.get_instances()

        # We'll flatten out some portion of these datasets into self.instance_data
        # to be used by our TSP code.
        self.instance_data = []
        self._prepare_instances()

    def _prepare_instances(self):
        """
        Convert the 1D peaks from each instance into 2D coordinates,
        then build a distance matrix, storing them in self.instance_data.

        self.instance_data will be a list of tuples:
          (coords, distance_matrix)
        where coords is an array of shape (n, 2) and distance_matrix is shape (n, n).
        """
        # We only take the first self.n_instances we come across
        collected = 0
        for dataset_key, instances_dict in self.all_data.items():
            for inst_name, inst_content in instances_dict.items():
                if collected >= self.n_instances:
                    return
                n = inst_content["n"]
                # Convert the 1D "peaks" into 2D coords: (peak, 0.0)
                peaks_1d = inst_content["peaks"]  # shape (n,)
                coords_2d = np.zeros((n, 2))
                coords_2d[:, 0] = peaks_1d  # x = peak
                coords_2d[:, 1] = 0.0  # y = 0

                dist_mat = self._compute_distance_matrix(coords_2d)

                self.instance_data.append((coords_2d, dist_mat))
                collected += 1

                if collected >= self.n_instances:
                    break

    def _compute_distance_matrix(self, coords):
        """
        Compute the Euclidean distance matrix for given 2D coordinates.
        coords has shape (n, 2).
        Return a matrix dist_mat of shape (n, n).
        """
        n = coords.shape[0]
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = np.linalg.norm(coords[i] - coords[j])
        return dist_mat

    def route_cost(self, coords, route):
        """
        Given an ordering of nodes in 'route' (list or array of node indices),
        compute the total cost of traveling in that order plus returning to the start.
        """
        cost = 0.0
        n = len(route)
        for i in range(n - 1):
            cost += np.linalg.norm(coords[route[i]] - coords[route[i + 1]])
        # add distance from last node back to first
        cost += np.linalg.norm(coords[route[-1]] - coords[route[0]])
        return cost

    def generate_neighbor_matrix(self, coords):
        """
        For each node, find the nearest neighbors in ascending distance order.
        Return an (n x n) array neighbor_matrix[i] = sorted list of node indices by
        distance from node i.
        """
        n = coords.shape[0]
        neighbor_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            dist = np.linalg.norm(coords[i] - coords, axis=1)
            sorted_indices = np.argsort(dist)
            neighbor_matrix[i] = sorted_indices
        return neighbor_matrix

    def simple_greedy_tour(self, heuristic_module):
        """
        Demonstrates how to do a TSP-like greedy approach that repeatedly
        calls the user-provided function 'select_next_node(...)'.
        We measure the average route cost across the set of self.instance_data.

        heuristic_module is a Python module that must define
        the function 'select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)'.
        """
        all_costs = []

        for coords_2d, dist_mat in self.instance_data:
            n = coords_2d.shape[0]
            # build a neighbor matrix if we want partial search
            neighbor_mat = self.generate_neighbor_matrix(coords_2d)

            # Suppose we fix the "destination_node" to be 0, just as a placeholder
            destination_node = 0
            # Start from node 0
            current_node = 0

            route = [current_node]

            # Repeatedly pick next nodes until we fill out all n
            for step in range(1, n - 1):
                # For demonstration, pick nearest neighbors, but only among those not visited
                neighbors_sorted = neighbor_mat[current_node]
                # Filter out visited
                unvisited_mask = ~np.isin(neighbors_sorted, route)
                unvisited_near = neighbors_sorted[unvisited_mask]

                # clip to neighbor_size
                unvisited_near = unvisited_near[: self.neighbor_size]

                # call the user-supplied function
                next_node = heuristic_module.select_next_node(
                    current_node, destination_node, unvisited_near, dist_mat
                )
                # check validity
                if (next_node in route) or (next_node < 0) or (next_node >= n):
                    # invalid
                    return None

                route.append(next_node)
                current_node = next_node

            # fill the last node
            # we will simply pick whichever is not in route
            unvisited = np.setdiff1d(np.arange(n), route)
            if len(unvisited) != 1:
                return None
            last_node = unvisited[0]
            route.append(last_node)

            cost_val = self.route_cost(coords_2d, route)
            all_costs.append(cost_val)

        return np.mean(all_costs) if len(all_costs) > 0 else None

    def evaluate(self, code_string):
            """
            Evaluate the user-provided code string as a python module,
            searching for the function 'select_next_node(...).'
            Returns the average route cost across self.instance_data if successful,
            or None if an error occurs.
            """
            try:
                # Print the code string for debugging
                print("Debug - Code String:")
                print(code_string)

                # Safely create a new module
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                
                # Now attempt to run the TSP-like routine
                avg_cost = self.simple_greedy_tour(heuristic_module)
                return avg_cost
            except Exception as e:
                # If anything goes wrong, return None
                print("Error in evaluate:", e)
                return None


