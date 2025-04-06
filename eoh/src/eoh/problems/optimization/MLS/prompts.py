class GetPrompts:
    def __init__(self):
        # Task description for the Multi-Facility Location Mechanism Design problem
        self.prompt_task = (
            "In the Multi-Facility Location problem, we have n agents with single-peaked preferences "
            "on the interval [0,1]. Each agent i has a peak at position p_i, representing their ideal "
            "facility location. The cost for agent i from a facility at position x is |x - p_i|. We "
            "need to place K facilities to minimize the total weighted cost. Given the current node, "
            "a list of candidate nodes, and a distance matrix, the function should determine which "
            "node to select next for placing a facility. The goal is to strategically place facilities "
            "to minimize the sum of distances from each agent to their nearest facility."
        )
        
        # The name of the function to be defined
        self.prompt_func_name = "select_next_node"
        
        # The inputs of the function
        self.prompt_func_inputs = ["current_node", "destination_node", "unvisited_nodes", "distance_matrix"]
        
        # The output of the function
        self.prompt_func_outputs = ["next_node_index"]
        
        # Explanation of inputs/outputs
        self.prompt_inout_inf = (
            "'current_node' is the index of the current position, 'destination_node' is a placeholder "
            "that can generally be ignored, 'unvisited_nodes' is an array of indices representing "
            "candidate positions for the next facility, and 'distance_matrix' contains the pairwise "
            "distances between all positions. The output 'next_node_index' must be an integer index "
            "chosen from the unvisited_nodes array, representing the position for the next facility."
        )
        
        # Additional implementation details
        self.prompt_other_inf = (
            "Include the following import at the top of your code: 'import numpy as np'. The function "
            "should always return an integer index that exists in the unvisited_nodes array. The "
            "distance_matrix represents the Euclidean distances, where distance_matrix[i][j] is the "
            "distance between positions i and j. The function should select a node that strategically "
            "minimizes the overall distance cost in the facility location problem. Any helper functions "
            "should be placed immediately above the 'select_next_node' function definition."
            """ remember you shoould define the return value next_node_index!!!
Sample code:

import numpy as np

def calculate_benefit(current_node, candidate_node, unvisited_nodes, distance_matrix):
    # Example helper function to calculate the potential benefit of choosing a node
    total_benefit = 0
    for node in range(len(distance_matrix)):
        # Skip the current and candidate nodes themselves
        if node == current_node or node == candidate_node:
            continue
        # Calculate how this candidate would reduce distances
        current_dist = distance_matrix[node][current_node]
        candidate_dist = distance_matrix[node][candidate_node]
        # Benefit is positive if candidate is closer than current
        benefit = current_dist - candidate_dist
        total_benefit += max(0, benefit)
    return total_benefit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return destination_node
        
    # Calculate the benefit of each candidate node
    benefits = []
    for node in unvisited_nodes:
        benefit = calculate_benefit(current_node, node, unvisited_nodes, distance_matrix)
        benefits.append((benefit, node))
        
    # Choose the node with maximum benefit
    benefits.sort(reverse=True)
    return benefits[0][1]
"""
        )
    
    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf
    
    def get_other_inf(self):
        return self.prompt_other_inf