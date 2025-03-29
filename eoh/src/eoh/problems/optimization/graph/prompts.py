class GetGraphAttackPrompts():
    def __init__(self):
        self.prompt_task = "I need help designing a novel heuristic algorithm for targeted graph attacks. \
The goal is to select edges to modify (add or remove) in a graph to maximize the attack success rate \
on target nodes while minimizing the impact on non-target nodes. The attack aims to change the prediction \
of target nodes by a graph neural network model."
        
        self.prompt_func_name = "select_edges"
        
        self.prompt_func_inputs = ['adj_matrix', 'node_features', 'node_labels', 'target_nodes', 'attack_budget']
        
        self.prompt_func_outputs = ['modified_adj_matrix']
        
        self.prompt_inout_inf = "'adj_matrix' is an N×N numpy array representing the adjacency matrix of the graph. \
'node_features' is an N×D numpy array of node features. \
'node_labels' is an N-dimensional numpy array containing the node class labels. \
'target_nodes' is a list of indices representing the nodes to be attacked. \
'attack_budget' is an integer specifying the maximum number of edges that can be modified. \
The output 'modified_adj_matrix' should be an N×N numpy array representing the adjacency matrix after the attack."
        
        self.prompt_other_inf = "Your heuristic should consider node importance, centrality measures, or graph structural \
properties to determine which edges to modify. The attack can add new edges or remove existing ones, but the total number \
of modifications should not exceed the attack budget. The goal is to maximize attack success (changing model predictions \
for target nodes) while being stealthy (minimizing changes to non-target nodes' predictions). \
\
Performance will be evaluated against a baseline random attack strategy. A good heuristic should achieve higher attack \
success rate while maintaining lower impact on non-target nodes. The fitness function is calculated as a negative value, \
where lower values (more negative) indicate better performance compared to the baseline. \
\
Import numpy as np and ensure your function is sufficiently complex to achieve strong attack performance."

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
        
    def get_initialization_prompt(self):
        """Return the prompt for initializing attack heuristics"""
        return f"""{self.prompt_task}

Please design a new heuristic.

Firstly, describe your new heuristic and main steps in a few sentences.

Next, implement it in Python as a function named '{self.prompt_func_name}'. 
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency. 
Do not give additional explanations beyond the heuristic description and code implementation.
"""

    def get_evolution_prompt_e1(self, parent_heuristics):
        """Return the E1 prompt (exploration) for evolving attack heuristics"""
        parents_str = self._format_parent_heuristics(parent_heuristics)
        
        return f"""{self.prompt_task}

I have several existing heuristics with their descriptions and codes as follows:

{parents_str}

Please help me design a new heuristic that is as much different as possible from the given ones, 
to explore new ideas for graph attacks.

Firstly, describe your new heuristic and main steps in a few sentences.

Next, implement it in Python as a function named '{self.prompt_func_name}'.
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency.
Do not give additional explanations beyond the heuristic description and code implementation.
"""

    def get_evolution_prompt_e2(self, parent_heuristics):
        """Return the E2 prompt (exploration with common ideas) for evolving attack heuristics"""
        parents_str = self._format_parent_heuristics(parent_heuristics)
        
        return f"""{self.prompt_task}

I have several existing heuristics with their descriptions and codes as follows:

{parents_str}

Please help me design a new heuristic by first identifying common ideas in the provided heuristics,
then creating a new approach that builds on these common ideas but introduces novel elements.

Firstly, identify the common idea in the provided heuristics.

Secondly, describe your new heuristic and main steps in a few sentences, building on the common ideas.

Next, implement it in Python as a function named '{self.prompt_func_name}'.
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency.
Do not give additional explanations beyond the common idea identification, heuristic description, and code implementation.
"""

    def get_evolution_prompt_m1(self, parent_heuristic):
        """Return the M1 prompt (modification) for evolving attack heuristics"""
        heuristic_str = self._format_single_heuristic(parent_heuristic)
        
        return f"""{self.prompt_task}

I have an existing heuristic with its description and code as follows:

{heuristic_str}

Please help me modify this heuristic to improve its performance for graph attacks.

Firstly, describe your modified heuristic and main steps in a few sentences.

Next, implement it in Python as a function named '{self.prompt_func_name}'.
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency.
Do not give additional explanations beyond the heuristic description and code implementation.
"""

    def get_evolution_prompt_m2(self, parent_heuristic):
        """Return the M2 prompt (parameter modification) for evolving attack heuristics"""
        heuristic_str = self._format_single_heuristic(parent_heuristic)
        
        return f"""{self.prompt_task}

I have an existing heuristic with its description and code as follows:

{heuristic_str}

Please help me modify the parameters of this heuristic to improve its performance, 
without changing its overall structure or approach.

Firstly, describe your parameter modifications and their expected impact in a few sentences.

Next, implement the modified version in Python as a function named '{self.prompt_func_name}'.
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency.
Do not give additional explanations beyond the modification description and code implementation.
"""

    def get_evolution_prompt_m3(self, parent_heuristic):
        """Return the M3 prompt (simplification) for evolving attack heuristics"""
        heuristic_str = self._format_single_heuristic(parent_heuristic)
        
        return f"""{self.prompt_task}

I have an existing heuristic with its description and code as follows:

{heuristic_str}

Please help me simplify this heuristic by identifying and removing redundant components while preserving its core functionality.

Firstly, analyze the heuristic and identify its main components and any redundant parts.

Secondly, describe your simplified heuristic in a few sentences.

Next, implement the simplified version in Python as a function named '{self.prompt_func_name}'.
This function should accept these inputs: {', '.join(self.prompt_func_inputs)}.
The function should return one output: {self.prompt_func_outputs[0]}.

{self.prompt_inout_inf}

{self.prompt_other_inf}

Note: Avoid utilizing random components, and it is crucial to maintain self-consistency.
Do not give additional explanations beyond the analysis, simplified description, and code implementation.
"""
    
    def _format_parent_heuristics(self, parent_heuristics):
        """Format a list of parent heuristics for inclusion in prompts"""
        result = ""
        for i, heuristic in enumerate(parent_heuristics, 1):
            result += f"No.{i} Heuristic description:\n{heuristic['description']}\n\n"
            result += f"Code:\n{heuristic['code']}\n\n"
        return result
    
    def _format_single_heuristic(self, heuristic):
        """Format a single parent heuristic for inclusion in prompts"""
        return f"Heuristic description:\n{heuristic['description']}\n\nCode:\n{heuristic['code']}"