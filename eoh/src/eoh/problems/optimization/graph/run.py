import numpy as np
import importlib
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import networkx as nx
import types
import warnings
import sys
import copy
from tqdm import tqdm
from .get_instance import GetGraphData
from .prompts import GetGraphAttackPrompts

class GraphAttackFramework():
    def __init__(self):
        # Get datasets and prompts
        self.data_handler = GetGraphData()
        self.graph_data, self.graph_metrics = self.data_handler.get_instances()
        self.prompts = GetGraphAttackPrompts()
        
        # Train a model for each dataset separately
        self.target_models = {}
        self.init_target_models()
        
        # Calculate baseline attack efficiency
        self.baseline_efficiency = self._calculate_baseline_attack_efficiency()
        
    def init_target_models(self):
        """Initialize and train a separate GNN model for each dataset"""
        from torch_geometric.nn import GCNConv
        import torch.nn.functional as F
        import torch.optim as optim
        
        print("Initializing and training GNN models for each dataset...")
        
        # Define GCN model class outside the loop
        class GCN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GCN, self).__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                return x
        
        # Create and train a model for each dataset
        for name, data in self.graph_data.items():
            print(f"\nProcessing dataset: {name}")
            
            # Make sure data is on CPU
            if hasattr(data, 'x'):
                data.x = data.x.cpu()
            if hasattr(data, 'edge_index'):
                data.edge_index = data.edge_index.cpu()
            if hasattr(data, 'y'):
                data.y = data.y.cpu()
            
            # Get dimensions
            in_features = data.x.size(1)
            
            # Calculate number of classes
            if hasattr(data, 'y') and data.y is not None:
                num_classes = int(data.y.max().item()) + 1
            else:
                # Default to binary classification if no labels
                num_classes = 2
                
            print(f"  Features: {in_features}, Classes: {num_classes}")
            
            # Create model
            model = GCN(
                in_channels=in_features,
                hidden_channels=16,
                out_channels=num_classes
            )
            
            # Create a train mask if it doesn't exist
            if not hasattr(data, 'train_mask'):
                num_nodes = data.num_nodes
                indices = torch.randperm(num_nodes)
                train_size = int(0.8 * num_nodes)
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.train_mask[indices[:train_size]] = True
                data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.test_mask[indices[train_size:]] = True
                print(f"  Created train/test masks with {train_size} training nodes")
            
            # Train model for a few epochs
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            model.train()
            
            # Skip training for non-citation datasets to save time
            if 'ENZYMES' in name:
                print(f"  Skipping full training for {name} to save time")
                # Just do a forward pass to initialize weights
                try:
                    _ = model(data.x, data.edge_index)
                except Exception as e:
                    print(f"  Error in forward pass: {e}")
                    print(f"  Using dummy model for {name}")
                    # If forward pass fails, create a dummy model
                    model = DummyModel(in_features, num_classes)
            else:
                # Train on citation dataset
                print(f"  Training model for {name}...")
                for epoch in range(3):  # Just 3 epochs for quick training
                    try:
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index)
                        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()
                        
                        # Calculate accuracy
                        pred = out.argmax(dim=1)
                        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
                        acc = int(correct) / int(data.train_mask.sum())
                        print(f'    Epoch: {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {acc:.4f}')
                    except Exception as e:
                        print(f"  Error during training: {e}")
                        print(f"  Using dummy model for {name}")
                        # If training fails, create a dummy model
                        model = DummyModel(in_features, num_classes)
                        break
            
            # Set to evaluation mode
            model.eval()
            
            # Store model
            self.target_models[name] = model
            
        print("All models initialized and trained.")
    
    def apply_attack_heuristic(self, graph, target_nodes, attack_budget, attack_heuristic, dataset_name=None):
        """Apply the attack heuristic to modify the graph"""
        # Extract graph components
        edge_index = graph.edge_index
        x = graph.x
        y = graph.y
        
        # Convert to adjacency matrix for easier manipulation
        adj_matrix = to_dense_adj(edge_index)[0]
        
        # Apply the heuristic to select edges to modify
        try:
            modified_adj = attack_heuristic.select_edges(
                adj_matrix.numpy(), 
                x.numpy(), 
                y.numpy(), 
                target_nodes, 
                attack_budget
            )
        except Exception as e:
            print(f"Error in attack heuristic: {e}")
            # If heuristic fails, just return the original adjacency matrix
            return graph
        
        # Convert back to edge_index format
        new_edge_index, _ = dense_to_sparse(torch.tensor(modified_adj))
        
        # Create modified graph with the same node features but new edges
        modified_graph = copy.deepcopy(graph)
        modified_graph.edge_index = new_edge_index
        
        return modified_graph
        
    def evaluate_attack(self, original_graph, modified_graph, target_nodes, dataset_name):
        """Evaluate the effectiveness of the attack using the appropriate model for the dataset"""
        # Get the appropriate model for this dataset
        if dataset_name not in self.target_models:
            print(f"No model found for dataset {dataset_name}, using first available model")
            model = next(iter(self.target_models.values()))
        else:
            model = self.target_models[dataset_name]
        
        # Forward pass on original graph
        try:
            with torch.no_grad():
                original_output = model(original_graph.x, original_graph.edge_index)
                original_pred = original_output.argmax(dim=1)
        except Exception as e:
            print(f"Error evaluating original graph: {e}")
            # Return default metrics
            return {'success_rate': 0.0, 'non_target_impact': 0.0, 'attack_efficiency': 0.0}
        
        # Forward pass on modified graph
        try:
            with torch.no_grad():
                modified_output = model(modified_graph.x, modified_graph.edge_index)
                modified_pred = modified_output.argmax(dim=1)
        except Exception as e:
            print(f"Error evaluating modified graph: {e}")
            # Return default metrics
            return {'success_rate': 0.0, 'non_target_impact': 0.0, 'attack_efficiency': 0.0}
        
        # Calculate attack success rate (how many target node predictions changed)
        success_count = sum(original_pred[idx] != modified_pred[idx] for idx in target_nodes)
        success_rate = success_count / len(target_nodes) if len(target_nodes) > 0 else 0.0
        
        # Calculate impact on non-target nodes (lower is better)
        non_target_nodes = [i for i in range(len(original_pred)) if i not in target_nodes]
        if non_target_nodes:
            non_target_impact = sum(original_pred[idx] != modified_pred[idx] for idx in non_target_nodes) / len(non_target_nodes)
        else:
            non_target_impact = 0.0
        
        # Calculate the attack efficiency score (higher is better)
        attack_efficiency = success_rate - 0.5 * non_target_impact
        
        return {
            'success_rate': success_rate,
            'non_target_impact': non_target_impact,
            'attack_efficiency': attack_efficiency
        }
    
    # Add this import at the top of your file
    

    def evaluate_heuristic(self, attack_heuristic):
        """Evaluate the attack heuristic on all datasets with progress bars"""
        results = []
        
        # Create a progress bar for dataset evaluation
        for name, graph in tqdm(self.graph_data.items(), desc="Evaluating datasets", unit="dataset"):
            print(f"\nEvaluating on {name}...")
            
            # Select target nodes (5% of nodes)
            num_nodes = graph.num_nodes
            target_size = max(1, int(0.05 * num_nodes))  # Ensure at least 1 target
            try:
                target_nodes = np.random.choice(num_nodes, size=target_size, replace=False)
            except:
                # If random choice fails, just take the first few nodes
                target_nodes = np.arange(target_size)
            
            # Define attack budget (3% of edges)
            attack_budget = max(1, int(0.03 * graph.num_edges // 2))  # Ensure at least 1 edge
            
            # Apply the attack
            try:
                print(f"  Applying attack (budget: {attack_budget} edges)...")
                modified_graph = self.apply_attack_heuristic(
                    graph, target_nodes, attack_budget, attack_heuristic, name
                )
                
                # Evaluate the attack
                print(f"  Evaluating attack effectiveness...")
                attack_metrics = self.evaluate_attack(graph, modified_graph, target_nodes, name)
                results.append(attack_metrics['attack_efficiency'])
                print(f"  Attack efficiency: {attack_metrics['attack_efficiency']:.4f}")
            except Exception as e:
                print(f"  Error evaluating heuristic on {name}: {e}")
                # Add a default poor result
                results.append(0.0)
        
        # Calculate average attack efficiency across all datasets
        avg_efficiency = np.mean(results) if results else 0.0
        print(f"\nAverage attack efficiency: {avg_efficiency:.4f}")
        
        # Compare against baseline attack
        baseline_efficiency = self.get_baseline_attack_efficiency()
        
        # Calculate relative performance
        if baseline_efficiency == 0:
            relative_performance = 0.0  # Avoid division by zero
        else:
            relative_performance = (avg_efficiency - baseline_efficiency) / abs(baseline_efficiency)
        
        # Make smaller values better (for minimization objective)
        fitness = -relative_performance
        print(f"Relative performance: {relative_performance:.4f}, Fitness: {fitness:.4f}")
        
        return fitness
        
    def _calculate_baseline_attack_efficiency(self):
        """Calculate the efficiency of a baseline random attack strategy"""
        print("Calculating baseline attack efficiency...")
        
        class RandomAttackHeuristic:
            def select_edges(self, adj_matrix, node_features, node_labels, target_nodes, attack_budget):
                n_nodes = adj_matrix.shape[0]
                modified_adj = adj_matrix.copy()
                
                # Randomly select edges to modify
                # First, get all possible edge pairs (i,j) where i < j to avoid duplicates
                all_edge_pairs = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
                
                # Shuffle these pairs
                np.random.shuffle(all_edge_pairs)
                
                # Apply modifications up to the budget
                modifications_made = 0
                for i, j in all_edge_pairs:
                    if modifications_made >= attack_budget:
                        break
                    
                    # Flip the edge status (add if absent, remove if present)
                    if modified_adj[i, j] == 0:
                        modified_adj[i, j] = 1
                        modified_adj[j, i] = 1  # Maintain symmetry for undirected graph
                    else:
                        modified_adj[i, j] = 0
                        modified_adj[j, i] = 0
                    
                    modifications_made += 1
                
                return modified_adj
        
        # Create an instance of the random attack heuristic
        random_heuristic = RandomAttackHeuristic()
        
        # Evaluate the random attack strategy
        results = []
        for name, graph in self.graph_data.items():
            print(f"Calculating baseline for {name}...")
            
            # Select target nodes (5% of nodes)
            num_nodes = graph.num_nodes
            target_size = max(1, int(0.05 * num_nodes))  # Ensure at least 1 target
            try:
                target_nodes = np.random.choice(num_nodes, size=target_size, replace=False)
            except:
                # If random choice fails, just take the first few nodes
                target_nodes = np.arange(target_size)
            
            # Define attack budget (3% of edges)
            attack_budget = max(1, int(0.03 * graph.num_edges // 2))  # Ensure at least 1 edge
            
            # Apply the random attack
            try:
                modified_graph = self.apply_attack_heuristic(
                    graph, target_nodes, attack_budget, random_heuristic, name
                )
                
                # Evaluate the attack
                attack_metrics = self.evaluate_attack(graph, modified_graph, target_nodes, name)
                results.append(attack_metrics['attack_efficiency'])
                print(f"  Baseline attack efficiency: {attack_metrics['attack_efficiency']:.4f}")
            except Exception as e:
                print(f"  Error calculating baseline for {name}: {e}")
        
        baseline_efficiency = np.mean(results) if results else 0.0
        print(f"Average baseline attack efficiency: {baseline_efficiency:.4f}")
        return baseline_efficiency
    
    def get_baseline_attack_efficiency(self):
        """Get the cached baseline attack efficiency"""
        return self.baseline_efficiency
    
    def evaluate(self, code_string):
        """Evaluate a given heuristic code string"""
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Print the code string for debugging
                print("Evaluating function source:")
                print(code_string)
                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)
                
                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module
                
                # Evaluate the heuristic
                fitness = self.evaluate_heuristic(heuristic_module)
                
                return fitness
        except Exception as e:
            print(f"Error evaluating heuristic: {str(e)}")
            # Return a very poor fitness value
            return 1000.0  # A large positive value for a minimization problem


class DummyModel(torch.nn.Module):
    """A dummy model that always returns random predictions, used as fallback"""
    def __init__(self, in_features, num_classes):
        super(DummyModel, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
    
    def forward(self, x, edge_index):
        # Generate random logits
        batch_size = x.size(0)
        return torch.randn(batch_size, self.num_classes)