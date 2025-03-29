import numpy as np
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx

class GetGraphData():
    def __init__(self):
        # Initialize datasets dictionary
        self.graph_datasets = {}
        
        # Load CiteSeer dataset
        self.load_planetoid_datasets()
        
        # Optionally load additional graph datasets
        self.load_tu_datasets()
        
    def load_planetoid_datasets(self):
        """Load and prepare the citation network datasets"""
        # Load Citeseer
        citeseer = Planetoid(root='./data', name='Citeseer')
        self.graph_datasets['Citeseer'] = citeseer[0]  # Citeseer dataset has a single graph
        
        # Load Cora
        cora = Planetoid(root='./data', name='Cora')
        self.graph_datasets['Cora'] = cora[0]  # Cora dataset has a single graph
        
    def load_tu_datasets(self):
        """Load and prepare additional graph datasets from the TU collection"""
        try:
            # ENZYMES dataset (enzyme tertiary structures)
            enzymes = TUDataset(root='./data', name='ENZYMES')
            # Take a subset of graphs (first 5) for diversity
            for i in range(min(5, len(enzymes))):
                self.graph_datasets[f'ENZYMES_{i}'] = enzymes[i]
        except Exception as e:
            print(f"Error loading TU datasets: {e}")
    
    def calculate_graph_metrics(self):
        """Calculate key metrics for each graph dataset"""
        metrics = {}
        
        for name, graph in self.graph_datasets.items():
            # Convert to networkx for easier analysis
            G = to_networkx(graph, to_undirected=True)
            
            # Calculate basic graph metrics
            metrics[name] = {
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges,
                'avg_degree': 2 * graph.num_edges / graph.num_nodes,
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G),
                'num_connected_components': nx.number_connected_components(G),
            }
            
            # Calculate vulnerability score (simplified)
            # Higher score means the graph is more vulnerable to attacks
            vulnerability_score = (
                metrics[name]['density'] + 
                metrics[name]['clustering_coefficient'] -
                (metrics[name]['num_connected_components'] / metrics[name]['num_nodes'])
            )
            metrics[name]['vulnerability_score'] = vulnerability_score
            
        return metrics
        
    def get_instances(self):
        """Return the loaded graph datasets and their metrics"""
        # Calculate metrics for each graph dataset
        metrics = self.calculate_graph_metrics()
        
        print("Loaded graph datasets:")
        for name, metric in metrics.items():
            print(f"  {name}: {metric['num_nodes']} nodes, {metric['num_edges']} edges")
            
        return self.graph_datasets, metrics