import os
import torch
from torch_geometric.datasets import Planetoid, TUDataset
import shutil

def download_datasets(data_dir="./data"):
    """
    Download necessary graph datasets for the graph attack EoH framework.
    
    Parameters:
    -----------
    data_dir : str
        Directory to store the datasets
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading datasets...")
    
    # Download citation network datasets
    print("\n1. Downloading Citation Networks (Planetoid):")
    
    # Download CiteSeer
    print("\n   Downloading CiteSeer dataset...")
    try:
        citeseer = Planetoid(root=os.path.join(data_dir, 'CiteSeer'), name='CiteSeer')
        print(f"   CiteSeer dataset downloaded successfully: {len(citeseer)} graphs")
        print(f"   Number of nodes: {citeseer[0].num_nodes}")
        print(f"   Number of edges: {citeseer[0].num_edges // 2}")  # Divide by 2 for undirected graph
        print(f"   Number of node features: {citeseer[0].num_node_features}")
        print(f"   Number of classes: {citeseer[0].num_classes}")
    except Exception as e:
        print(f"   Error downloading CiteSeer dataset: {e}")
    
    # Download Cora
    print("\n   Downloading Cora dataset...")
    try:
        cora = Planetoid(root=os.path.join(data_dir, 'Cora'), name='Cora')
        print(f"   Cora dataset downloaded successfully: {len(cora)} graphs")
        print(f"   Number of nodes: {cora[0].num_nodes}")
        print(f"   Number of edges: {cora[0].num_edges // 2}")  # Divide by 2 for undirected graph
        print(f"   Number of node features: {cora[0].num_node_features}")
        print(f"   Number of classes: {cora[0].num_classes}")
    except Exception as e:
        print(f"   Error downloading Cora dataset: {e}")
    
    # Download TU datasets
    print("\n2. Downloading TU datasets:")
    
    # Download ENZYMES
    print("\n   Downloading ENZYMES dataset...")
    try:
        enzymes = TUDataset(root=os.path.join(data_dir, 'ENZYMES'), name='ENZYMES')
        print(f"   ENZYMES dataset downloaded successfully: {len(enzymes)} graphs")
        print(f"   Average number of nodes: {sum(g.num_nodes for g in enzymes) / len(enzymes):.1f}")
        print(f"   Average number of edges: {sum(g.num_edges for g in enzymes) / len(enzymes) / 2:.1f}")  # Divide by 2 for undirected graph
        print(f"   Number of node features: {enzymes[0].num_node_features}")
        print(f"   Number of classes: {enzymes.num_classes}")
    except Exception as e:
        print(f"   Error downloading ENZYMES dataset: {e}")
    
    print("\nAll datasets downloaded successfully to:", os.path.abspath(data_dir))
    print("Dataset structure:")
    print_directory_structure(data_dir)

def print_directory_structure(root_dir, indent=0):
    """
    Print the directory structure of the downloaded datasets.
    
    Parameters:
    -----------
    root_dir : str
        Root directory to start printing from
    indent : int
        Indentation level
    """
    if os.path.isdir(root_dir):
        print(" " * indent + f"📁 {os.path.basename(root_dir)}/")
        for item in os.listdir(root_dir):
            path = os.path.join(root_dir, item)
            if os.path.isdir(path):
                print_directory_structure(path, indent + 4)
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > 0.1:  # Only show files larger than 0.1 MB
                    print(" " * (indent + 4) + f"📄 {item} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    download_datasets()
    print("\nDatasets are now ready for use with the graph attack EoH framework!")