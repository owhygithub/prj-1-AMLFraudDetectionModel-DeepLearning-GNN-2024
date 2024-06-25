import pickle
import torch
import numpy as np
import networkx as nx
from functions1 import AMLDataPreprocessing
from scipy.sparse import coo_matrix, save_npz

print("Import Successful...")

dataset = "HI-Small_Trans_half"
filename = f'/var/scratch/hwg580/{dataset}.csv'

# Create an instance of the AMLDataPreprocessing class
print("Initializing AML Data Preprocessing...")
data_preprocessor = AMLDataPreprocessing(filename)
# Process the data
print("Processing data...")
input_data, graph_full, x, y, labels, links, edges_amount, node_features, edges_features, time_closeness = data_preprocessor.process_data()
print("Data Processed Successfully!")

# Visualize
visual = data_preprocessor.visualize_graph(links, labels)

# Convert to a boolean adjacency matrix
print("Adjacency matrix...")
adjacency_matrix = nx.adjacency_matrix(graph_full).astype(bool)
print("Adjacency matrix created...\n")

# Convert the adjacency matrix to a PyTorch tensor
print("Adjacency matrix tensor...")
# Save the sparse adjacency matrix to a file
save_npz("/var/scratch/hwg580/adjacency_matrix.npz", adjacency_matrix)

# Convert the adjacency matrix to COO format and then to a PyTorch sparse tensor
coo = coo_matrix(adjacency_matrix)
values = torch.tensor(coo.data, dtype=torch.float32)
indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)
adjacency_tensor = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
print(adjacency_tensor)
print(f"Size of adjacency_tensor: {adjacency_tensor.size()}")

print("Adjacency matrix tensor created...")

print(f"input data: {input_data}")

with open(f"/var/scratch/hwg580/graph_{dataset}.pickle", "wb") as f:
    pickle.dump({
        'dataset': dataset,
        'visual': visual,
        'edges_features': edges_features,
        'links': links,
        'graph_full': graph_full,
        'adjacency_matrix': adjacency_matrix,
        'adjacency_tensor': adjacency_tensor,
        'node_features': node_features,
        'labels': labels,
        'input_data': input_data,
        'x': x,
        'y': y,
        'time_closeness': time_closeness
    }, f)
