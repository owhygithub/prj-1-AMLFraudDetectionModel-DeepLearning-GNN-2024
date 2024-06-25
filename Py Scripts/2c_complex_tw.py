# 2c - COMPLEX - T+W

model_name = "ComplEx-T+W"

import numpy as np

import os
import csv
import math
import torch
import optuna
import pickle
import random
import datetime
from datetime import datetime
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# LOADING GRAPH from Jupyter Notebook 
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")

def calculate_mrr(sorted_indices, true_values):
    # Suppress the specific UserWarning
    # warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")
    
    true_values_tensor = torch.tensor(true_values, dtype=torch.float32).detach().requires_grad_(True)

    # Reset the warning filter to default after tensor creation
    # warnings.resetwarnings()

    # Find indices of true positive labels
    positive_indices = torch.nonzero(true_values_tensor).squeeze()

    if positive_indices.numel() == 0:
        return 0.0

    # Map indices in sorted_indices to their ranks
    rank_map = {}
    for rank, idx in enumerate(sorted_indices, start=1):
        rank_map[idx.item()] = rank

    reciprocal_ranks = []
    for idx in positive_indices:
        rank = rank_map.get(idx.item(), 0)
        if rank != 0:
            reciprocal_ranks.append(1.0 / rank)

    if len(reciprocal_ranks) == 0:
        return 0.0

    # Calculate the mean reciprocal rank
    mrr = torch.mean(torch.tensor(reciprocal_ranks, dtype=torch.float32))

    return mrr.item()  # Return MRR as a Python float

print("Started the program...")
# Specify the file path where the data is saved
file_path = "/var/scratch/hwg580/graph_Balanced_HI-Large_Trans.pickle"

# Load the data from the file
with open(file_path, "rb") as f:
    saved_data = pickle.load(f)

print("Loading data...")
# Now, you can access the saved data using the keys used during saving
dataset = saved_data['dataset']
edges_features = saved_data['edges_features']
links = saved_data['links']
labels = saved_data['labels']
graph_full = saved_data['graph_full']
adjacency_matrix = saved_data['adjacency_matrix']
visual = saved_data['visual']
node_features = saved_data['node_features']
x = saved_data['x']
y = saved_data['y']
input_data = saved_data['input_data']
time_closeness = saved_data['time_closeness']
adjacency_tensor = saved_data['adjacency_tensor']

print("Splitting data...")
# Split the nodes into training, validation, and test sets
num_edges = edges_features.shape[0]
print(f"Number of edge: {num_edges}")
indices = list(range(num_edges))
train_indices, test_val_indices = train_test_split(indices, test_size=0.4, stratify=labels)
val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, stratify=labels[test_val_indices])

print("Creating mask data...")
# Convert indices to a tensor
indices = torch.arange(num_edges)

# Create masks efficiently using torch.zeros_like and indexing
train_mask = torch.zeros_like(indices, dtype=torch.bool)
val_mask = torch.zeros_like(indices, dtype=torch.bool)
test_mask = torch.zeros_like(indices, dtype=torch.bool)

# Set True at indices present in train_indices, val_indices, test_indices
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True
print("Mask data created...")

# Convert time_closeness to a tensor
time_closeness_tensor = torch.tensor(time_closeness, dtype=torch.float32)
train_time_closeness = time_closeness_tensor[train_mask]
val_time_closeness = time_closeness_tensor[val_mask]
test_time_closeness = time_closeness_tensor[test_mask]

print(train_time_closeness.size())
print(val_time_closeness.size())
print(test_time_closeness.size())


# Define your GNNLayer class
class GNNLayer(MessagePassing):
    def __init__(self, node_features, edge_features, out_channels, dropout):
        super(GNNLayer, self).__init__(aggr='add')
        self.node_features = node_features
        self.edge_features = edge_features
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        
        # Learnable parameters
        self.weight_node = nn.Parameter(torch.Tensor(node_features, out_channels))
        self.weight_edge = nn.Parameter(torch.Tensor(edge_features, out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_node)
        nn.init.xavier_uniform_(self.weight_edge)
        
    def forward(self, x, edge_index, edge_attr):
        # AXW0 + EW1
        global adjacency_tensor
        self.adjacency_matrix = adjacency_tensor
        
        axw = torch.sparse.mm(self.adjacency_matrix, x) @ self.weight_node
        ew = torch.matmul(edge_attr, self.weight_edge)

        axw = self.dropout(axw)  # Apply dropout to node features
        ew = self.dropout(ew)    # Apply dropout to edge features

        return axw, ew

    def update(self, aggr_out):
        return aggr_out
    
# Define your GNNModel class
class GNNModel(nn.Module):
    def __init__(self, node_features, edge_features, out_channels, dropout):
        super(GNNModel, self).__init__()
        self.conv1 = GNNLayer(node_features, edge_features, out_channels, dropout)

        # Learnable parameters
        self.learnable_weight = nn.Parameter(torch.Tensor(1))
        print(self.learnable_weight.size())
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.learnable_weight, mean=0, std=0.1)  # Adjust std as needed # FIX

    def forward(self, x, edge_index, edge_attr, time_closeness_tensor):
        axw1, ew1 = self.conv1(x, edge_index, edge_attr)

        head_indices, tail_indices = self.mapping(ew1, edge_index)
        scores = self.complex(axw1, ew1, head_indices, tail_indices, time_closeness_tensor) # add the timestamp
        
        return axw1, ew1, scores # returning x and e embeddings

    def update_edge_attr(self, edge_attr, new_channels):
        num_edge_features = edge_attr.size(1)
        if new_channels > num_edge_features:
            updated_edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.size(0), new_channels - num_edge_features), device=edge_attr.device)), dim=1)
        else:
            updated_edge_attr = edge_attr[:, :new_channels]
        return updated_edge_attr
    
    def complex(self, axw, ew, head_indices, tail_indices, time_closeness_tensor):
        # Convert time_closeness to a tensor (assuming it's already a list)
        # time_closeness_tensor = torch.tensor(time_closeness, dtype=torch.float32, device=axw.device)
        learnable_weight_tensor = torch.tensor(self.learnable_weight, dtype=torch.float32, device=axw.device)

        # Gather node embeddings and edge features for head and tail nodes
        heads = axw[head_indices]
        tails = axw[tail_indices]

        # Calculate element-wise product of head, tail, and edge embeddings
        # element_wise_product = head_embeddings * ew * tail_embeddings

        # Convert real tensors to complex tensors
        heads = torch.complex(heads, torch.zeros_like(heads))
        ew = torch.complex(ew, torch.zeros_like(ew))
        tails = torch.complex(tails, torch.zeros_like(tails))

        raw_scores = torch.real(torch.sum(heads * ew * torch.conj(tails), dim=-1)) * (time_closeness_tensor*learnable_weight_tensor)

        # Apply sigmoid activation
        normalized_scores = torch.sigmoid(raw_scores)

        return normalized_scores
    
    def mapping(self, ew, edge_index):
        head_indices = edge_index[0]
        tail_indices = edge_index[1]
        return head_indices, tail_indices
    
# # Get Hyperparams
# file_path = "/var/scratch/hwg580/complex_hyperparams.pickle"

# # Load the data from the file
# with open(file_path, "rb") as f:
#     saved_data = pickle.load(f)

# print("Hyperparameter Loading...")
# # Now, you can access the saved data using the keys used during saving
# best_epochs = saved_data['best_epochs']
# best_lr = saved_data['best_lr']
# best_out_channels = saved_data['best_out_channels']
# best_weight_decay = saved_data['best_weight_decay']
# best_dropout = saved_data['best_dropout']
# best_annealing_rate = saved_data['best_annealing_rate']
# annealing_epochs = saved_data['annealing_epochs']

# Hyperparams --- adjust to model best hyperparams
learning_rate = 0.01
out_channels = 20
weight_decay = 0.00005  # L2 regularization factor
epochs = 100
dropout = 0.1 # dropout probability

# Annealing parameters
annealing_rate = 0.001  # Rate at which to decrease the learning rate
annealing_epochs = 10  # Number of epochs before decreasing learning rate

print("Loading Model...")
model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
def train(data):
    model.train()
    optimizer.zero_grad()
    x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, train_mask], data.edge_attr[train_mask])

    loss = criterion(scores, labels[train_mask].float())
    loss.backward()
    optimizer.step()
    
    return loss.item(), x_embedding, e_embedding, scores

# Validation function
def validate(data):
    model.eval()
    with torch.no_grad():
        x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, val_mask], data.edge_attr[val_mask])
        val_loss = criterion(scores, labels[val_mask].float()).item()
    return x_embedding, e_embedding, scores, val_loss

# Test function
def test(data, test_time_closeness):
    model.eval()
    with torch.no_grad():
        x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, test_mask], data.edge_attr[test_mask], test_time_closeness)
        test_loss = criterion(scores, labels[test_mask].float()).item()
    return x_embedding, e_embedding, scores, test_loss

def assign_top_n_predictions(val_scores, val_labels):
    # Sort indices of val_scores in descending order
    sorted_indices = torch.argsort(val_scores, descending=True)

    # Determine the number of top predictions to assign as 1
    num_ones = int(torch.sum(val_labels).item())

    # Create a tensor to store the predicted labels (initialized with all zeros)
    predicted_labels = torch.zeros_like(val_labels)

    # Assign the top n predictions as 1
    predicted_labels[sorted_indices[:num_ones]] = 1

    return predicted_labels, sorted_indices

def assign_predictions(val_scores, threshold=0.6):
    # Assign labels based on a threshold
    predicted_labels = (val_scores >= threshold).float()
    return predicted_labels

# Initialize lists for storing fold-wise metrics
# fold_accuracy_list = []
# fold_precision_list = []
# fold_recall_list = []
# fold_f1_list = []
# fold_mrr_list = []

# k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# best_model_state = None
# best_recall = -1.0
# best_fold = -1
# best_train_losses = []
# best_val_losses = []
# best_epoch_metrics = {
#     'accuracy': [],
#     'precision': [],
#     'recall': [],
#     'f1': [],
#     'mrr': []
# }

# # Initialize variables for average metrics across all folds
# avg_accuracy = 0.0
# avg_precision = 0.0
# avg_recall = 0.0
# avg_f1 = 0.0
# avg_mrr = 0.0

# for fold, (train_fold_indices, val_fold_indices) in enumerate(kf.split(range(input_data.edge_attr.shape[0]))):
#     print(f"Fold {fold+1}/{k}")
#     train_fold_mask = torch.zeros(input_data.edge_attr.shape[0], dtype=torch.bool)
#     val_fold_mask = torch.zeros(input_data.edge_attr.shape[0], dtype=torch.bool)

#     train_fold_mask[train_fold_indices] = True
#     val_fold_mask[val_fold_indices] = True

#     model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     criterion = nn.BCEWithLogitsLoss()

#     train_losses = []
#     val_losses = []
#     accuracy_per_epoch = []
#     precision_per_epoch = []
#     recall_per_epoch = []
#     f1_per_epoch = []
#     mrr_per_epoch = []

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         x_embedding, e_embedding, scores = model(input_data.x, input_data.edge_index[:, train_fold_mask], input_data.edge_attr[train_fold_mask], time_closeness_tensor[train_fold_mask])
#         loss = criterion(scores, labels[train_fold_mask].float())
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_x_embedding, val_e_embedding, val_scores = model(input_data.x, input_data.edge_index[:, val_fold_mask], input_data.edge_attr[val_fold_mask], time_closeness_tensor[val_fold_mask])
#             val_loss = criterion(val_scores, labels[val_fold_mask].float()).item()

#         train_losses.append(loss.item())
#         val_losses.append(val_loss)

#         val_labels = labels[val_fold_mask]
#         val_predictions = assign_predictions(val_scores)

#         sorted_indices = torch.argsort(val_scores, descending=True)

#         val_accuracy = accuracy_score(val_labels, val_predictions)
#         val_precision = precision_score(val_labels, val_predictions)
#         val_recall = recall_score(val_labels, val_predictions)
#         val_f1 = f1_score(val_labels, val_predictions)
#         val_mrr = calculate_mrr(torch.argsort(val_scores, descending=True), val_labels)

#         accuracy_per_epoch.append(val_accuracy)
#         precision_per_epoch.append(val_precision)
#         recall_per_epoch.append(val_recall)
#         f1_per_epoch.append(val_f1)
#         mrr_per_epoch.append(val_mrr)

#         print(f"\nEpoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
#         print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MRR: {val_mrr}")

#     best_epoch_metrics['accuracy'].append(accuracy_per_epoch)
#     best_epoch_metrics['precision'].append(precision_per_epoch)
#     best_epoch_metrics['recall'].append(recall_per_epoch)
#     best_epoch_metrics['f1'].append(f1_per_epoch)
#     best_epoch_metrics['mrr'].append(mrr_per_epoch)

#     best_train_losses.append(train_losses)
#     best_val_losses.append(val_losses)

#     current_fold_best_metrics = {
#         'accuracy': max(accuracy_per_epoch),
#         'precision': max(precision_per_epoch),
#         'recall': max(recall_per_epoch),
#         'f1': max(f1_per_epoch),
#         'mrr': max(mrr_per_epoch),
#         'val_predictions': val_predictions,
#         'sorted_indices': sorted_indices
#     }

#     if current_fold_best_metrics['recall'] > best_recall:
#         best_recall = current_fold_best_metrics['recall']
#         best_fold = fold
#         best_model_state = model.state_dict()

#     fold_accuracy_list.append(current_fold_best_metrics['accuracy'])
#     fold_precision_list.append(current_fold_best_metrics['precision'])
#     fold_recall_list.append(current_fold_best_metrics['recall'])
#     fold_f1_list.append(current_fold_best_metrics['f1'])
#     fold_mrr_list.append(current_fold_best_metrics['mrr'])

# avg_accuracy = sum(fold_accuracy_list) / k
# avg_precision = sum(fold_precision_list) / k
# avg_recall = sum(fold_recall_list) / k
# avg_f1 = sum(fold_f1_list) / k
# avg_mrr = sum(fold_mrr_list) / k

# print("\nAverage Metrics Across All Folds:")
# print(f"Avg Accuracy: {avg_accuracy:.4f}")
# print(f"Avg Precision: {avg_precision:.4f}")
# print(f"Avg Recall: {avg_recall:.4f}")
# print(f"Avg F1 Score: {avg_f1:.4f}")
# print(f"Avg MRR: {avg_mrr:.4f}")

# if best_fold >= 0:
#     print("\nBest Model Metrics:")
#     print(f"Best Accuracy: {max(best_epoch_metrics['accuracy'][best_fold]):.4f}")
#     print(f"Best Precision: {max(best_epoch_metrics['precision'][best_fold]):.4f}")
#     print(f"Best Recall: {max(best_epoch_metrics['recall'][best_fold]):.4f}")
#     print(f"Best F1 Score: {max(best_epoch_metrics['f1'][best_fold]):.4f}")
#     print(f"Best MRR: {max(best_epoch_metrics['mrr'][best_fold]):.4f}")

# if best_model_state is not None:
#     torch.save(best_model_state, f'/var/scratch/hwg580/{model_name}_best.pt')
#     print(f"\nBest model saved with highest recall: {best_recall:.4f}")

# results_folder = f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/Validation'
# os.makedirs(results_folder, exist_ok=True)

# epoch_numbers = list(range(1, len(best_train_losses[best_fold]) + 1))

# plt.figure(figsize=(10, 6))
# plt.plot(epoch_numbers, best_train_losses[best_fold], label="Training Loss")
# plt.plot(epoch_numbers, best_val_losses[best_fold], label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Losses")
# plt.title("Training and Validation Losses Over Epochs (Best Fold)")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(results_folder, 'training_validation_losses.png'))
# plt.show()

# plt.figure(figsize=(14, 10))

# plt.subplot(2, 2, 1)
# plt.plot(epoch_numbers, best_epoch_metrics['accuracy'][best_fold], label="Accuracy")
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(epoch_numbers, best_epoch_metrics['precision'][best_fold], label="Precision")
# plt.xlabel('Epoch')
# plt.ylabel('Precision')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(epoch_numbers, best_epoch_metrics['recall'][best_fold], label="Recall")
# plt.xlabel('Epoch')
# plt.ylabel('Recall')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(epoch_numbers, best_epoch_metrics['f1'][best_fold], label="F1 Score")
# plt.xlabel('Epoch')
# plt.ylabel('F1 Score')
# plt.legend()

# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, 'evaluation_metrics.png'))
# plt.show()


import now 
def evaluate_model(predictions, true_values, sorted_indices, mask, model_name):
    # Define the results folder path
    results_folder = f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/Testing'
    os.makedirs(results_folder, exist_ok=True)
    
    true_values = true_values[mask].float()

    # Convert tensors to numpy arrays
    predictions = predictions.cpu().numpy()
    true_values = true_values.cpu().numpy()
    
    mrr = calculate_mrr(sorted_indices, true_values) # TODO --> this is just the accuracy now ... 
    print(f"MRR = {mrr}")
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions)
    recall = recall_score(true_values, predictions)
    f1 = f1_score(true_values, predictions)
    cm = confusion_matrix(true_values, predictions)
    classification_rep = classification_report(true_values, predictions)
    
    # ROC Curve and AUC if applicable
    try:
        fpr, tpr, thresholds = roc_curve(true_values, predictions)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        fpr, tpr, roc_auc = None, None, None
    
    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "Classification Report": classification_rep,
        "ROC Curve": (fpr, tpr, roc_auc)
    }

    # Get current datetime for filenames
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(results_folder, f'roc_curve_{current_datetime}.png')
    plt.savefig(roc_curve_path)
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(results_folder, f'confusion_matrix_{current_datetime}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    return metrics_dict


best_model_state = torch.load(f'/var/scratch/hwg580/{model_name}_best.pt')
model.load_state_dict(best_model_state)

print("Testing...")
# TEST DATA
test_x_embedding, test_e_embedding, test_scores, test_loss = test(input_data, test_time_closeness)
print(f"Test Loss: {test_loss:.4f}")

test_labels = labels[test_mask]
test_predictions = assign_predictions(test_scores)

sorted_indices = torch.argsort(test_scores, descending=True)

# Calculate evaluation metrics
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
test_recall = recall_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)

test_mrr = calculate_mrr(sorted_indices, test_labels)

print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")
print(f"This is the MRR testing data: {test_mrr}")
metrics_dict = evaluate_model(test_predictions, labels, sorted_indices, test_mask, model_name)

# Print Evaluation Metrics
print("Evaluation Metrics:")
print("-------------------\n")
for metric_name, metric_value in metrics_dict.items():
    if metric_name == "Confusion Matrix":
        print("Confusion Matrix:")
        print(metric_value)
    elif metric_name == "Classification Report":
        print("Classification Report:")
        print(metric_value)
    elif metric_name == "ROC Curve":
        fpr, tpr, roc_auc = metric_value
        print("ROC Curve:")
        print("- False Positive Rate:", fpr)
        print("- True Positive Rate:", tpr)
        print("- AUC:", roc_auc)
    else:
        print(f"{metric_name}: {metric_value}")
    print()

# LOGGING
# Function to log the experiment
def log_experiment(model_name, learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr, roc_auc):
    # Create a folder for the experiment if it doesn't exist
    folder_name = f"/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save metrics and other information to a file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{folder_name}/run_{timestamp}.txt"
    with open(file_name, "w") as f:
        f.write(f"-- HYPERPARAMS:\n")
        f.write(f"TimeStamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"out_channels: {out_channels}\n")
        f.write(f"Epoch: {epoch}\n\n")
        f.write(f"Weight_decay: {weight_decay}\n\n")
        f.write(f"Dropout: {dropout}\n\n")
        f.write(f"-- RESULTS:\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"MRR: {mrr}\n")
        f.write(f"ROC-AUC: {roc_auc}\n")
    
    # Update the general CSV file
    print("Updating csv file...")
    csv_file = f"/home/hwg580/thesis/AML-fraud-detector/general.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Timestamp", "learning_rate", "out_channels", "Epoch", "Weight_decay", "Dropout", "Loss", "Accuracy", "Precision", "Recall", "F1 Score", "MRR", "ROC-AUC"])
        writer.writerow([model_name, timestamp,  learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr, roc_auc])

# Inside the training loop, after each epoch:
# Log the experiment
print("Logging...")
log_experiment(model_name=model_name, learning_rate=learning_rate, out_channels=out_channels, epoch=epochs, weight_decay=weight_decay, dropout=dropout, loss=test_loss, accuracy=test_accuracy, precision=test_precision, recall=test_recall, f1=test_f1, mrr=test_mrr, roc_auc=roc_auc)

# PYTORCH.save --> save the tensor for predictions for the graph
torch.save({'test_labels': test_labels}, f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/labels.pt')
torch.save({'predictions': test_predictions}, f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/predictions_{model_name}_{random.randint(1, 100)}.pt')

# Save in RDF format
# import gzip
# import torch

# def save_embeddings_as_triples(edge_index, node_embeddings, edge_embeddings, file_path):
#     with gzip.open(file_path, 'wt') as f:
#         for i in range(edge_index.size(1)):
#             head_index = edge_index[0, i].item()
#             tail_index = edge_index[1, i].item()
#             head_uri = f"http://example.org/node/{head_index}"
#             tail_uri = f"http://example.org/node/{tail_index}"
#             relation_uri = f"http://example.org/relation/{i}"

#             # Write connectedTo triple
#             f.write(f'<{head_uri}> <http://example.org/ontology#connectedTo> <{tail_uri}>.\n')
            
#             # Write head node embedding triple
#             head_embedding = node_embeddings[head_index].numpy()
#             head_str = ' '.join(map(str, head_embedding))
#             f.write(f'<{head_uri}> <http://example.org/ontology#hasEmbedding> "{head_str}."\n')
            
#             # Write tail node embedding triple
#             tail_embedding = node_embeddings[tail_index].numpy()
#             tail_str = ' '.join(map(str, tail_embedding))
#             f.write(f'<{tail_uri}> <http://example.org/ontology#hasEmbedding> "{tail_str}".\n')
            
#             # Write edge embedding triple
#             relation_embedding = edge_embeddings[i].numpy()
#             relation_str = ' '.join(map(str, relation_embedding))
#             f.write(f'<{relation_uri}> <http://example.org/ontology#hasEmbedding> "{relation_str}".\n')

# edge_index_train_triples = input_data.edge_index[:, train_mask]

# # Example usage after training
# x_embeddings = torch.tensor(all_x_embeddings[-1])  # Use the last epoch's embeddings
# e_embeddings = torch.tensor(all_e_embeddings[-1])  # Use the last epoch's embeddings

# save_embeddings_as_triples(edge_index_train_triples, x_embeddings, e_embeddings, f"Saved-Data/rdf_triples_{model_name}.nt.gz")
