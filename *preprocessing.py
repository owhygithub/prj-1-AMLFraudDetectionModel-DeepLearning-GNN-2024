import torch
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from functions import *

class AMLDataPreprocessing:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.labels = None
        self.node_features = None
        self.edges_features = None
        self.links = None
        self.graph_full = None
        self.adjacency_matrix = None
        self.edge_index = None
        self.input_data = None

    def process_data(self):
        # BASICS
        self.data = pd.read_csv(self.filename)
        laundering_accounts = list(self.data[self.data["Is Laundering"]==1]["Account"])
        labels = self.data["Is Laundering"].to_numpy()
        merged_unique_accounts = pd.concat([self.data["Account"], self.data["Account.1"]]).unique()

        print(f"Shape of DataFrame - {self.data.shape}")
        print("---- info ----")
        print(self.data.info())
        print("---- basic calculations ----")
        print(self.data.describe())
        print("Null elements?")
        print(self.data.isnull().sum())
        print("Fraudulent or Not? - Y labels")
        print(f"Number of fraudulent transactions - {len(self.data[self.data['Is Laundering']==1])}")
        print(f"Number of non-fraudulent transactions - {len(self.data[self.data['Is Laundering']==0])}")
        print(f"Laundering Accounts - {laundering_accounts}")
        print("Are Amount Paid entirely equal to Amount Received?\n - " + str(self.data["Amount Paid"].equals(self.data["Amount Received"])))
        print("Are Currency Received entirely equal to Currency Paid?\n - " + str(self.data["Payment Currency"].equals(self.data["Receiving Currency"])))
        print(sorted(self.data["Receiving Currency"].unique()))
        print(sorted(self.data["Payment Currency"].unique()))
        print(sorted(self.data["Payment Format"].unique()))
        print(f"Number of unique accounts: {len(merged_unique_accounts)}")

        # NODE MATRIX
        unique_accounts = get_nodes(self.data)
        node_features = one_hot_encoding(unique_accounts, column="Currency")
        from_bank_col = node_features.pop('Bank')
        account_col = node_features.pop('Accounts')
        node_labels = pd.DataFrame(account_col)
        df = pd.DataFrame(account_col, columns=['Accounts'])
        df.reset_index(drop=True, inplace=True) # Ensure the DataFrame has the same number of rows as the original series
        vectors = hashing_vectorization(df['Accounts'], vector_size=9)
        # Convert vectors into DataFrame
        vectors_df = pd.DataFrame(vectors, columns=[f'col_{i}' for i in range(len(vectors[0]))])
        result_df = pd.concat([df, vectors_df], axis=1)
        accounts_df = result_df.drop(columns=["Accounts"])
        from_bank_binary = [bin(x).split("b")[1] for x in from_bank_col]
        binary_lists = [[int(bit) for bit in binary] for binary in from_bank_binary]
        longest_str, len_longest_str = get_longest_string_in_list(from_bank_binary)
        binary_lists = make_binary_fixed_length(binary_lists, longest_str)
        bin_vectors_df = pd.DataFrame(binary_lists, columns=[f'bin_{i}' for i in range(len(binary_lists[0]))])
        from_bank_df = pd.DataFrame(from_bank_col)
        accounts_df = pd.DataFrame(accounts_df)
        accounts_df_norm = normalize(accounts_df,0,1)
        node_features.reset_index(drop=True, inplace=True) # Ensure the DataFrame has the same number of rows as the original series
        accounts_df_norm.reset_index(drop=True, inplace=True) # Ensure the DataFrame has the same number of rows as the original series
        node_features = pd.concat([node_features, accounts_df_norm], axis=1)
        node_features = pd.concat([node_features, bin_vectors_df], axis=1)
        unique_ids_set = set()
        while len(unique_ids_set) < len(node_features): # uniqueness kept
            unique_ids_set.add(random.random())
        unique_ids = list(unique_ids_set)
        node_features.insert(0, "Unique ID", unique_ids)
        x = node_features.to_numpy()

        # EDGE MATRIX
        links = [{'source': source, 'destination': destination} for source, destination in zip(self.data['Account'], self.data['Account.1'])]
        edges_df = self.data[["Timestamp", "Amount Paid", "Payment Currency", "Payment Format"]]
        edges_amount = edges_df["Amount Paid"].astype(str)
        edges_amount = list(edges_amount)
        maximum = str(max(edges_df["Amount Paid"]))
        max_len = len(maximum.split(".")[0])
        minimum = min(edges_df["Amount Paid"])
        minimum = format(minimum, 'f')
        min_len = len(str(minimum.split('.')[1]))
        new_min, count = count_unused_decimals(minimum)
        min_len = min_len - count
        number_columns = max_len + min_len
        number_columns
        a = split_into_vectors(edges_amount) # INEFFICIENT !!!! # INEFFICIENT !!!!# INEFFICIENT !!!!# INEFFICIENT !!!!# INEFFICIENT !!!!
        new_payment_list = encode_payment_amount(a, max_len, min_len) # INEFFICIENT !!!! # INEFFICIENT !!!!# INEFFICIENT !!!!# INEFFICIENT !!!!# INEFFICIENT !!!!
        new_payment_list = nested_list_int = [[int(item) for item in sublist] for sublist in new_payment_list]
        payment_vectors_df = pd.DataFrame(new_payment_list, columns=[f'payment_{i}' for i in range(len(new_payment_list[0]))])
        edges_features = pd.concat([edges_df, payment_vectors_df], axis=1)
        edges_features.drop("Amount Paid", axis='columns')
        # TODO USE GENERAL FUNCTION FOR ONE-HOT ENCODING
        positions = edges_features["Payment Currency"].str.split(",", expand=True) # creating new columns by splitting receiving currency --> all are added
        edges_features["first_position"] = positions[0] # first currency in each row is extracted --> actual currency used and that we want as TRUE
        edges_features = pd.concat([edges_features, pd.get_dummies(edges_features["first_position"],dtype='int')], axis=1, join='inner') # effectively adds actual currency to dummy variables/columns
        edges_features.drop(["Amount Paid","Payment Currency", "first_position"], axis=1, inplace=True) # drop the axiliary columns
        edges_features.head()
        # DONE convert Payment Format
        positions_2 = edges_features["Payment Format"].str.split(",", expand=True)
        edges_features["second_position"] = positions_2[0]
        edges_features = pd.concat([edges_features, pd.get_dummies(edges_features["second_position"],dtype='int')], axis=1, join='inner') # effectively adds actual currency to dummy variables/columns
        edges_features.drop(["Payment Format", "second_position"], axis=1, inplace=True) # drop the axiliary columns
        edges_features.head()
        edges_features["Timestamp"] = pd.to_datetime(edges_features['Timestamp']).astype(int) // 10**9 # does not interpret time well... circular definition for months --> sinus calculations
        edges_features.head()
        y = edges_features.to_numpy()

        # CREATE GRAPH:
        graph_full = create_graph(links, edges_amount)

        adjacency_matrix = nx.adjacency_matrix(graph_full)
        adjacency_matrix

        accounts = unique_accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Accounts'], accounts['ID']))
        self.data['From'] = self.data['Account'].map(mapping_dict)
        self.data['To'] = self.data['Account.1'].map(mapping_dict)
        self.data = self.data.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)

        edge_index = torch.stack([torch.from_numpy(self.data['From'].values), torch.from_numpy(self.data['To'].values)], dim=0)
        adjacency_matrix = adjacency_matrix.todense()
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to(torch.float)
        num_ones = (adjacency_matrix == 1).sum().item()
        node_features = node_features.to_numpy()
        edges_features = edges_features.to_numpy()
        node_features = torch.from_numpy(node_features).to(torch.float)
        edges_features = torch.from_numpy(edges_features).to(torch.float)
        labels = torch.from_numpy(labels).to(torch.float)
        input_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edges_features,
            y=labels
        )

        with open("graph_data.pickle", "wb") as f:
            pickle.dump({
                'edges_features': edges_features,
                'links': links,
                'unique_accounts': unique_accounts,
                'graph_full': graph_full,
                'adjacency_matrix': adjacency_matrix,
                'node_features': node_features,
                'edge_index': edge_index,
                'labels': labels,
                'input_data': input_data
            }, f)

        return input_data, graph_full, x, y, labels, links, edges_amount

    def visualize_graph(self, links, edges_amount, limit=150):
        # DONE Creating smaller graph for visualization:
        # limit = 150
        small_graph = create_graph(links, edges_amount, limit=limit)
        pos = nx.random_layout(small_graph) # shell, circular, spectral, spring, random,
        plt.figure(figsize=(25, 15))  # Increase figure size

        nx.draw(
            small_graph,
            pos,
            node_size=300,  # Reduce node size for better visibility
            with_labels=True,
            font_size=7,
            font_weight='bold',
            node_color='lightblue',  # Specify node color
            edge_color='gray',  # Specify edge color
            width=1,  # Adjust edge width
            arrows=True,  # Show arrows for directed edges
            arrowstyle='->',  # Specify arrow style
            arrowsize=20,  # Adjust arrow size
        )

        edge_labels = nx.get_edge_attributes(small_graph, 'label')
        nx.draw_networkx_edge_labels(
            small_graph,
            pos,
            edge_labels=edge_labels,
            label_pos=0.5,  # Adjust label position along edges
            font_size=7,  # Adjust font size
            font_color='green',  # Specify font color
        )

        if 'limit' in locals():
            plt.title(f'Graph Visualization of first {limit} transactions')  # Add title to the plot
        else:
            plt.title(f'Graph Visualization of all transactions')  # Add title to the plot
        plt.axis('off')  # Hide axis
        plt.show()