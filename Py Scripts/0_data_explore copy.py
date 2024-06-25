import pickle

# Specify the file path where the data is saved
file_path = "/var/scratch/hwg580/graph_Balanced_HI-Large_Trans.pickle"

# Load the data from the pickle file
with open(file_path, "rb") as f:
    saved_data = pickle.load(f)

# Print the size of the input x
x_size = saved_data['x'].size()
print(f"Size of input x: {x_size}")

# You can print other information as needed
# For example, to print size of another key 'y'
# y_size = saved_data['y'].size()
# print(f"Size of input y: {y_size}")
