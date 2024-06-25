import pandas as pd
import numpy as np

# Step 1: Reduce the size of the original CSV file to 30% and save it in the same folder

# Read the original CSV file
input_file = "/var/scratch/hwg580/HI-Large_Trans.csv"
output_file = "/var/scratch/hwg580/Reduced_HI-Large_Trans.csv"

print("Loading original CSV file...")
df = pd.read_csv(input_file)

# Reduce to 30% of original size
# print("Reducing the size of the dataset to 50%...")
# df_sampled = df.sample(frac=0.5, random_state=42)

# # Save reduced dataframe to a new CSV file
# print(f"Saving reduced CSV file to {output_file}...")
# df_sampled.to_csv(output_file, index=False)

# # Step 2: Load the just-saved reduced CSV file
# print(f"Loading reduced CSV file from {output_file}...")
# df_reduced = pd.read_csv(output_file)

# Step 3: Create a balanced dataset with 10% fraudulent transactions and rest non-fraudulent transactions, max 2 million transactions, and maintain connectivity

# Separate rows with "Is Laundering" equal to 0 and 1 from the reduced dataset
print("Separating fraudulent and non-fraudulent transactions...")
# df_reduced_0 = df_reduced[df_reduced["Is Laundering"] == 0]
# df_reduced_1 = df_reduced[df_reduced["Is Laundering"] == 1]

df_reduced_0 = df[df["Is Laundering"] == 0]
df_reduced_1 = df[df["Is Laundering"] == 1]

# Print the number of instances in each category
print(f"Number of non-fraudulent transactions: {len(df_reduced_0)}")
print(f"Number of fraudulent transactions: {len(df_reduced_1)}")

# Calculate the number of instances for the new balanced dataset
total_instances = min(500000, len(df))
instances_0 = int(0.6 * total_instances)
instances_1 = total_instances - instances_0

print(f"Creating a balanced dataset with {instances_0} non-fraudulent and {instances_1} fraudulent transactions...")

# Randomly select instances from each category
selected_indices_0 = np.random.choice(df_reduced_0.index, instances_0, replace=False)
selected_indices_1 = np.random.choice(df_reduced_1.index, instances_1, replace=False)

# Create balanced dataset
selected_df_0 = df_reduced_0.loc[selected_indices_0]
selected_df_1 = df_reduced_1.loc[selected_indices_1]
balanced_df = pd.concat([selected_df_0, selected_df_1])

# Shuffle the concatenated rows
print("Shuffling balanced dataset...")
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Write the balanced dataset to a new CSV file
balanced_output_file = "/var/scratch/hwg580/Balanced_HI-Large_Trans.csv"
print(f"Saving balanced dataset to {balanced_output_file}...")
balanced_df.to_csv(balanced_output_file, index=False)

print(f"Balanced dataset created and saved to {balanced_output_file}.")
