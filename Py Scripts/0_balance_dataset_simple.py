# CREATING BALANCED DATASETS depsite highly unbalanced input data.

import pandas as pd
import numpy as np

print("Import Successful...")

print("Loading...")
# Read the original CSV file
df = pd.read_csv("/var/scratch/hwg580/HI-Large_Trans.csv") # 132 million

print("Separations...")
# Separate rows with "Is Laundering" equal to 0 and 1
df_0 = df[df["Is Laundering"] == 0]
df_1 = df[df["Is Laundering"] == 1]

print("Calculations...")
# Calculate the number of instances for the new CSV file
total_instances = 2000000
instances_0 = int(0.90 * total_instances)
instances_1 = total_instances - instances_0

print("Random selection...")
# Randomly select instances from each category
selected_indices_0 = np.random.choice(df_0.index, instances_0, replace=False)
selected_indices_1 = np.random.choice(df_1.index, instances_1, replace=False)

print("Creating balanced dataset...")
# Concatenate the selected rows
selected_df_0 = df_0.loc[selected_indices_0]
selected_df_1 = df_1.loc[selected_indices_1]
selected_df = pd.concat([selected_df_0, selected_df_1])

print("shuffle...")
# Shuffle the concatenated rows
selected_df = selected_df.sample(frac=1).reset_index(drop=True)
# Write the shuffled rows to a new CSV file

print("creating file...")
selected_df.to_csv("/var/scratch/hwg580/Balanced_HI-Large_Trans.csv", index=False)