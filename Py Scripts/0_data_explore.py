import pandas as pd
import numpy as np

print("Import Successful...")
# input_file = "/var/scratch/hwg580/HI-Large_Trans.csv"
# output_file = "/var/scratch/hwg580/Reduced_HI-Large_Trans.csv"

print("Loading original CSV file...")
# Load the data with Pandas
filename = '/var/scratch/hwg580/Balanced_HI-Large_Trans.csv'
data = pd.read_csv(filename, parse_dates=['Timestamp'], infer_datetime_format=True)

# Step 3: Create a balanced dataset with 10% fraudulent transactions and rest non-fraudulent transactions, max 2 million transactions, and maintain connectivity

# Separate rows with "Is Laundering" equal to 0 and 1 from the reduced dataset
print("Separating fraudulent and non-fraudulent transactions...")
# df_reduced_0 = df_reduced[df_reduced["Is Laundering"] == 0]
# df_reduced_1 = df_reduced[df_reduced["Is Laundering"] == 1]

df_reduced_0 = data[data["Is Laundering"] == 0]
df_reduced_1 = data[data["Is Laundering"] == 1]

# Print the number of instances in each category
print(f"Number of non-fraudulent transactions: {len(df_reduced_0)}")
print(f"Number of fraudulent transactions: {len(df_reduced_1)}")
