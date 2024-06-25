import pandas as pd
import numpy as np

print("Import Successful...")

print("Loading...")
# Load the data with Pandas
filename = '/var/scratch/hwg580/HI-Large_Trans.csv'
data = pd.read_csv(filename, parse_dates=['Timestamp'], infer_datetime_format=True)

print("Computing Statistics...")
# Extract unique accounts from both 'Account' and 'Account.1' columns
unique_accounts = pd.concat([data['Account'], data['Account.1']]).unique()
print(f"Number of unique accounts (Account and Account.1 combined): {len(unique_accounts)}")

# Compute statistics using Pandas operations
total_transactions = len(data)
fraudulent_transactions = data['Is Laundering'].sum()
fraud_percentage = (fraudulent_transactions / total_transactions) * 100

# Compute percentages of different currencies
currency_counts = data['Receiving Currency'].value_counts()
currency_percentage = (currency_counts / currency_counts.sum()) * 100

# Compute percentages of different payment formats
payment_format_counts = data['Payment Format'].value_counts()
payment_format_percentage = (payment_format_counts / payment_format_counts.sum()) * 100

# Print statistics
print(f"Total transactions: {total_transactions}")
print(f"Fraudulent transactions: {fraudulent_transactions}")
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
print("\nPercentage of different currencies:")
print(currency_percentage)
print("\nPercentage of different payment formats:")
print(payment_format_percentage)
