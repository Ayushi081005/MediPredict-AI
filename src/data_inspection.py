
import pandas as pd
import os

# Dataset path
DATASET_PATH = "dataset"

# Load main dataset
data = pd.read_csv(
    os.path.join(DATASET_PATH, "dataset.csv")
)

print("\n========== DATASET INSPECTION ==========")

# Display first five records
print("\nFirst 5 Records:")
print(data.head())

# Dataset dimensions
print("\nDataset Shape:")
print(data.shape)

# Column names
print("\nColumn Names:")
print(data.columns.tolist())

# Missing values
print("\nMissing Values:")
print(data.isnull().sum().sum())

# Duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Disease distribution
print("\nDisease Distribution:")
print(data["Disease"].value_counts())

# Dataset information
print("\nDataset Information:")
data.info()

# Load supporting files
severity = pd.read_csv(
    os.path.join(DATASET_PATH, "Symptom-severity.csv")
)

description = pd.read_csv(
    os.path.join(DATASET_PATH, "symptom_Description.csv")
)

precaution = pd.read_csv(
    os.path.join(DATASET_PATH, "symptom_precaution.csv")
)

print("\nSupporting Dataset Shapes:")
print("Severity:", severity.shape)
print("Description:", description.shape)
print("Precaution:", precaution.shape)

print("\n========== INSPECTION COMPLETED ==========")
