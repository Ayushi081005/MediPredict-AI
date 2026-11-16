
import pandas as pd

# Load dataset
data = pd.read_csv("dataset/dataset.csv")

# Identify symptom columns
symptom_columns = [
    col for col in data.columns if col.startswith("Symptom_")
]

# Remove extra spaces and standardize symptom names
for col in symptom_columns:
    data[col] = data[col].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else x
    )

# 1. Exact duplicate records
exact_duplicates = data.duplicated().sum()

print("\n========== DUPLICATE ANALYSIS ==========")

print("\nExact duplicate records:", exact_duplicates)

# 2. Convert each row into a sorted symptom combination
# Sorting ensures that the same symptoms in different
# column positions are treated as the same combination.

data["symptom_signature"] = data[symptom_columns].apply(
    lambda row: tuple(sorted(row.dropna().tolist())),
    axis=1
)

# 3. Count unique symptom combinations
unique_combinations = data["symptom_signature"].nunique()

print("\nUnique symptom combinations:", unique_combinations)

# 4. Find repeated symptom combinations
combination_counts = data["symptom_signature"].value_counts()

repeated_combinations = combination_counts[
    combination_counts > 1
]

print("\nRepeated symptom combinations:",
      len(repeated_combinations))

# 5. Check whether the same symptom combination
# is associated with multiple diseases

disease_counts = data.groupby(
    "symptom_signature"
)["Disease"].nunique()

conflicting_combinations = disease_counts[
    disease_counts > 1
]

print("\nConflicting symptom combinations:",
      len(conflicting_combinations))

# 6. Display examples of conflicting combinations
if len(conflicting_combinations) > 0:

    print("\nExamples of conflicting records:")

    conflict_data = data[
        data["symptom_signature"].isin(
            conflicting_combinations.index
        )
    ]

    print(
        conflict_data[
            ["Disease", "symptom_signature"]
        ].head(10).to_string(index=False)
    )

else:
    print("\nNo conflicting symptom combinations found.")

print("\n========== ANALYSIS COMPLETED ==========")
