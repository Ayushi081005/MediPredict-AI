
import pandas as pd

from pathlib import Path
from collections import defaultdict


# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

data = pd.read_csv(
    BASE_DIR / "dataset" / "dataset.csv"
)

symptom_columns = [
    col for col in data.columns
    if col.startswith("Symptom_")
]


# ==========================================
# 2. NORMALIZE SYMPTOMS
# ==========================================

def clean_symptoms(row):

    symptoms = []

    for value in row:

        if pd.notna(value):

            symptom = str(value).strip().lower()

            symptom = "_".join(symptom.split())

            if symptom:
                symptoms.append(symptom)

    return tuple(sorted(set(symptoms)))


data["Symptoms"] = data[symptom_columns].apply(
    clean_symptoms,
    axis=1
)


# ==========================================
# 3. REMOVE DUPLICATE COMBINATIONS
# ==========================================

data = data.drop_duplicates(
    subset="Symptoms"
).copy()


# ==========================================
# 4. MAP SYMPTOMS TO DISEASES
# ==========================================

symptom_to_diseases = defaultdict(set)

for _, row in data.iterrows():

    for symptom in row["Symptoms"]:

        symptom_to_diseases[symptom].add(
            row["Disease"]
        )


# ==========================================
# 5. ANALYZE SYMPTOM OVERLAP
# ==========================================

shared_symptoms = {
    symptom: diseases
    for symptom, diseases in symptom_to_diseases.items()
    if len(diseases) > 1
}


print("\n========== SYMPTOM PATTERN ANALYSIS ==========")

print("Unique Disease Classes:", data["Disease"].nunique())

print("Unique Symptom Names:", len(symptom_to_diseases))

print("Symptoms Shared Across Diseases:", len(shared_symptoms))

print("\n========== MOST SHARED SYMPTOMS ==========")

sorted_shared = sorted(
    shared_symptoms.items(),
    key=lambda item: len(item[1]),
    reverse=True
)

for symptom, diseases in sorted_shared[:20]:

    print(
        symptom,
        "->",
        len(diseases),
        "diseases"
    )


# ==========================================
# 6. CHECK EXACT SYMPTOM SET OVERLAP
# ==========================================

print("\n========== EXACT PATTERN CHECK ==========")

print(
    "Unique symptom combinations:",
    data["Symptoms"].nunique()
)

print(
    "Conflicting combinations:",
    data.groupby("Symptoms")["Disease"].nunique().gt(1).sum()
)

print("\nAnalysis completed!")
