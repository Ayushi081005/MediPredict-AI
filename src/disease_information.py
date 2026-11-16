
import pandas as pd

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "dataset"


# ==========================================
# 2. LOAD SUPPORTING DATA
# ==========================================

description_data = pd.read_csv(
    DATASET_PATH / "symptom_Description.csv"
)

precaution_data = pd.read_csv(
    DATASET_PATH / "symptom_precaution.csv"
)


# ==========================================
# 3. NORMALIZE DISEASE NAMES
# ==========================================

def normalize_disease(name):

    return str(name).strip().lower()


description_data["Disease_Key"] = (
    description_data["Disease"].apply(normalize_disease)
)

precaution_data["Disease_Key"] = (
    precaution_data["Disease"].apply(normalize_disease)
)


# ==========================================
# 4. GET DISEASE DESCRIPTION
# ==========================================

def get_description(disease):

    disease_key = normalize_disease(disease)

    result = description_data[
        description_data["Disease_Key"] == disease_key
    ]

    if result.empty:
        return "Description is not available."

    return result.iloc[0]["Description"]


# ==========================================
# 5. GET DATASET PRECAUTIONS
# ==========================================

def get_precautions(disease):

    disease_key = normalize_disease(disease)

    result = precaution_data[
        precaution_data["Disease_Key"] == disease_key
    ]

    if result.empty:
        return []

    precaution_columns = [
        col for col in precaution_data.columns
        if col.startswith("Precaution_")
    ]

    precautions = []

    for column in precaution_columns:

        value = result.iloc[0][column]

        if pd.notna(value) and str(value).strip():

            precautions.append(str(value).strip())

    return precautions


# ==========================================
# 6. TEST INFORMATION MODULE
# ==========================================

if __name__ == "__main__":

    disease = "Fungal infection"

    print("\n========== DISEASE INFORMATION ==========")

    print("\nDisease:", disease)

    print("\nDescription:")

    print(get_description(disease))

    print("\nDataset-provided precaution information:")

    for index, precaution in enumerate(
        get_precautions(disease),
        start=1
    ):

        print(f"{index}. {precaution}")
