
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# ==========================================
# 1. DEFINE PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports"

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

data = pd.read_csv(DATASET_PATH)

print("\n========== DATA PREPROCESSING ==========")

print("\nOriginal Dataset Shape:", data.shape)


# ==========================================
# 3. IDENTIFY SYMPTOM COLUMNS
# ==========================================

symptom_columns = [
    col for col in data.columns
    if col.startswith("Symptom_")
]


# ==========================================
# 4. NORMALIZE SYMPTOMS
# ==========================================

def clean_symptoms(row):

    symptoms = []

    for value in row:

        if pd.notna(value):

            symptom = str(value).strip().lower()

            symptom = "_".join(symptom.split())

            if symptom:
                symptoms.append(symptom)

    # Remove repeated symptoms within a record
    return sorted(set(symptoms))


data["Symptoms"] = data[symptom_columns].apply(
    clean_symptoms,
    axis=1
)


# ==========================================
# 5. REMOVE DUPLICATE COMBINATIONS
# ==========================================

data["Symptom_Signature"] = data["Symptoms"].apply(
    tuple
)

unique_data = data.drop_duplicates(
    subset="Symptom_Signature"
).copy()

print("\nAfter Duplicate Removal:")
print("Unique Records:", len(unique_data))


# ==========================================
# 6. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nDisease Distribution:")

print(
    unique_data["Disease"].value_counts()
)

minimum_class_count = (
    unique_data["Disease"].value_counts().min()
)

if minimum_class_count < 2:
    raise ValueError(
        "Some disease classes have fewer than "
        "2 unique records. Stratified splitting "
        "needs further adjustment."
    )


# ==========================================
# 7. BINARY SYMPTOM ENCODING
# ==========================================

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(
    unique_data["Symptoms"]
)

X = pd.DataFrame(
    X,
    columns=mlb.classes_,
    index=unique_data.index
)

y = unique_data["Disease"].reset_index(drop=True)

X = X.reset_index(drop=True)


print("\n========== ENCODING RESULTS ==========")

print("Number of Unique Symptoms:", X.shape[1])

print("Feature Matrix Shape:", X.shape)

print("Target Shape:", y.shape)


# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== SPLIT RESULTS ==========")

print("Training Features:", X_train.shape)

print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)

print("Testing Labels:", y_test.shape)


# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================

X_train.to_csv(
    REPORT_PATH / "X_train.csv",
    index=False
)

X_test.to_csv(
    REPORT_PATH / "X_test.csv",
    index=False
)

y_train.to_csv(
    REPORT_PATH / "y_train.csv",
    index=False
)

y_test.to_csv(
    REPORT_PATH / "y_test.csv",
    index=False
)


# Save symptom encoder for future predictions
joblib.dump(
    mlb,
    MODEL_PATH / "symptom_binarizer.pkl"
)


# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING COMPLETED ==========")

print("Processed training and testing data saved.")

print("Symptom encoder saved successfully.")

print("\nNext Step: Machine Learning Model Training")
