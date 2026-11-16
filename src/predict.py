
import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models"

REPORT_PATH = BASE_DIR / "reports"


# ==========================================
# 2. LOAD TRAINED MODEL AND ENCODER
# ==========================================

model = joblib.load(
    MODEL_PATH / "best_model.pkl"
)

encoder = joblib.load(
    MODEL_PATH / "symptom_binarizer.pkl"
)

# Recover the exact training feature order
feature_names = pd.read_csv(
    REPORT_PATH / "X_train.csv",
    nrows=0
).columns.tolist()


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict_disease(selected_symptoms):

    if not selected_symptoms:

        raise ValueError(
            "Please select at least one symptom."
        )

    # Normalize user inputs
    selected_symptoms = [
        symptom.strip().lower().replace(" ", "_")
        for symptom in selected_symptoms
    ]

    # Check for invalid symptoms
    available_symptoms = set(encoder.classes_)

    invalid_symptoms = [
        symptom
        for symptom in selected_symptoms
        if symptom not in available_symptoms
    ]

    if invalid_symptoms:

        raise ValueError(
            f"Unknown symptoms: {invalid_symptoms}"
        )

    # Remove repeated selections
    selected_symptoms = list(set(selected_symptoms))

    # Convert symptoms into binary features
    encoded = encoder.transform(
        [selected_symptoms]
    )

    input_data = pd.DataFrame(
        encoded,
        columns=encoder.classes_
    )

    # Match exact training feature order
    input_data = input_data[feature_names]

    # Predict disease
    prediction = model.predict(input_data)[0]

    return prediction


# ==========================================
# 4. TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    print("\n========== DISEASE PREDICTION SYSTEM ==========")

    print("\nAvailable sample symptoms:")

    print(", ".join(encoder.classes_[:20]))

    user_input = input(
        "\nEnter symptoms separated by commas: "
    )

    symptoms = user_input.split(",")

    try:

        result = predict_disease(symptoms)

        print("\n========== PREDICTION RESULT ==========")

        print("Predicted Disease:", result)

        print(
            "\nNote: This is an educational ML prediction, "
            "not a medical diagnosis."
        )

    except ValueError as error:

        print("\nError:", error)
