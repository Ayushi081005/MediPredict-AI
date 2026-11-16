
import pandas as pd
import joblib

from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================

X_train = pd.read_csv(REPORT_PATH / "X_train.csv")
X_test = pd.read_csv(REPORT_PATH / "X_test.csv")

y_train = pd.read_csv(REPORT_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(REPORT_PATH / "y_test.csv").squeeze()


print("\n========== MODEL TRAINING ==========")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}


# ==========================================
# 4. TRAIN AND EVALUATE
# ==========================================

results = []

best_model = None
best_model_name = None
best_f1 = -1

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_model_name = name


# ==========================================
# 5. SAVE MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    REPORT_PATH / "model_comparison.csv",
    index=False
)


print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))


# ==========================================
# 6. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_PATH / "best_model.pkl"
)

print("\nBest Model:", best_model_name)

print("Best Macro F1 Score:", round(best_f1, 4))


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

best_predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")

print(report)

with open(
    REPORT_PATH / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# 8. SAVE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=best_model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=best_model.classes_,
    columns=best_model.classes_
)

cm_df.to_csv(
    REPORT_PATH / "confusion_matrix.csv"
)


print("\nTraining and evaluation completed!")
