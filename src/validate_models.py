
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_PATH = BASE_DIR / "reports"


# ==========================================
# 2. LOAD COMPLETE PROCESSED DATA
# ==========================================

X_train = pd.read_csv(REPORT_PATH / "X_train.csv")
X_test = pd.read_csv(REPORT_PATH / "X_test.csv")

y_train = pd.read_csv(
    REPORT_PATH / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    REPORT_PATH / "y_test.csv"
).squeeze()

X = pd.concat(
    [X_train, X_test],
    ignore_index=True
)

y = pd.concat(
    [y_train, y_test],
    ignore_index=True
)


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
        class_weight="balanced",
        random_state=42
    )
}


# ==========================================
# 4. DEFINE CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]


# ==========================================
# 5. EVALUATE EACH MODEL
# ==========================================

results = []

print("\n========== 5-FOLD CROSS VALIDATION ==========")

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Model": name,

        "Accuracy": scores["test_accuracy"].mean(),

        "Precision": scores["test_precision_macro"].mean(),

        "Recall": scores["test_recall_macro"].mean(),

        "F1 Score": scores["test_f1_macro"].mean()
    }

    results.append(result)

    print("Accuracy:", round(result["Accuracy"], 4))
    print("Precision:", round(result["Precision"], 4))
    print("Recall:", round(result["Recall"], 4))
    print("F1 Score:", round(result["F1 Score"], 4))


# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    REPORT_PATH / "cross_validation_results.csv",
    index=False
)


print("\n========== FINAL CROSS VALIDATION RESULTS ==========")

print(results_df.to_string(index=False))

print("\nResults saved successfully!")
