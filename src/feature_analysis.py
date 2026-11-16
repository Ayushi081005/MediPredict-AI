
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
REPORT_PATH = BASE_DIR / "reports"

REPORT_PATH.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD MODEL AND FEATURES
# ==========================================

model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(
    REPORT_PATH / "X_train.csv"
)


# ==========================================
# 3. CHECK MODEL TYPE
# ==========================================

if not hasattr(model, "feature_importances_"):

    raise TypeError(
        "The selected model does not support "
        "tree-based feature importance."
    )


# ==========================================
# 4. CALCULATE FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({

    "Symptom": X_train.columns,

    "Importance": model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 5. DISPLAY TOP 15 SYMPTOMS
# ==========================================

top_features = importance_df.head(15)

print("\n========== TOP 15 IMPORTANT SYMPTOMS ==========")

print(top_features.to_string(index=False))


# ==========================================
# 6. SAVE FEATURE IMPORTANCE
# ==========================================

importance_df.to_csv(
    REPORT_PATH / "feature_importance.csv",
    index=False
)


# ==========================================
# 7. VISUALIZE RESULTS
# ==========================================

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Symptom"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")

plt.ylabel("Symptoms")

plt.title(
    "Top 15 Symptoms - Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    REPORT_PATH / "feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature analysis completed!")
