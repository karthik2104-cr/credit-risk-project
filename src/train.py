"""
Train (or retrain) the ExtraTrees credit-risk model.

Run from the project root:
    python src/train.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "german_credit_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, index_col=0)
print("Columns:", df.columns.tolist())
print("Shape  :", df.shape)

# ---------------------------------------------------------------------------
# Fill missing values
# ---------------------------------------------------------------------------
df["Saving accounts"] = df["Saving accounts"].fillna("little")
df["Checking account"] = df["Checking account"].fillna("little")

# ---------------------------------------------------------------------------
# Encode categorical features
# ---------------------------------------------------------------------------
encoders = {}
for col in ["Sex", "Housing", "Saving accounts", "Checking account"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    out_path = MODELS_DIR / f"{col}_encoder.pkl"
    joblib.dump(le, out_path)
    print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Encode target
# ---------------------------------------------------------------------------
target_le = LabelEncoder()
df["Risk"] = target_le.fit_transform(df["Risk"])
joblib.dump(target_le, MODELS_DIR / "target_encoder.pkl")
print("Saved target_encoder.pkl")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Age", "Sex", "Job", "Housing",
    "Saving accounts", "Checking account",
    "Credit amount", "Duration",
]
X, y = df[FEATURE_COLS], df["Risk"]

print("\nFeatures used for training:", FEATURE_COLS)
print("Target classes           :", target_le.classes_)

model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

model_path = MODELS_DIR / "extra_trees_credit_model.pkl"
joblib.dump(model, model_path)
print("\nModel saved to:", model_path)
print("Feature names :", list(model.feature_names_in_))
