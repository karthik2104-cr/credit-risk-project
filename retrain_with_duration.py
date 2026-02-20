import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("german_credit_data.csv", index_col=0)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Fill missing values
df["Saving accounts"] = df["Saving accounts"].fillna("little")
df["Checking account"] = df["Checking account"].fillna("little")

# Encode categorical columns
encoders = {}
for col in ["Sex", "Housing", "Saving accounts", "Checking account"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")
    print(f"Saved {col}_encoder.pkl")

# Encode target
target_le = LabelEncoder()
df["Risk"] = target_le.fit_transform(df["Risk"])
joblib.dump(target_le, "target_encoder.pkl")
print("Saved target_encoder.pkl")

# Features now INCLUDING Duration
feature_cols = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"]
X = df[feature_cols]
y = df["Risk"]

print("\nFeatures used for training:", feature_cols)
print("Target classes:", target_le.classes_)

# Train ExtraTrees model (same type as before)
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "extra_trees_credit_model.pkl")
print("\nModel retrained and saved with features:", list(model.feature_names_in_))
