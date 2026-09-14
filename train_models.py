"""Train and persist the ML artifacts required by the Flask app."""
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "dataset", "real_drug_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df = df.drop(columns=[c for c in ["Patient_ID", "Side_Effects", "Improvement_Score", "Treatment_Duration_days"] if c in df.columns])

    le_gender = LabelEncoder(); le_condition = LabelEncoder()
    le_drug = LabelEncoder(); le_dosage = LabelEncoder()
    df["ge"] = le_gender.fit_transform(df["Gender"])
    df["ce"] = le_condition.fit_transform(df["Condition"])
    df["de"] = le_drug.fit_transform(df["Drug_Name"])
    df["doe"] = le_dosage.fit_transform(df["Dosage_mg"])

    X = df[["Age", "ge", "ce"]]
    Xtr, _, y_drug, _ = train_test_split(X, df["de"], test_size=0.2, random_state=42, stratify=df["de"])
    Xtr2, _, y_dose, _ = train_test_split(X, df["doe"], test_size=0.2, random_state=42, stratify=df["doe"])

    drug_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    drug_model.fit(Xtr, y_drug)
    dosage_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    dosage_model.fit(Xtr2, y_dose)

    artifacts = {
        "drug_model.pkl": drug_model, "dosage_model.pkl": dosage_model,
        "le_gender.pkl": le_gender, "le_condition.pkl": le_condition,
        "le_drug.pkl": le_drug, "le_dosage.pkl": le_dosage,
    }
    for name, obj in artifacts.items():
        joblib.dump(obj, os.path.join(MODEL_DIR, name))
    print(f"Created {len(artifacts)} model artifacts in {MODEL_DIR}")


if __name__ == "__main__":
    main()
