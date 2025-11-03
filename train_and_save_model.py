"""
Train and Save Random Forest Model for Addiction Risk Prediction
Optimized version — same naming conventions and logic, but cleaner and faster.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Training Random Forest Model for Addiction Risk Prediction")
print("=" * 60)

# === Step 1: Load and preprocess the dataset === #
print("\n📂 Loading dataset...")

try:
    # Load dataset once
    df = pd.read_excel("Dataset.xlsx", sheet_name="Form Responses 1", engine="openpyxl")
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean column names and drop unnecessary ones
    df.columns = df.columns.str.strip()
    df.drop(columns=["Email Address", "Timestamp", "Name"], inplace=True, errors="ignore")

    # Rename columns for consistency
    column_mapping = {
        "Age": "age",
        "Which substances have you used or are currently using?": "substances_used",
        "How often do you consume any of the substances listed above?": "substance_freq",
        "0": "first_use_age",
        "Do you experience cravings or withdrawal symptoms when you stop using these substances?": "withdrawal_symptoms",
        "Do you engage in any of the following behaviors? (Select all that apply)": "risky_behaviors",
        "Do you use substances or engage in behaviors to cope with stress or emotional challenges?": "coping_mechanism",
        "Have you been diagnosed with any of the following mental health conditions?": "mental_health_diagnosis",
        "How often do you feel overwhelmed or stressed in your daily life?": "stress_level",
        "Do you have access to a support system (family, friends, professional help)?": "support_system",
    }
    df.rename(columns=column_mapping, inplace=True)

    # Fill missing values efficiently
    default_fill = {
        "substances_used": "none",
        "substance_freq": "never",
        "risky_behaviors": "none",
        "mental_health_diagnosis": "none",
    }
    df.fillna(default_fill, inplace=True)

    # Clean string columns
    df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == "object" else col)
    print("✅ Data preprocessing completed successfully")

    # === Step 2: Encode categorical variables === #
    print("\n🔄 Encoding categorical variables...")

    ordinal_mapping = {
        "substance_freq": {"never": 1, "rarely": 2, "monthly": 3, "weekly": 4, "daily": 5},
        "withdrawal_symptoms": {"no": 1, "sometimes": 2, "yes": 3},
        "stress_level": {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5},
        "support_system": {"no": 1, "sometimes": 2, "yes": 3},
        "coping_mechanism": {"no": 1, "sometimes": 2, "yes": 3},
    }
    for col, mapping in ordinal_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(1)

    # Substances encoding
    substance_mapping = {
        "alcohol": 1,
        "tobacco": 2,
        "prescription drugs (e.g., opioids, painkillers)": 3,
        "cannabis": 4,
        "recreational drugs (e.g., cocaine, heroin)": 5,
        "none": 6,
        "others": 7,
    }
    df["substances_used"] = df["substances_used"].map(substance_mapping).fillna(6)

    # Mental health encoding
    mental_health_mapping = {
        "depression": 1,
        "anxiety": 2,
        "bipolar disorder": 3,
        "ocd": 4,
        "ptsd": 5,
        "none": 6,
    }
    df["mental_health_diagnosis"] = df["mental_health_diagnosis"].map(mental_health_mapping).fillna(6)

    # Scale numerical features (single fit_transform call for both)
    scaler = MinMaxScaler()
    df[["age_scaled", "first_use_age_scaled"]] = scaler.fit_transform(df[["age", "first_use_age"]])

    print("✅ Encoding and scaling completed")

except Exception as e:
    print(f"❌ Error loading/preprocessing dataset: {e}")
    exit(1)

# === Step 3: Define features and target === #
print("\n🎯 Preparing features and target variable...")

selected_features = [
    "substance_freq", "first_use_age", "first_use_age_scaled",
    "age", "age_scaled", "mental_health_diagnosis",
    "stress_level", "support_system", "withdrawal_symptoms", "coping_mechanism"
]

missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    exit(1)

X, y = df[selected_features], df["substances_used"]

print(f"✅ Features prepared: {X.shape[1]} features")
print(f"✅ Target variable: {y.name} with {y.nunique()} unique classes")

# === Step 4: Split the data === #
print("\n🔀 Splitting data (80/20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

# === Step 5: Train Random Forest === #
print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training completed")

# === Step 6: Evaluate model === #
print("\n📊 Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"   Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# === Step 7: Feature importance === #
print("\n🔍 Top 10 Most Important Features:")
feature_importance = (
    pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
    .sort_values(by="Importance", ascending=False)
)
for _, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")

# === Step 8: Save and verify the model === #
print("\n💾 Saving model...")
model_filename = "random_forest_model.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved successfully as '{model_filename}'")

print("\n🔍 Verifying saved model...")
loaded_model = joblib.load(model_filename)
test_prediction = loaded_model.predict(X_test[:1])
print("✅ Model loaded and verified")
print(f"   Sample prediction: {test_prediction[0]}")

print("\n" + "=" * 60)
print("✅ ALL DONE! Model is ready for use.")
print("=" * 60)
print(f"\nUse '{model_filename}' in:")
print("  - app.py (Streamlit UI)")
print("  - SHAP_analysis.ipynb (Interpretability)")
print("  - scripts/add_shap_analysis.py (Visualizations)")
