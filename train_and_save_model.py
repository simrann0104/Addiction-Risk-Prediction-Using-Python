"""
Train and Save Random Forest Model for Addiction Risk Prediction
This script preprocesses data, trains the model, and saves it as 'random_forest_model.pkl'
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

print("="*60)
print("Training Random Forest Model for Addiction Risk Prediction")
print("="*60)

# === Step 1: Load and preprocess the dataset === #
print("\n📂 Loading dataset...")
try:
    # Load the raw dataset
    df = pd.read_excel('Dataset.xlsx', sheet_name='Form Responses 1', engine='openpyxl')
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop unnecessary columns
    df.drop(columns=['Email Address', 'Timestamp', 'Name'], inplace=True, errors='ignore')
    
    # Rename columns for easier handling
    column_mapping = {
        'Age': 'age',
        'Which substances have you used or are currently using?': 'substances_used',
        'How often do you consume any of the substances listed above?': 'substance_freq',
        '0': 'first_use_age',
        'Do you experience cravings or withdrawal symptoms when you stop using these substances?': 'withdrawal_symptoms',
        'Do you engage in any of the following behaviors? (Select all that apply)': 'risky_behaviors',
        'Do you use substances or engage in behaviors to cope with stress or emotional challenges?': 'coping_mechanism',
        'Have you been diagnosed with any of the following mental health conditions?': 'mental_health_diagnosis',
        'How often do you feel overwhelmed or stressed in your daily life?': 'stress_level',
        'Do you have access to a support system (family, friends, professional help)?': 'support_system'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Fill missing values
    df.fillna({'substances_used': 'none', 'substance_freq': 'never', 'risky_behaviors': 'none', 'mental_health_diagnosis': 'none'}, inplace=True)
    
    # Clean text - lowercase and strip spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    
    print("✅ Data preprocessing: Column names cleaned and missing values filled")
    
    # === Step 2: Encode categorical variables === #
    print("\n🔄 Encoding categorical variables...")
    
    # Ordinal encodings
    ordinal_mapping = {
        'substance_freq': {'never': 1, 'rarely': 2, 'monthly': 3, 'weekly': 4, 'daily': 5},
        'withdrawal_symptoms': {'no': 1, 'sometimes': 2, 'yes': 3},
        'stress_level': {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},
        'support_system': {'no': 1, 'sometimes': 2, 'yes': 3},
        'coping_mechanism': {'no': 1, 'sometimes': 2, 'yes': 3}
    }
    
    for col, mapping in ordinal_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    
    # Substances used encoding
    substance_mapping = {
        'alcohol': 1,
        'tobacco': 2,
        'prescription drugs (e.g., opioids, painkillers)': 3,
        'cannabis': 4,
        'recreational drugs (e.g., cocaine, heroin)': 5,
        'none': 6,
        'others': 7
    }
    df['substances_used'] = df['substances_used'].replace(substance_mapping)
    
    # Mental health encoding
    mental_health_mapping = {
        'depression': 1,
        'anxiety': 2,
        'bipolar disorder': 3,
        'ocd': 4,
        'ptsd': 5,
        'none': 6
    }
    df['mental_health_diagnosis'] = df['mental_health_diagnosis'].apply(
        lambda x: mental_health_mapping.get(x.lower() if isinstance(x, str) else x, 6)
    )
    
    # Min-Max Scaling for age and first_use_age
    scaler = MinMaxScaler()
    df['age_scaled'] = scaler.fit_transform(df[['age']])
    df['first_use_age_scaled'] = scaler.fit_transform(df[['first_use_age']])
    
    print("✅ Categorical encoding completed")
    
except Exception as e:
    print(f"❌ Error loading/preprocessing dataset: {e}")
    exit(1)

# === Step 2: Define features and target === #
print("\n🎯 Preparing features and target variable...")

# Define the features to use (based on feature importance analysis)
selected_features = [
    'substance_freq', 'first_use_age', 'first_use_age_scaled',
    'age', 'age_scaled', 'mental_health_diagnosis', 'stress_level',
    'support_system', 'withdrawal_symptoms', 'coping_mechanism'
]

# Check if all features exist
missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    print("Available columns:", df.columns.tolist())
    exit(1)

X = df[selected_features]
y = df['substances_used']

print(f"✅ Features prepared: {X.shape[1]} features")
print(f"✅ Target variable: {y.name} with {y.nunique()} unique classes")

# === Step 3: Train/Test Split === #
print("\n🔀 Splitting data into train and test sets (80/20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# === Step 4: Train Random Forest Model === #
print("\n🌲 Training Random Forest Classifier...")
print("   Parameters: n_estimators=100, random_state=42")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# === Step 5: Evaluate the Model === #
print("\n📊 Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"   Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*60}")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# === Step 6: Display Feature Importance === #
print("\n🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")

# === Step 7: Save the Model === #
print("\n💾 Saving model...")
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Model saved successfully as '{model_filename}'")

# === Step 8: Verify Model Loading === #
print("\n🔍 Verifying saved model...")
loaded_model = joblib.load(model_filename)
test_prediction = loaded_model.predict(X_test[:1])
print(f"✅ Model loaded and tested successfully!")
print(f"   Test prediction result: {test_prediction[0]}")

print("\n" + "="*60)
print("✅ ALL DONE! Model is ready for use.")
print("="*60)
print(f"\nYou can now use '{model_filename}' with:")
print("  - app.py (Streamlit application)")
print("  - SHAP_analysis.ipynb (Model interpretability)")
print("  - scripts/add_shap_analysis.py (SHAP visualizations)")
