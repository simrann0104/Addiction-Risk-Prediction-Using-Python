"""
SHAP Analysis Script for Addiction Risk Prediction Model
This script adds model interpretability using SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
print("Loading trained model...")
try:
    model = joblib.load('random_forest_model.pkl')
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: 'random_forest_model.pkl' not found!")
    print("Please run 'python train_and_save_model.py' first to generate the model.")
    exit(1)

# Load and preprocess the dataset
print("Loading and preprocessing dataset...")
try:
    df = pd.read_excel('Dataset.xlsx', sheet_name='Form Responses 1', engine='openpyxl')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop unnecessary columns
    df.drop(columns=['Email Address', 'Timestamp', 'Name'], inplace=True, errors='ignore')
    
    # Rename columns
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
    
    # Clean text
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
    
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
    
    # Min-Max Scaling
    scaler = MinMaxScaler()
    df['age_scaled'] = scaler.fit_transform(df[['age']])
    df['first_use_age_scaled'] = scaler.fit_transform(df[['first_use_age']])
    
    print("✅ Dataset loaded and preprocessed successfully")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Define features (same as used in training)
selected_features = [
    'substance_freq', 'first_use_age', 'first_use_age_scaled',
    'age', 'age_scaled', 'mental_health_diagnosis', 'stress_level',
    'support_system', 'withdrawal_symptoms', 'coping_mechanism'
]

# Prepare feature matrix
X = df[selected_features]

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X)

# Debug: Check the structure
print(f"SHAP values type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"Number of classes: {len(shap_values)}")
    print(f"Shape of each class: {[sv.shape for sv in shap_values]}")
elif isinstance(shap_values, np.ndarray):
    print(f"SHAP values shape: {shap_values.shape}")

# Handle multi-class output
# For multi-class, shap_values is typically a list with one array per class
if isinstance(shap_values, list) and len(shap_values) > 1:
    # Multi-class classification
    print(f"✓ Multi-class model detected with {len(shap_values)} classes")
    # Use the first class for visualization
    shap_values_plot = shap_values[0]
    use_class_idx = 0
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Alternative multi-class format: (n_samples, n_features, n_classes)
    print(f"✓ Multi-class model (3D array) with shape {shap_values.shape}")
    shap_values_plot = shap_values[:, :, 0]  # Use first class
    use_class_idx = 0
else:
    # Binary or regression
    shap_values_plot = shap_values if not isinstance(shap_values, list) else shap_values[1]
    use_class_idx = None
    print("✓ Using binary/regression format")

print(f"Final shap_values_plot shape: {shap_values_plot.shape}")

# Plot 1: SHAP Summary Plot (Feature Importance)
print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_plot, X, show=False)
plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP summary plot saved as 'shap_summary_plot.png'")

# Plot 2: SHAP Bar Plot (Mean Absolute SHAP Values)
print("Generating SHAP bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_plot, X, plot_type="bar", show=False)
plt.title('Mean Absolute SHAP Values by Feature', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP bar plot saved as 'shap_bar_plot.png'")

# Plot 3: SHAP Waterfall Plot (Single Prediction Explanation)
print("Generating SHAP waterfall plot for sample prediction...")
sample_idx = 0

# Get SHAP values for a single sample
# shap_values_plot has shape (n_samples, n_features)
shap_values_single = shap_values_plot[sample_idx, :]

# Get the expected value (base value)
if isinstance(explainer.expected_value, (list, np.ndarray)):
    if use_class_idx is not None:
        expected_val = explainer.expected_value[use_class_idx]
    else:
        expected_val = explainer.expected_value[0] if len(explainer.expected_value) > 0 else explainer.expected_value
else:
    expected_val = explainer.expected_value

shap_explanation = shap.Explanation(
    values=shap_values_single,
    base_values=expected_val,
    data=X.iloc[sample_idx].values,
    feature_names=X.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_explanation, show=False)
class_info = f" (Class {use_class_idx})" if use_class_idx is not None else ""
plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}{class_info}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP waterfall plot saved as 'shap_waterfall_plot.png'")

# Calculate and save feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Absolute_SHAP': np.abs(shap_values_plot).mean(axis=0)
}).sort_values(by='Mean_Absolute_SHAP', ascending=False)

feature_importance.to_csv('shap_feature_importance.csv', index=False)
print("✅ SHAP feature importance saved as 'shap_feature_importance.csv'")

print("\n" + "="*60)
print("SHAP Analysis Complete!")
print("="*60)
print("\nGenerated Files:")
print("  1. shap_summary_plot.png - Feature importance with value distributions")
print("  2. shap_bar_plot.png - Mean absolute SHAP values")
print("  3. shap_waterfall_plot.png - Single prediction explanation")
print("  4. shap_feature_importance.csv - Numerical feature importance values")
