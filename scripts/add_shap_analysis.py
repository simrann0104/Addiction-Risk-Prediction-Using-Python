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

# Load the trained model
print("Loading trained model...")
model = joblib.load('random_forest_model.pkl')

# Load the dataset
print("Loading dataset...")
df = pd.read_excel('Cleaned_Dataset_Encoded.xlsx')

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

# If binary classification, use class 1 (positive class)
if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
else:
    shap_values_plot = shap_values

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
shap_explanation = shap.Explanation(
    values=shap_values_plot[sample_idx],
    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    data=X.iloc[sample_idx].values,
    feature_names=X.columns.tolist()
)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_explanation, show=False)
plt.title(f'SHAP Waterfall Plot - Sample Prediction {sample_idx}', fontsize=14, fontweight='bold')
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
