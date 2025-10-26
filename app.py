import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. DEFINE FEATURE ORDER (CRITICAL) ---
# This MUST match the order your model was trained on
EXPECTED_FEATURES = [
    'substance_freq', 'first_use_age', 'first_use_age_scaled',
    'age', 'age_scaled', 'mental_health_diagnosis', 'stress_level',
    'support_system', 'withdrawal_symptoms', 'coping_mechanism'
]

# --- 2. LOAD MODEL & SHAP EXPLAINER ---
# This function runs only once, caching the model & explainer
@st.cache_resource
def load_model_and_explainer(model_path):
    """
    Loads the saved model and initializes the SHAP explainer.
    Caches the objects so they don't reload on every interaction.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure 'random_forest_model.pkl' is in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    # We need a sample of training data for the explainer's background
    # Make sure 'balanced_dataset.csv' is in the same folder
    try:
        df = pd.read_csv('balanced_dataset.csv')
        X = df.drop(columns=['Substances_Used', 'substances_used_label'])
        
        # Use a sample of 100 rows as the background (faster for the app)
        X_train_sample = X.sample(100, random_state=42) 
        
        explainer = shap.TreeExplainer(model, X_train_sample)
    except FileNotFoundError:
        st.warning("Warning: 'balanced_dataset.csv' not found. SHAP plots may be less accurate.")
        # Create a dummy background if file not found
        dummy_data = pd.DataFrame(columns=EXPECTED_FEATURES) 
        explainer = shap.TreeExplainer(model, dummy_data)
    except Exception as e:
        st.error(f"Error initializing SHAP explainer: {e}")
        explainer = None

    return model, explainer

# --- Load the resources ---
MODEL_PATH = 'random_forest_model.pkl'
model, explainer = load_model_and_explainer(MODEL_PATH)

if model is None:
    st.stop() # Stop the app if the model didn't load

# --- 3. DEFINE MAPPINGS ---
# These maps convert user-friendly names to the numbers your model needs
freq_map = {'Never': 1, 'Rarely': 2, 'Monthly': 3, 'Weekly': 4, 'Daily': 5}
yes_no_sometimes_map = {'No': 1, 'Sometimes': 2, 'Yes': 3}
stress_map = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}
mental_health_map = {'None': 6, 'Depression': 1, 'Anxiety': 2, 'Bipolar Disorder': 3, 'OCD': 4, 'PTSD': 5}

# --- 4. BUILD THE STREAMLIT UI (User Interface) ---

st.set_page_config(page_title="Addiction Risk Predictor", page_icon="🚀", layout="wide") # Use wide layout
st.title("🚀 Addiction Risk Prediction Tool")
st.write("This tool predicts addiction risk and explains *why* it made that decision using SHAP.")
st.write("---")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Usage Info")
    age = st.number_input("Current Age", min_value=10, max_value=100, value=25)
    first_use_age = st.number_input("Age of First Substance Use (Enter 0 if None)", min_value=0, max_value=100, value=16)
    substance_freq = st.selectbox("Substance Frequency", options=list(freq_map.keys()), index=1) # Default 'Rarely'
    withdrawal_symptoms = st.selectbox("Experience Withdrawal Symptoms?", options=list(yes_no_sometimes_map.keys()), index=0) # Default 'No'

with col2:
    st.subheader("Psychological Info")
    stress_level = st.selectbox("Stress Level in Daily Life", options=list(stress_map.keys()), index=2) # Default 'Sometimes'
    coping_mechanism = st.selectbox("Use Substances to Cope?", options=list(yes_no_sometimes_map.keys()), index=1) # Default 'Sometimes'
    support_system = st.selectbox("Have a Support System?", options=list(yes_no_sometimes_map.keys()), index=2) # Default 'Yes'
    mental_health_diagnosis = st.selectbox("Mental Health Diagnosis", options=list(mental_health_map.keys()), index=0) # Default 'None'

# --- 5. PREDICTION LOGIC ---

if st.button("Calculate Risk", type="primary"):
    
    # 5a. Preprocess Data
    # Simple scaling (assuming 0-100 range). 
    # For perfect accuracy, you should load your saved scaler object.
    age_scaled = age / 100.0  
    first_use_age_scaled = first_use_age / 100.0

    # 5b. Map Inputs to Numerical Values
    input_data = {
        'substance_freq': freq_map[substance_freq],
        'first_use_age': first_use_age,
        'first_use_age_scaled': first_use_age_scaled,
        'age': age,
        'age_scaled': age_scaled,
        'mental_health_diagnosis': mental_health_map[mental_health_diagnosis],
        'stress_level': stress_map[stress_level],
        'support_system': yes_no_sometimes_map[support_system],
        'withdrawal_symptoms': yes_no_sometimes_map[withdrawal_symptoms],
        'coping_mechanism': yes_no_sometimes_map[coping_mechanism]
    }

    # 5c. Create DataFrame for Model and SHAP
    features_list = [input_data[feature] for feature in EXPECTED_FEATURES]
    input_array = np.array(features_list).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=EXPECTED_FEATURES)

    # 5d. Make Prediction
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    # 5e. Display Results and SHAP plot in columns
    st.write("---")
    # 1/3 width for result, 2/3 for plot
    result_col, shap_col = st.columns([1, 2]) 

    with result_col:
        st.subheader("Prediction")
        # Assuming your model outputs 1 for "Yes" (Risk) and 0 for "No" (No Risk)
        if prediction[0] == 1:
            st.error("### High Risk (YES)")
            st.write(f"**Confidence:** {prediction_proba[0][1] * 100:.2f}%")
        else:
            st.success("### Low Risk (NO)")
            st.write(f"**Confidence:** {prediction_proba[0][0] * 100:.2f}%")
        
        st.write("This result is based on the provided inputs. The plot on the right shows which factors were most influential.")

        with st.expander("Show Raw Model Input"):
            st.json(input_data)

    # --- 5f. ADD SHAP EXPLANATION PLOT ---
    with shap_col:
        st.subheader("Why did the model decide this?")
        
        if explainer is not None:
            try:
                # Get SHAP values for class 1 ("Yes")
                shap_values_sample = explainer.shap_values(input_df)[1]
                base_value = explainer.expected_value[1]
                
                st.write("Factors pushing the risk **higher (red)** or **lower (blue)**.")

                # Create a new matplotlib figure for the plot
                fig, ax = plt.subplots() 
                
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values_sample[0],
                    base_values=base_value,
                    data=input_df.iloc[0],
                    feature_names=EXPECTED_FEATURES
                ), max_display=10, show=False) # show=False is crucial
                
                # Display the plot in Streamlit
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig) # Close the plot to save memory
                
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
        else:
            st.warning("SHAP explainer could not be loaded. Cannot display plot.")