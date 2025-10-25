"""
Streamlit Web Application for Addiction Risk Prediction
Interactive demo for predicting addiction risk based on user inputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Addiction Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'random_forest_model.pkl' exists.")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Addiction Risk Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application uses machine learning to predict addiction risk based on various factors.
    Please provide accurate information for the best prediction results.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Input fields
    age = st.sidebar.slider("Age", min_value=10, max_value=80, value=25, help="Patient's current age")
    
    first_use_age = st.sidebar.slider(
        "Age of First Substance Use", 
        min_value=5, 
        max_value=50, 
        value=16,
        help="Age when first used any substance"
    )
    
    substance_freq = st.sidebar.selectbox(
        "Substance Use Frequency",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Rarely", 2: "Occasionally", 3: "Regularly", 4: "Frequently", 5: "Daily"}[x],
        help="How often substances are used"
    )
    
    mental_health = st.sidebar.selectbox(
        "Mental Health Diagnosis",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Has a diagnosed mental health condition"
    )
    
    stress_level = st.sidebar.slider(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5,
        help="Self-reported stress level (1=Low, 10=High)"
    )
    
    support_system = st.sidebar.selectbox(
        "Support System Quality",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Very Poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Excellent"}[x],
        help="Quality of family/social support"
    )
    
    withdrawal_symptoms = st.sidebar.selectbox(
        "Withdrawal Symptoms",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Experiences withdrawal symptoms"
    )
    
    coping_mechanism = st.sidebar.selectbox(
        "Coping Mechanism Quality",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Very Poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Excellent"}[x],
        help="Quality of stress coping mechanisms"
    )
    
    # Calculate scaled features (simple normalization)
    age_scaled = (age - 10) / (80 - 10)
    first_use_age_scaled = (first_use_age - 5) / (50 - 5)
    
    # Create feature DataFrame
    features = pd.DataFrame({
        'substance_freq': [substance_freq],
        'first_use_age': [first_use_age],
        'first_use_age_scaled': [first_use_age_scaled],
        'age': [age],
        'age_scaled': [age_scaled],
        'mental_health_diagnosis': [mental_health],
        'stress_level': [stress_level],
        'support_system': [support_system],
        'withdrawal_symptoms': [withdrawal_symptoms],
        'coping_mechanism': [coping_mechanism]
    })
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Input Summary")
        
        # Display input summary
        summary_data = {
            "Factor": [
                "Age", "First Use Age", "Substance Frequency", 
                "Mental Health Diagnosis", "Stress Level", "Support System",
                "Withdrawal Symptoms", "Coping Mechanism"
            ],
            "Value": [
                f"{age} years",
                f"{first_use_age} years",
                {1: "Rarely", 2: "Occasionally", 3: "Regularly", 4: "Frequently", 5: "Daily"}[substance_freq],
                "Yes" if mental_health == 1 else "No",
                f"{stress_level}/10",
                {1: "Very Poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Excellent"}[support_system],
                "Yes" if withdrawal_symptoms == 1 else "No",
                {1: "Very Poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Excellent"}[coping_mechanism]
            ]
        }
        st.table(pd.DataFrame(summary_data))
    
    with col2:
        st.subheader("🎯 Risk Factors")
        
        # Calculate risk indicators
        risk_factors = []
        if substance_freq >= 4:
            risk_factors.append("High frequency use")
        if first_use_age < 15:
            risk_factors.append("Early first use")
        if mental_health == 1:
            risk_factors.append("Mental health condition")
        if stress_level >= 7:
            risk_factors.append("High stress")
        if support_system <= 2:
            risk_factors.append("Poor support system")
        if withdrawal_symptoms == 1:
            risk_factors.append("Withdrawal symptoms")
        if coping_mechanism <= 2:
            risk_factors.append("Poor coping skills")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(f"⚠️ {factor}")
        else:
            st.success("✅ No major risk factors identified")
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Predict Addiction Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing risk factors..."):
            # Make prediction
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            
            # Display results
            st.markdown("## 📈 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                risk_class = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "#f44336" if prediction == 1 else "#4caf50"
                
                st.markdown(f"""
                <div class="prediction-box {'high-risk' if prediction == 1 else 'low-risk'}">
                    <h2 style="color: {risk_color}; margin: 0;">
                        {'⚠️ HIGH RISK' if prediction == 1 else '✅ LOW RISK'}
                    </h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">
                        Addiction Risk Classification: <strong>{risk_class}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba[1] * 100,
                    title={'text': "Risk Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prediction_proba[1] > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction == 1:
                st.error("""
                **Immediate Actions Recommended:**
                - Consult with a healthcare professional or addiction specialist
                - Consider joining a support group or counseling program
                - Develop a comprehensive treatment plan
                - Strengthen support network (family, friends, community)
                - Learn and practice healthy coping mechanisms
                """)
            else:
                st.success("""
                **Preventive Measures:**
                - Maintain healthy lifestyle habits
                - Continue building strong support systems
                - Practice stress management techniques
                - Stay aware of risk factors
                - Seek help early if concerns arise
                """)
            
            # Feature importance visualization
            st.markdown("### 📊 Feature Contribution Analysis")
            
            feature_names = features.columns.tolist()
            feature_values = features.iloc[0].values
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values
            })
            
            fig = px.bar(
                importance_df,
                x='Value',
                y='Feature',
                orientation='h',
                title='Input Feature Values',
                color='Value',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Disclaimer:</strong> This tool is for educational and screening purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.</p>
        <p>Always consult with qualified healthcare professionals for addiction assessment and treatment.</p>
        <p style="margin-top: 1rem;">© 2025 Addiction Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
