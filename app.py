import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# --------------------------- #
# PAGE CONFIGURATION
# --------------------------- #
st.set_page_config(
    page_title="Addiction Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# --------------------------- #
# CUSTOM CSS
# --------------------------- #
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}
h1, h2, h3 {
    text-align: center;
    color: #1f77b4;
}
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 12px;
    font-size: 1.1rem;
    width: 100%;
    height: 3rem;
}
.stButton > button:hover {
    background-color: #125a91;
}
.card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-top: 2rem;
}
.result-box {
    border-radius: 15px;
    padding: 2rem;
    color: white;
    text-align: center;
    font-size: 1.3rem;
}
.low-risk {
    background-color: #4caf50;
}
.high-risk {
    background-color: #f44336;
}
</style>
""", unsafe_allow_html=True)


# --------------------------- #
# LOAD MODEL
# --------------------------- #
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'random_forest_model.pkl' exists.")
        return None


# --------------------------- #
# MAIN FUNCTION
# --------------------------- #
def main():
    st.markdown("<h1>🏥 Addiction Risk Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Answer a few simple questions to estimate addiction risk based on lifestyle and mental health factors.</p>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    # Initialize step state
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}

    # Helper for navigation
    def next_step():
        st.session_state.step += 1

    def prev_step():
        if st.session_state.step > 1:
            st.session_state.step -= 1

    # --------------------------- #
    # STEP 1: Age Details
    # --------------------------- #
    if st.session_state.step == 1:
        with st.container():
            st.markdown("<div class='card'><h3>🧑 Age Information</h3>", unsafe_allow_html=True)
            age = st.slider("Select your age", 10, 80, 25)
            first_use_age = st.slider("Age of first substance use", 5, 50, 16)
            st.session_state.form_data["age"] = age
            st.session_state.form_data["first_use_age"] = first_use_age
            st.button("Next ➡️", on_click=next_step)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------- #
    # STEP 2: Health & Lifestyle
    # --------------------------- #
    elif st.session_state.step == 2:
        with st.container():
            st.markdown("<div class='card'><h3>🧠 Health & Lifestyle Factors</h3>", unsafe_allow_html=True)
            substance_freq = st.select_slider(
                "Substance Use Frequency",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: {1:"Rarely",2:"Occasionally",3:"Regularly",4:"Frequently",5:"Daily"}[x]
            )
            mental_health = st.radio("Do you have a diagnosed mental health condition?", ["No", "Yes"])
            stress_level = st.slider("Rate your current stress level", 1, 10, 5)
            st.session_state.form_data["substance_freq"] = substance_freq
            st.session_state.form_data["mental_health_diagnosis"] = 1 if mental_health == "Yes" else 0
            st.session_state.form_data["stress_level"] = stress_level
            st.button("⬅️ Back", on_click=prev_step)
            st.button("Next ➡️", on_click=next_step)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------- #
    # STEP 3: Support & Coping
    # --------------------------- #
    elif st.session_state.step == 3:
        with st.container():
            st.markdown("<div class='card'><h3>🤝 Support System & Coping</h3>", unsafe_allow_html=True)
            support_system = st.select_slider(
                "Quality of Support System",
                options=[1,2,3,4,5],
                format_func=lambda x:{1:"Very Poor",2:"Poor",3:"Moderate",4:"Good",5:"Excellent"}[x]
            )
            withdrawal_symptoms = st.radio("Do you experience withdrawal symptoms?", ["No","Yes"])
            coping_mechanism = st.select_slider(
                "Coping Mechanism Quality",
                options=[1,2,3,4,5],
                format_func=lambda x:{1:"Very Poor",2:"Poor",3:"Moderate",4:"Good",5:"Excellent"}[x]
            )
            st.session_state.form_data["support_system"] = support_system
            st.session_state.form_data["withdrawal_symptoms"] = 1 if withdrawal_symptoms == "Yes" else 0
            st.session_state.form_data["coping_mechanism"] = coping_mechanism

            st.button("⬅️ Back", on_click=prev_step)
            st.button("Next ➡️", on_click=next_step)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------- #
    # STEP 4: Prediction
    # --------------------------- #
    elif st.session_state.step == 4:
        data = st.session_state.form_data
        st.markdown("<div class='card'><h3>📊 Summary of Your Inputs</h3>", unsafe_allow_html=True)
        st.table(pd.DataFrame.from_dict(data, orient="index", columns=["Value"]))
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Predict Addiction Risk"):
            with st.spinner("Analyzing your responses..."):
                time.sleep(1)

                # Prepare data for model
                age_scaled = (data["age"] - 10) / 70
                first_use_age_scaled = (data["first_use_age"] - 5) / 45
                features = pd.DataFrame([{
                    "substance_freq": data["substance_freq"],
                    "first_use_age": data["first_use_age"],
                    "first_use_age_scaled": first_use_age_scaled,
                    "age": data["age"],
                    "age_scaled": age_scaled,
                    "mental_health_diagnosis": data["mental_health_diagnosis"],
                    "stress_level": data["stress_level"],
                    "support_system": data["support_system"],
                    "withdrawal_symptoms": data["withdrawal_symptoms"],
                    "coping_mechanism": data["coping_mechanism"]
                }])

                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0][1] * 100

                # Result UI
                risk_label = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"
                risk_class = "high-risk" if prediction == 1 else "low-risk"

                st.markdown(f"<div class='result-box {risk_class}'><h2>{risk_label}</h2><p>Predicted Probability: {prediction_proba:.2f}%</p></div>", unsafe_allow_html=True)

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba,
                    title={'text': "Addiction Risk (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        st.button("⬅️ Back", on_click=prev_step)

    # --------------------------- #
    # FOOTER
    # --------------------------- #
    st.markdown("""
    <hr>
    <div style='text-align:center; color:gray;'>
        <p><b>Disclaimer:</b> This app is for educational and screening purposes only.</p>
        <p>Always consult with qualified healthcare professionals for medical advice.</p>
        <p>© 2025 Addiction Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()