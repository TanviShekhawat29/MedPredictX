import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from backend.router import predict

st.set_page_config(page_title="MedPredictX", layout="wide")

# =========================
# TITLE
# =========================
st.title("🏥 MedPredictX")
st.markdown("### Diabetes Readmission Risk Predictor")

st.write("AI-powered system to predict hospital readmission risk for diabetic patients.")

# =========================
# INPUTS
# =========================
st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 30)
    time_in_hospital = st.slider("Days in Hospital", 1, 20, 3)
    num_medications = st.slider("Number of Medications", 0, 50, 5)

with col2:
    number_inpatient = st.slider("Previous Admissions", 0, 10, 0)
    number_emergency = st.slider("Emergency Visits", 0, 10, 0)
    number_outpatient = st.slider("Outpatient Visits", 0, 10, 1)

input_data = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_medications": num_medications,
    "number_inpatient": number_inpatient,
    "number_emergency": number_emergency,
    "number_outpatient": number_outpatient,
}

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Risk"):

    prediction, prob = predict(input_data)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Readmission")
    else:
        st.success("✅ Low Risk")

    if prob > 0.75:
        st.write(f"Risk Level: 🔴 High ({prob:.2f})")
    elif prob > 0.5:
        st.write(f"Risk Level: 🟡 Moderate ({prob:.2f})")
    else:
        st.write(f"Risk Level: 🟢 Low ({prob:.2f})")

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("🧠 Interpretation")

    if prediction == 1:
        st.write("Higher risk due to frequent hospital visits and medication usage.")
    else:
        st.write("Lower risk based on stable health indicators.")

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.write("• Regular monitoring required")
        st.write("• Follow-up visits advised")
        st.write("• Medication review recommended")
    else:
        st.write("• Maintain healthy lifestyle")
        st.write("• Routine check-ups advised")