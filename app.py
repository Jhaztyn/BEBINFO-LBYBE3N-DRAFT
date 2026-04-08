# =========================
# IMPORTS
# =========================
import streamlit as st
import joblib
import numpy as np
from datetime import date
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Diabetes CDSS", layout="wide")

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model/scaler not found.")
    st.stop()

# Optional: Properly trained calibrator
try:
    calibrator = joblib.load("calibrator.pkl")
    USE_CALIBRATOR = True
except:
    USE_CALIBRATOR = False

# =========================
# DATASET MEDIANS (PIMA-BASED)
# =========================
MEDIANS = {
    "skin": 20,
    "insulin": 79
}

# =========================
# CALIBRATION
# =========================
def predict_probability(X_scaled):
    if USE_CALIBRATOR:
        return calibrator.predict_proba(X_scaled)[0][1]
    return model.predict_proba(X_scaled)[0][1]

# =========================
# CLINICAL RULES
# =========================
def classify_risk(prob, glucose):
    if glucose >= 200:
        return "High Risk (Clinical Override)"
    elif prob >= 0.65:
        return "High Risk"
    elif prob >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"

def medication_recommendation(risk, glucose):
    if glucose >= 200:
        return "🔴 Possible Diabetes → Refer for insulin therapy evaluation"
    elif risk == "High Risk":
        return "🟠 Consider Metformin (first-line) with physician guidance"
    elif risk == "Medium Risk":
        return "🟡 Lifestyle modification ± Metformin (case-dependent)"
    else:
        return "🟢 No medication required; preventive care advised"

def lifestyle_advice(risk):
    if "High" in risk:
        return "Strict diet control, regular monitoring, and immediate consultation."
    elif risk == "Medium Risk":
        return "Improve diet, increase physical activity, monitor glucose."
    else:
        return "Maintain healthy lifestyle."

# =========================
# INTERPRETATION
# =========================
def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

def glucose_category(glucose):
    if glucose < 140: return "Normal"
    elif glucose < 200: return "Prediabetes"
    else: return "Diabetes"

# =========================
# VISUALIZATION
# =========================
def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 35], 'color': "#12c06a"},
                {'range': [35, 65], 'color': "#ffcc00"},
                {'range': [65, 100], 'color': "#ff4d4d"},
            ]
        }
    ))
    return fig

# =========================
# HEADER
# =========================
st.title("🩺 Diabetes Clinical Decision Support System")
st.caption("Machine Learning + Clinical Guidelines Integration")

# =========================
# INPUTS
# =========================
st.markdown("## 📋 Patient Data")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300)

with col2:
    bmi = st.number_input("BMI", 0.0, 70.0)
    age = st.number_input("Age", 1, 120)

with col3:
    bp = st.number_input("Diastolic BP", 0, 150)
    dpf = st.number_input("DPF", 0.0, 3.0)

# =========================
# OPTIONAL INPUTS
# =========================
st.markdown("## 🧪 Optional Clinical Inputs")

col4, col5 = st.columns(2)

with col4:
    skin = st.number_input("Skin Thickness (optional)", 0, 100, value=0)

with col5:
    insulin = st.number_input("Insulin (optional)", 0, 900, value=0)

used_defaults = False

if skin == 0:
    skin = MEDIANS["skin"]
    used_defaults = True

if insulin == 0:
    insulin = MEDIANS["insulin"]
    used_defaults = True

# =========================
# VALIDATION
# =========================
if glucose > 300 or bmi > 60:
    st.warning("⚠️ Unusual values detected. Please verify inputs.")

# =========================
# ANALYSIS
# =========================
if st.button("🔍 Analyze"):

    X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    X_scaled = scaler.transform(X)

    prob = predict_probability(X_scaled)
    risk = classify_risk(prob, glucose)

    # =========================
    # RESULT
    # =========================
    st.markdown(f"### 🎯 Risk: **{risk}**")
    st.metric("Predicted Probability", f"{prob*100:.2f}%")

    st.plotly_chart(gauge(prob), use_container_width=True)

    # =========================
    # INTERPRETATION
    # =========================
    st.markdown("## 📊 Interpretation")

    st.write(f"BMI: {bmi_category(bmi)}")
    st.write(f"Glucose: {glucose_category(glucose)}")

    # =========================
    # MEDICATION (NEW 🔥)
    # =========================
    st.markdown("## 💊 Medication Guidance")
    st.info(medication_recommendation(risk, glucose))

    # =========================
    # LIFESTYLE
    # =========================
    st.markdown("## 🥗 Lifestyle Advice")
    st.success(lifestyle_advice(risk))

    # =========================
    # CONFIDENCE EXPLANATION
    # =========================
    st.markdown("## 🧠 Model Confidence")

    st.info(f"""
Prediction Confidence: {prob*100:.2f}%

This probability is derived directly from the trained machine learning model.
No manual weighting or artificial adjustments were applied.

Confidence depends on:
• Data quality  
• Feature completeness  
• Similarity to training dataset  
""")

    # =========================
    # ASSUMPTION DISCLOSURE
    # =========================
    if used_defaults:
        st.warning("""
⚠️ Missing values detected.

Median dataset values were used:
• Skin Thickness = 20  
• Insulin = 79  

This may slightly affect prediction accuracy.
""")

    # =========================
    # DISCLAIMER
    # =========================
    st.markdown("""
---
⚠️ **Medical Disclaimer**  
This system is intended as a clinical decision support aid only and does not replace professional medical judgment.
""")
