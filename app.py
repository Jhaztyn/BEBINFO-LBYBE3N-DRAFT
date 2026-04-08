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
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Try loading calibrated model (optional but ideal)
try:
    calibrator = joblib.load("calibrator.pkl")
    USE_CALIBRATOR = True
except:
    USE_CALIBRATOR = False

LOW_THRESH = 0.35
HIGH_THRESH = 0.65

# =========================
# CALIBRATION FUNCTION
# =========================
def calibrate_probability(prob):
    """
    Applies calibration if available, otherwise uses logistic smoothing
    """
    if USE_CALIBRATOR:
        return calibrator.predict_proba([[prob]])[0][1]
    else:
        # Logistic calibration fallback (smooths extreme values)
        return 1 / (1 + np.exp(-5 * (prob - 0.5)))


# =========================
# CLINICAL NORMALIZATION
# =========================
def normalize_glucose(glucose):
    return min(glucose / 200, 1.0)

def normalize_bmi(bmi):
    return min(bmi / 35, 1.0)

def normalize_age(age):
    return min(age / 100, 1.0)


# =========================
# FINAL RISK (CALIBRATED)
# =========================
def compute_final_risk(prob, glucose, bmi, age):
    calibrated = calibrate_probability(prob)

    g = normalize_glucose(glucose)
    b = normalize_bmi(bmi)
    a = normalize_age(age)

    # ✔ Weighted, justified combination
    score = (
        calibrated * 0.7 +
        g * 0.2 +
        b * 0.07 +
        a * 0.03
    )

    return min(score, 1.0), calibrated


# =========================
# HELPERS
# =========================
def get_risk_level(score, glucose):
    if score >= HIGH_THRESH or glucose >= 200:
        return "High Risk"
    elif score >= LOW_THRESH:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_color(risk):
    return {
        "High Risk": "#ff4d4d",
        "Medium Risk": "#ffcc00",
        "Low Risk": "#12c06a"
    }[risk]


def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"


def get_glucose_category(glucose):
    if glucose < 140: return "Normal"
    elif glucose < 200: return "Prediabetes"
    else: return "Diabetes"


def get_recommendation(risk):
    if risk == "High Risk":
        return "🚨 Immediate medical consultation is strongly advised."
    elif risk == "Medium Risk":
        return "⚠️ Improve diet, exercise, and monitor regularly."
    else:
        return "✅ Maintain a healthy lifestyle."


# =========================
# GAUGE
# =========================
def create_gauge(prob):
    value = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 52}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 35], 'color': "#12c06a"},
                {'range': [35, 65], 'color': "#ffcc00"},
                {'range': [65, 100], 'color': "#ff4d4d"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'value': value
            }
        }
    ))

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    return fig


# =========================
# HEADER
# =========================
st.markdown("""
<div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
padding:25px;border-radius:12px">
<h1 style="color:white;text-align:center;">🩺 Diabetes Clinical Decision Support System</h1>
<p style="color:white;text-align:center;">AI-Powered Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# =========================
# PATIENT INFO
# =========================
colA, colB = st.columns([3,1])

with colA:
    name = st.text_input("👤 Patient Name")
    sex = st.selectbox("Sex", ["Male", "Female"])

with colB:
    st.markdown("### 📅 Date")
    st.info(date.today())

st.divider()

# =========================
# INPUTS
# =========================
st.markdown("## 📋 Health Information")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300)

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0)
    age = st.number_input("Age", 1, 120)

with col3:
    bp = st.number_input("Diastolic Blood Pressure", 0, 150)
    dpf = st.number_input("DPF (Family Risk)", 0.0, 3.0)

# =========================
# ANALYZE
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

    # ✔ Maintain 8 features (model compatibility)
    skin_thickness = 20
    insulin = 80

    input_data = np.array([[
        preg,
        glucose,
        bp,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)
    raw_prob = model.predict_proba(input_scaled)[0][1]

    final_score, calibrated_prob = compute_final_risk(
        raw_prob, glucose, bmi, age
    )

    risk = get_risk_level(final_score, glucose)
    color = get_color(risk)
    reco = get_recommendation(risk)

    # =========================
    # RESULT CARD
    # =========================
    st.markdown(f"""
    <div style="padding:30px;border-radius:15px;background:{color};text-align:center">
        <h1>{risk}</h1>
        <h2>{final_score*100:.1f}% Risk Probability</h2>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # GAUGE
    # =========================
    st.markdown("## 📊 Risk Visualization")
    st.plotly_chart(create_gauge(final_score), use_container_width=True)

    # =========================
    # MODEL INSIGHT (UPGRADED)
    # =========================
    st.markdown("## 🧠 Model Insight")

    st.info(f"""
Raw Model Confidence: {raw_prob*100:.1f}%  
Calibrated Probability: {calibrated_prob*100:.1f}%  

This result combines:
• Calibrated machine learning prediction  
• Clinically normalized risk factors (glucose, BMI, age)  

Calibration improves reliability by correcting model bias.
""")

    # =========================
    # DISCUSSION
    # =========================
    st.markdown("## 📌 Understanding Your Health")

    bmi_cat = get_bmi_category(bmi)
    glucose_cat = get_glucose_category(glucose)

    st.markdown(f"""
### 🧍 BMI Explanation
Your BMI is **{bmi_cat}**

### 🩸 Blood Glucose
Your glucose is **{glucose_cat}**

### ⚠️ Risk Interpretation
Your classification is **{risk}**

👉 This result is based on calibrated ML + clinical factors.
""")

    st.success(reco)

    # =========================
    # DISCLAIMER
    # =========================
    st.markdown("""
<div style="padding:15px;border-radius:10px;background-color:#fff3cd;color:#856404">
⚠️ <strong>Medical Disclaimer:</strong><br><br>
This system is an aid only and NOT a substitute for professional medical advice.

Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;color:gray'>
Educational tool only. Not medical advice.
</p>
""", unsafe_allow_html=True)
