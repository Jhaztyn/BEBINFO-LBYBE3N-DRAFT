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

# Optional calibrator (must be trained properly)
try:
    calibrator = joblib.load("calibrator.pkl")
    USE_CALIBRATOR = True
except:
    USE_CALIBRATOR = False

# =========================
# CONSTANTS
# =========================
LOW_THRESH = 0.35
HIGH_THRESH = 0.65

MEDIANS = {
    "skin": 20,
    "insulin": 79
}

# =========================
# MODEL PREDICTION
# =========================
def predict_probability(X_scaled):
    if USE_CALIBRATOR:
        return calibrator.predict_proba(X_scaled)[0][1]
    return model.predict_proba(X_scaled)[0][1]

# =========================
# CLINICAL LOGIC
# =========================
def get_risk_level(prob, glucose):
    if glucose >= 200:
        return "High Risk (Clinical Override)"
    elif prob >= HIGH_THRESH:
        return "High Risk"
    elif prob >= LOW_THRESH:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_color(risk):
    if "High" in risk:
        return "#ff4d4d"
    elif "Medium" in risk:
        return "#ffcc00"
    else:
        return "#12c06a"

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
    if "High" in risk:
        return "🚨 Immediate medical consultation is strongly advised."
    elif risk == "Medium Risk":
        return "⚠️ Improve diet, exercise, and monitor regularly."
    else:
        return "✅ Maintain a healthy lifestyle."

def medication_recommendation(risk, glucose):
    if glucose >= 200:
        return "🔴 Possible Diabetes → Refer for insulin therapy evaluation"
    elif "High" in risk:
        return "🟠 Consider Metformin (first-line) with physician guidance"
    elif risk == "Medium Risk":
        return "🟡 Lifestyle modification ± Metformin (case-dependent)"
    else:
        return "🟢 No medication required; preventive care advised"

# =========================
# GAUGE (KEEP YOUR STYLE)
# =========================
def create_gauge(prob):
    value = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 52}},
        gauge={
            'shape': "angular",
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
    return fig

# =========================
# HEADER (KEEP YOUR UI)
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
# INPUTS (YOUR UI STYLE)
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
# OPTIONAL INPUTS (MATCH YOUR UI)
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
# ANALYZE
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

    X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    X_scaled = scaler.transform(X)

    prob = predict_probability(X_scaled)
    risk = get_risk_level(prob, glucose)
    color = get_color(risk)

    # =========================
    # RESULT CARD (KEEP EXACT STYLE)
    # =========================
    st.markdown(f"""
    <div style="padding:30px;border-radius:15px;background:{color};text-align:center">
        <h1>{risk}</h1>
        <h2>{prob*100:.1f}% Risk Probability</h2>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # GAUGE
    # =========================
    st.markdown("## 📊 Risk Visualization")
    st.plotly_chart(create_gauge(prob), use_container_width=True)

    # =========================
    # DISCUSSION (UNCHANGED STYLE)
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
""")

    st.success(get_recommendation(risk))

    # =========================
    # MEDICATION (ADDED 🔥)
    # =========================
    st.markdown("## 💊 Medication Guidance")
    st.info(medication_recommendation(risk, glucose))

    # =========================
    # DISCLOSURE
    # =========================
    if used_defaults:
        st.warning("""
⚠️ Missing values detected.

Default dataset medians used:
• Skin Thickness = 20  
• Insulin = 79  
""")

    # =========================
    # DISCLAIMER
    # =========================
    st.markdown("""
<div style="padding:15px;border-radius:10px;background-color:#fff3cd;color:#856404">
⚠️ <strong>Medical Disclaimer:</strong><br><br>
This system is an aid only and NOT a substitute for professional medical advice.
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
