# =========================
# IMPORTS
# =========================
import streamlit as st
import joblib
import numpy as np
from datetime import date
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Diabetes CDSS", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

LOW_THRESH = 0.35
HIGH_THRESH = 0.65

# =========================
# FUNCTIONS
# =========================
def get_risk_level(prob, glucose):
    if prob >= HIGH_THRESH or glucose >= 200:
        return "High Risk"
    elif prob >= LOW_THRESH:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 23: return "Normal"
    elif bmi < 25: return "Overweight"
    elif bmi < 30: return "Obese I"
    else: return "Obese II"

def get_glucose_category(glucose):
    if glucose < 140: return "Normal"
    elif glucose < 200: return "Prediabetes"
    else: return "Diabetes"

def get_recommendation(risk):
    if risk == "High Risk":
        return "🚨 You are at high risk. Please consult a doctor immediately."
    elif risk == "Medium Risk":
        return "⚠️ You may be at risk. Consider improving diet and lifestyle."
    else:
        return "✅ Your results look healthy. Maintain a good lifestyle."

# =========================
# HEADER (GRADIENT)
# =========================
st.markdown("""
<div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
padding:25px;border-radius:12px;margin-bottom:20px">
<h1 style="color:white;text-align:center;">🩺 Diabetes Clinical Decision Support System</h1>
<p style="color:white;text-align:center;">AI-Powered Health Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# =========================
# PATIENT INFO (TOP)
# =========================
colA, colB = st.columns([3,1])

with colA:
    name = st.text_input("👤 Patient Name")
    sex = st.selectbox("⚧ Sex", ["Male", "Female"])

with colB:
    st.markdown("### 📅 Date")
    st.info(date.today())

st.divider()

# =========================
# INPUT SECTION
# =========================
st.markdown("## 📋 Health Information")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20)
    st.caption("Number of times pregnant")

    glucose = st.number_input("Glucose (mg/dL)", 0, 300)
    st.caption("Blood sugar level")

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0)
    st.caption("Body fat based on height and weight")

    age = st.number_input("Age", 1, 120)
    st.caption("Risk increases with age")

with col3:
    bp = st.number_input("Diastolic Blood Pressure", 0, 150)
    st.caption("Lower blood pressure value")

    dpf = st.number_input("DPF (Genetic Risk Score)", 0.0, 3.0)
    st.caption("Family history of diabetes")

st.divider()

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

    with st.spinner("Analyzing patient data..."):

        input_data = np.array([[preg, glucose, bp, 0, 0, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        risk = get_risk_level(prob, glucose)
        reco = get_recommendation(risk)

    # =========================
    # RESULTS
    # =========================
    st.markdown("## 📊 Results")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Diagnosis", "Positive" if pred == 1 else "Negative")
    colB.metric("Probability", f"{prob:.2f}")
    colC.metric("Risk Level", risk)
    colD.metric("Confidence", "High" if prob > 0.75 else "Moderate")

    # Risk Highlight
    if risk == "High Risk":
        st.markdown("<h2 style='color:#ff4d4d;'>🔴 HIGH RISK</h2>", unsafe_allow_html=True)
    elif risk == "Medium Risk":
        st.markdown("<h2 style='color:#ffcc00;'>🟡 MEDIUM RISK</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:#00cc66;'>🟢 LOW RISK</h2>", unsafe_allow_html=True)

    # =========================
    # GAUGE
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 35], 'color': "green"},
                {'range': [35, 65], 'color': "yellow"},
                {'range': [65, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # INTERPRETATION (BMI STYLE)
    # =========================
    st.markdown("## 📌 Understanding Your Results")

    bmi_cat = get_bmi_category(bmi)
    glucose_cat = get_glucose_category(glucose)

    st.markdown(f"""
### 🧍 Body Mass Index (BMI)
**Category:** {bmi_cat}

BMI estimates body fat.
- Normal: 18.5–25  
- High BMI increases risk of diabetes and heart disease.

---

### 🩸 Blood Glucose
**Status:** {glucose_cat}

- Normal: <140  
- Prediabetes: 140–199  
- Diabetes: 200+  

---

### ⚠️ Diabetes Risk
**Overall Risk:** {risk}

This combines:
- Blood sugar
- Body weight
- Age
- Family history
""")

    st.success(reco)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;color:gray'>
⚠️ For educational use only. Not a substitute for professional medical advice.
</p>
""", unsafe_allow_html=True)
