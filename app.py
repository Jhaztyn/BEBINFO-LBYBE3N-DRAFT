# =========================
# IMPORTS
# =========================
import streamlit as st
import joblib
import numpy as np
from datetime import date
import plotly.graph_objects as go
import math

# =========================
# CONFIG
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
# BMI-STYLE GAUGE FUNCTION
# =========================
def create_pointer_gauge(prob):
    value = prob * 100

    # Convert value to angle (180° semicircle)
    angle = (value / 100) * 180
    radians = math.radians(180 - angle)

    # Needle position
    x = 0.5 + 0.4 * math.cos(radians)
    y = 0.5 + 0.4 * math.sin(radians)

    fig = go.Figure()

    # Colored zones
    fig.add_trace(go.Pie(
        values=[35, 30, 35],
        rotation=180,
        hole=0.6,
        marker=dict(colors=["green", "yellow", "red"]),
        text=["Low", "Medium", "High"],
        direction="clockwise",
        showlegend=False
    ))

    # Needle
    fig.add_shape(type="line",
                  x0=0.5, y0=0.5,
                  x1=x, y1=y,
                  line=dict(color="white", width=4))

    fig.add_shape(type="circle",
                  x0=0.48, y0=0.48,
                  x1=0.52, y1=0.52,
                  fillcolor="white",
                  line_color="white")

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Risk Level Indicator"
    )

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
    st.caption("Number of pregnancies")

    glucose = st.number_input("Glucose (mg/dL)", 0, 300)
    st.caption("Blood sugar level")

with col2:
    bmi = st.number_input("BMI", 0.0, 70.0)
    st.caption("Body Mass Index")

    age = st.number_input("Age", 1, 120)
    st.caption("Age increases risk")

with col3:
    bp = st.number_input("Diastolic BP", 0, 150)
    st.caption("Lower blood pressure")

    dpf = st.number_input("DPF (Family Risk)", 0.0, 3.0)
    st.caption("Family history influence")

# =========================
# ANALYZE
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

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

    colA, colB, colC = st.columns(3)
    colA.metric("Diagnosis", "Positive" if pred == 1 else "Negative")
    colB.metric("Probability", f"{prob:.2f}")
    colC.metric("Risk Level", risk)

    # BMI STYLE GAUGE
    fig = create_pointer_gauge(prob)
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EDUCATIONAL SECTION (LIKE YOUR REFERENCE)
    # =========================
    st.markdown("## 📌 Understanding Your Health")

    bmi_cat = get_bmi_category(bmi)
    glucose_cat = get_glucose_category(glucose)

    st.markdown(f"""
### 🧍 BMI Explanation
Your BMI is categorized as **{bmi_cat}**

BMI helps determine if your weight is healthy.
- High BMI → risk of diabetes & heart disease
- Low BMI → possible nutritional deficiency

---

### 🩸 Glucose Explanation
Your glucose level is **{glucose_cat}**

- Normal: <140  
- Prediabetes: 140–199  
- Diabetes: ≥200  

---

### ⚠️ Risk Interpretation
Your overall risk is **{risk}**

This combines:
- Blood sugar
- Body weight
- Age
- Family history

👉 Higher risk means greater likelihood of diabetes.
""")

    st.success(reco)

    # =========================
    # RISKS SECTION (LIKE YOUR IMAGE)
    # =========================
    st.markdown("## ⚠️ Health Risks")

    st.markdown("""
Possible risks associated with high diabetes risk:

- High blood pressure  
- Heart disease  
- Kidney problems  
- Nerve damage  
- Vision problems  

Maintaining a healthy lifestyle can reduce these risks.
""")

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;color:gray'>
⚠️ This tool is for educational purposes only.
</p>
""", unsafe_allow_html=True)
