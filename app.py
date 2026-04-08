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

def get_color(risk):
    if risk == "High Risk":
        return "#ff4d4d"
    elif risk == "Medium Risk":
        return "#ffcc00"
    else:
        return "#00cc66"

def get_recommendation(risk):
    if risk == "High Risk":
        return "🚨 Immediate medical consultation is strongly advised."
    elif risk == "Medium Risk":
        return "⚠️ Improve diet, exercise, and monitor regularly."
    else:
        return "✅ Maintain a healthy lifestyle."

# =========================
# WHY EXPLANATION
# =========================
def explain_result(glucose, bmi, dpf, age):
    reasons = []

    if glucose >= 180:
        reasons.append("High blood sugar level")
    if bmi < 18.5:
        reasons.append("Low body weight (underweight)")
    if bmi > 30:
        reasons.append("High body weight (obesity)")
    if dpf > 0.8:
        reasons.append("Strong family history of diabetes")
    if age > 45:
        reasons.append("Age-related risk factor")

    return reasons

# =========================
# GAUGE
# =========================
def create_gauge(prob):
    value = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 42}},
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 35], 'color': "#00cc66"},
                {'range': [35, 65], 'color': "#ffcc00"},
                {'range': [65, 100], 'color': "#ff4d4d"},
            ],
        }
    ))

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
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
    bmi = st.number_input("BMI", 0.0, 70.0)
    age = st.number_input("Age", 1, 120)

with col3:
    bp = st.number_input("Diastolic BP", 0, 150)
    dpf = st.number_input("DPF", 0.0, 3.0)

# =========================
# ANALYZE
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, 0, 0, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    risk = get_risk_level(prob, glucose)
    color = get_color(risk)
    reco = get_recommendation(risk)

    # =========================
    # RESULT CARD (MAIN FOCUS)
    # =========================
    st.markdown(f"""
    <div style="padding:30px;border-radius:15px;background:{color};text-align:center;color:black">
        <h1>{risk}</h1>
        <h2>{prob*100:.1f}% Risk Probability</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 Risk Visualization")
    st.plotly_chart(create_gauge(prob), use_container_width=True)

    # =========================
    # WHY SECTION
    # =========================
    st.markdown("## 🧠 Why this result?")

    reasons = explain_result(glucose, bmi, dpf, age)

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("No strong risk factors detected.")

    # =========================
    # RECOMMENDATION
    # =========================
    st.markdown("## 💡 Recommendation")
    st.success(reco)

    # =========================
    # EDUCATIONAL SECTION
    # =========================
    st.markdown("## 📌 Understanding Your Health")

    st.info("""
    • BMI indicates body fat level  
    • Glucose shows blood sugar levels  
    • Higher values increase diabetes risk  
    """)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;color:gray'>
⚠️ For educational use only. Not medical advice.
</p>
""", unsafe_allow_html=True)
