# =========================
# IMPORTS
# =========================
import streamlit as st
import joblib
import numpy as np
from datetime import date
import plotly.graph_objects as go

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# =========================
# FIXED THRESHOLDS
# =========================
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

def get_confidence(prob):
    if prob >= 0.8 or prob <= 0.2:
        return "High Confidence"
    elif prob >= 0.6 or prob <= 0.4:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

def get_recommendation(risk, glucose, bmi):
    if risk == "High Risk":
        if glucose >= 200:
            return "🚨 Severe hyperglycemia detected. Immediate physician evaluation required."
        elif bmi >= 30:
            return "⚠️ Obesity-related risk. Clinical intervention recommended."
        else:
            return "⚠️ High risk detected. Further diagnostic testing required."
    elif risk == "Medium Risk":
        return "Lifestyle modification and monitoring recommended."
    else:
        return "Maintain healthy lifestyle."

def validate_inputs(glucose, bmi, age):
    if glucose <= 0 or bmi <= 0:
        return False, "Glucose and BMI must be greater than 0."
    return True, ""

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

# =========================
# PDF
# =========================
def generate_pdf(name, sex, date, glucose, bmi, age, prob, risk, reco):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Diabetes Clinical Decision Report", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Sex: {sex}", styles['Normal']))
    content.append(Paragraph(f"Date: {date}", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Probability: {prob:.2f}", styles['Normal']))
    content.append(Paragraph(f"Risk Level: {risk}", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph(reco, styles['Normal']))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>🩺 Diabetes Clinical Decision Support System</h1>
<p style='text-align:center;color:gray;'>AI-Powered Clinical Risk Assessment</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# INPUT CARD
# =========================
st.markdown("### 📋 Patient Information")

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

name = st.text_input("Patient Name")
sex = st.selectbox("Sex", ["Female", "Male"])
visit_date = st.date_input("Date", value=date.today())

st.divider()

# =========================
# BUTTON
# =========================
if st.button("🔍 Analyze Patient Risk", use_container_width=True):

    valid, msg = validate_inputs(glucose, bmi, age)
    if not valid:
        st.error(msg)
        st.stop()

    with st.spinner("Analyzing patient data..."):
        input_data = np.array([[preg, glucose, bp, 0, 0, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        risk = get_risk_level(prob, glucose)
        confidence = get_confidence(prob)
        reco = get_recommendation(risk, glucose, bmi)

    # =========================
    # RESULTS
    # =========================
    st.markdown("## 📊 Results")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Diagnosis", "Positive" if pred == 1 else "Negative")
    colB.metric("Probability", f"{prob:.2f}")
    colC.metric("Risk Level", risk)
    colD.metric("Confidence", confidence)

    # =========================
    # RISK COLOR
    # =========================
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
                {'range': [0, LOW_THRESH * 100], 'color': "green"},
                {'range': [LOW_THRESH * 100, HIGH_THRESH * 100], 'color': "yellow"},
                {'range': [HIGH_THRESH * 100, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # INTERPRETATION
    # =========================
    st.markdown("## 📌 Interpretation")

    st.write("**BMI:**", get_bmi_category(bmi))
    st.write("**Glucose:**", get_glucose_category(glucose))

    st.markdown(f"""
<div style="padding:20px;border-radius:10px;background:#3b3b0a;color:white">
{reco}
</div>
""", unsafe_allow_html=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    if HAS_MATPLOTLIB and hasattr(model, "feature_importances_"):
        st.markdown("## 📊 Model Insights")

        features = ["Preg", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]
        importances = model.feature_importances_

        fig2, ax = plt.subplots()
        ax.barh(features, importances)
        st.pyplot(fig2)

    # =========================
    # PDF
    # =========================
    pdf_data = generate_pdf(name, sex, visit_date, glucose, bmi, age, prob, risk, reco)

    st.download_button(
        label="📄 Download Clinical Report",
        data=pdf_data,
        file_name="report.pdf",
        mime="application/pdf"
    )

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style='text-align:center;color:gray;font-size:12px'>
⚠️ For decision support only. Not a substitute for medical professionals.
</p>
""", unsafe_allow_html=True)
