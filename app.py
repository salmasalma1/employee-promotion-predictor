import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ================== Page Config ==================
st.set_page_config(
    page_title="AI HR Analytics Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS Styling ==================
st.markdown("""
<style>

/* ===== Global ===== */
.stApp {
    background: linear-gradient(160deg, #0d1117, #161b22);
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* ===== Titles ===== */
h1, h2, h3 {
    letter-spacing: 1px;
}

/* ===== Labels ===== */
label {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #f0f6fc !important;
}

/* ===== Input Fields ===== */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background-color: #0d1117 !important;
    border-radius: 10px;
    font-size: 1rem;
}

/* ===== Columns as Cards ===== */
div[data-testid="column"] {
    background-color: #161b22;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #30363d;
}

/* ===== Button ===== */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff6a00, #ff8c00);
    color: white;
    border: none;
    padding: 18px;
    border-radius: 14px;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 30px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(255,140,0,0.35);
}

/* ===== Prediction Result ===== */
.prediction-result {
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    margin-top: 35px;
    backdrop-filter: blur(8px);
}

</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("""
<div style="text-align:center; margin-bottom:40px;">
    <h1 style="font-size:3.2rem; font-weight:900;">AI Promotion Predictor</h1>
    <p style="color:#8b949e; font-size:1.1rem;">
        Enterprise-Grade HR Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ================== Load Model ==================
@st.cache_resource
def load_hr_engine():
    model = xgb.Booster()
    model.load_model("employee_promotion_model.json")
    return model

model = load_hr_engine()

# ================== Input Section ==================
st.markdown("<h2 style='text-align:center;'>Employee Data Profile</h2>", unsafe_allow_html=True)
st.write("")

# -------- Row 1 --------
c1, c2, c3 = st.columns(3)

with c1:
    department = st.selectbox(
        "Department",
        ['Analytics', 'Sales & Marketing', 'Operations', 'Technology',
         'Procurement', 'HR', 'Finance', 'Legal', 'R&D']
    )
    education = st.selectbox(
        "Education Level",
        ["Bachelor's", "Master's & above", "Below Secondary"]
    )

with c2:
    region = st.selectbox(
        "Region",
        [f"region_{i}" for i in range(1, 35)]
    )
    gender = st.selectbox(
        "Gender",
        ['m', 'f']
    )

with c3:
    recruitment_channel = st.selectbox(
        "Recruitment Channel",
        ['referred', 'sourcing', 'other']
    )
    kpis_met = st.selectbox(
        "KPIs Met Above 80%",
        ['Yes', 'No']
    )

st.write("")

# -------- Row 2 --------
c4, c5, c6 = st.columns(3)

with c4:
    age = st.slider("Employee Age", 20, 60, 30)
    no_of_trainings = st.number_input(
        "Number of Trainings Completed",
        1, 10, 1
    )

with c5:
    length_of_service = st.slider("Years of Service", 1, 37, 5)
    previous_year_rating = st.select_slider(
        "Previous Year Rating",
        options=[1.0, 2.0, 3.0, 4.0, 5.0],
        value=3.0
    )

with c6:
    avg_training_score = st.slider("Average Training Score", 40, 99, 60)
    awards_won = st.radio(
        "Awards Won",
        ['Yes', 'No'],
        horizontal=True
    )

# ================== Feature Engineering ==================
features_order = [
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'avg_training_score',
    'high_training_score', 'has_awards',
    'long_service_high_rating', 'department',
    'region', 'education',
