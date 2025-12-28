import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ================== 1. إعدادات الصفحة (Design) ==================
st.set_page_config(page_title="AI Promotion Predictor Pro", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; padding: 15px;
        border-radius: 12px; font-weight: bold; font-size: 1.2rem;
    }
    label { font-size: 1.1rem !important; font-weight: 600 !important; color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ================== 2. تحميل الموديل الحقيقي ==================
@st.cache_resource
def load_assets():
    try:
        # تأكد أن الملفات دي موجودة في نفس الفولدر على GitHub
        model = xgb.Booster()
        model.load_model('employee_promotion_model.json')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_assets()

# ================== 3. واجهة المدخلات (Modern Layout) ==================
st.markdown("<h1 style='text-align:center;'>🚀 AI PROMOTION PREDICTOR</h1>", unsafe_allow_html=True)
st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    department = st.selectbox("Department", ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'Procurement', 'HR', 'Finance', 'Legal', 'R&D'])
    education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary"])
    gender = st.selectbox("Gender", ['m', 'f'])

with col2:
    region = st.selectbox("Region", [f'region_{i}' for i in range(1, 35)])
    recruitment_channel = st.selectbox("Channel", ['sourcing', 'referred', 'other'])
    kpis_met = st.selectbox("KPIs Met >80%", ['Yes', 'No'])

with col3:
    age = st.slider("Age", 20, 60, 30)
    length_of_service = st.slider("Service Years", 1, 37, 5)
    previous_year_rating = st.select_slider("Rating", options=[1.0, 2.0, 3.0, 4.0, 5.0], value=3.0)

c4, c5 = st.columns(2)
with c4:
    avg_training_score = st.slider("Training Score", 40, 99, 60)
with c5:
    no_of_trainings = st.number_input("No. of Trainings", 1, 10, 1)
    awards_won = st.radio("Awards Won?", ['Yes', 'No'], horizontal=True)

# ================== 4. معالجة البيانات ==================
# الترتيب ده هو اللي الموديل مستنيه بالظبط
features_order = [
    'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
    'avg_training_score', 'high_training_score', 'has_awards', 
    'long_service_high_rating', 'department', 'region', 'education', 
    'gender', 'recruitment_channel', 'age_group'
]

# Feature Engineering
has_awards = 1 if awards_won == 'Yes' else 0
high_training_score = 1 if avg_training_score > 80 else 0
long_service_high_rating = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0

if age < 30: age_group = '<30'
elif age <= 40: age_group = '30-40'
elif age <= 50: age_group = '40-50'
else: age_group = '>50'

input_df = pd.DataFrame([[
    no_of_trainings, age, previous_year_rating, length_of_service, 
    avg_training_score, high_training_score, has_awards, 
    long_service_high_rating, department, region, education, 
    gender, recruitment_channel, age_group
]], columns=features_order)

for col in ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']:
    input_df[col] = input_df[col].astype("category")

# ================== 5. التحليل والعرض ==================
if st.button("🚀 RUN ANALYSIS"):
    if model is not None:
        with st.spinner("Calculating probability..."):
            dmat = xgb.DMatrix(input_df, enable_categorical=True)
            prob = model.predict(dmat)[0]
        
        # كارت النتيجة (الـ GUI الفاشخ)
        color = "#2ecc71" if prob > 0.5 else "#e74c3c"
        status = "HIGH PROMOTION POTENTIAL" if prob > 0.5 else "PROMOTION UNLIKELY"
        icon = "🚀" if prob > 0.5 else "⚠️"
        msg = "Employee strongly meets promotion criteria." if prob > 0.5 else "Current indicators are below the threshold."

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)); 
                    border:2px solid {color}; border-radius:18px; padding:32px; text-align:center;">
            <div style="font-size:3.5rem;">{icon}</div>
            <h1 style="color:{color}; font-weight:900;">{status}</h1>
            <div style="font-size:3rem; font-weight:900; color:white; margin:10px 0;">{prob*100:.1f}%</div>
            
            <div style="height:12px; background:rgba(255,255,255,0.1); border-radius:10px; margin:20px 0;">
                <div style="width:{prob*100}%; height:100%; background:{color}; border-radius:10px;"></div>
            </div>
            <p style="color:#94a3b8; font-size:1.1rem;">{msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if prob > 0.5: st.balloons()
    else:
        st.error("Model not loaded properly. Check the file path.")
