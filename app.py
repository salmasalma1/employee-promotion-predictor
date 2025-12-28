import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb
import os

# --- 1. إعدادات الصفحة والتصميم الفاشخ --- 
st.set_page_config(page_title="AI HR Analytics Pro", layout="wide", initial_sidebar_state="collapsed")

# CSS Customization for a Premium Look
st.markdown("""
    <style>
    /* تغيير الخلفية العامة */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
        color: #e6edf3;
    }
    
    /* تنسيق الحاويات (Cards) */
    .input-card {
        background-color: #1c2128;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    
    /* تنسيق العناوين */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* زر التوقع (The "Hero" Button) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7575 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* كارت النتيجة النهائي */
    .prediction-result {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid;
        margin-top: 30px;
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3rem; margin-bottom: 0;'>🚀 AI Promotion Predictor</h1>
        <p style='color: #8b949e; font-size: 1.1rem;'>Advanced HR Analytics Engine • XGBoost v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. تحميل الموديل ---
@st.cache_resource
def load_hr_engine():
    try:
        model = xgb.Booster()
        model.load_model('employee_promotion_model.json')
        return model
    except Exception as e:
        st.error(f"Engine Error: {e}")
        st.stop()

model = load_hr_engine()

# --- 3. قسم المدخلات (Modern Dashboard Grid) ---
st.markdown("### 📋 Employee Profile Details")

with st.container():
    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1:
        department = st.selectbox("🏢 Department", ['Analytics', 'Sales & Marketing', 'Operations', 'Technology', 'Procurement', 'HR', 'Finance', 'Legal', 'R&D'])
        education = st.selectbox("🎓 Education", ["Bachelor's", "Master's & above", "Below Secondary"])
    with c2:
        region = st.selectbox("🌍 Region", [f'region_{i}' for i in range(1, 35)])
        gender = st.selectbox("👤 Gender", ['m', 'f'])
    with c3:
        recruitment_channel = st.selectbox("🔗 Recruitment Channel", ['referred', 'sourcing', 'other'])
        kpis_met = st.selectbox("🎯 KPIs Met >80%?", ['Yes', 'No'])

    # Row 2
    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    with c4:
        age = st.slider("🎂 Age", 20, 60, 30)
        no_of_trainings = st.number_input("📚 Number of Trainings", 1, 10, 1)
    with c5:
        length_of_service = st.slider("⏳ Years of Service", 1, 37, 5)
        previous_year_rating = st.select_slider("⭐️ Previous Rating", options=[1.0, 2.0, 3.0, 4.0, 5.0], value=3.0)
    with c6:
        avg_training_score = st.slider("📈 Training Score", 40, 99, 60)
        awards_won = st.radio("🏆 Award Winner?", ['Yes', 'No'], horizontal=True)

# --- 4. Logic & Processing ---
# نفس الترتيب اللي بيضمن دقة التوقع
features_order = [
    'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
    'avg_training_score', 'high_training_score', 'has_awards', 
    'long_service_high_rating', 'department', 'region', 'education', 
    'gender', 'recruitment_channel', 'age_group'
]

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
    input_df[col] = input_df[col].astype('category')

# --- 5. زر التوقع والنتيجة الـ "فاشخة" ---
st.write("") # Spacer
if st.button("RUN PREDICTION ANALYSIS"):
    try:
        dmat = xgb.DMatrix(input_df, enable_categorical=True)
        prob = model.predict(dmat)[0]
        
        if prob > 0.5:
            st.markdown(f"""
                <div class="prediction-result" style="background: rgba(46, 204, 113, 0.1); border-color: #2ecc71;">
                    <h1 style="color: #2ecc71; margin-bottom:0;">HIGH PROMOTION POTENTIAL 🎉</h1>
                    <p style="color: #e6edf3; font-size: 1.5rem;">Confidence Score: <b>{prob*100:.1f}%</b></p>
                    <p style="color: #8b949e;">This employee exhibits strong performance indicators.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="prediction-result" style="background: rgba(231, 76, 60, 0.1); border-color: #e74c3c;">
                    <h1 style="color: #e74c3c; margin-bottom:0;">PROMOTION UNLIKELY 📉</h1>
                    <p style="color: #e6edf3; font-size: 1.5rem;">Promotion Probability: <b>{prob*100:.1f}%</b></p>
                    <p style="color: #8b949e;">Criteria for promotion are not fully met at this stage.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Critical System Error: {e}")
