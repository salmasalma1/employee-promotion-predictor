import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb
import os

# --- 1. إعدادات الصفحة والتصميم (UI Enhancements) --- 
st.set_page_config(page_title="Employee Promotion Prediction", layout="centered")

# CSS لتجميل الواجهة لتشبه الصور التي أرفقتها
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: white;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("👔 Employee Promotion Prediction")
st.markdown("<h4 style='text-align: center; color: #888;'>XGBoost model trained on 300,000 HR records</h4>", unsafe_allow_html=True)
st.write("---")

# --- 2. تحميل الموديل والملفات ---
@st.cache_resource
def load_model_artifacts():
    model_path = 'employee_promotion_model.json'
    try:
        # تحميل السكيلر وقائمة الأعمدة الأصلية (مهم جداً للترتيب)
        scaler = joblib.load('scaler.pkl')
        # الموديل يتم تحميله كـ Booster للتعامل مع الـ mismatch
        model = xgb.Booster()
        model.load_model(model_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler = load_model_artifacts()

# --- 3. تصميم واجهة المدخلات (Grid System) ---
# تقسيم الشاشة لعمودين زي الصور
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", ['Analytics', 'Sales & Marketing', 'Operations', 'Technology', 'Procurement', 'HR', 'Finance', 'R&D', 'Legal'])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's & above", "Below Secondary"])
    gender = st.selectbox("Gender", ['m', 'f'])
    recruitment_channel = st.selectbox("Recruitment Channel", ['referred', 'sourcing', 'other'])
    no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=10, value=1)
    kpis_met = st.selectbox("KPIs_met >80%?", ['Yes', 'No'])

with col2:
    region = st.selectbox("Region", [f'region_{i}' for i in range(1, 35)])
    age = st.slider("Age", 20, 60, 30)
    length_of_service = st.slider("Years of Service", 1, 37, 5)
    previous_year_rating = st.selectbox("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0])
    awards_won = st.selectbox("Awards Won?", ['Yes', 'No'])
    avg_training_score = st.slider("Average Training Score", 40, 99, 60)

# --- 4. معالجة البيانات لحل مشكلة الـ Feature Mismatch ---
# الأيرور بيقول إن الموديل مستني الأعمدة دي بالظبط وبالترتيب ده:
expected_features = [
    'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
    'avg_training_score', 'high_training_score', 'has_awards', 
    'long_service_high_rating', 'department', 'region', 'education', 
    'gender', 'recruitment_channel', 'age_group'
]

# تحويل المدخلات لقيم رقمية للميزات التفاعلية
has_awards = 1 if awards_won == 'Yes' else 0
high_training_score = 1 if avg_training_score > 80 else 0
long_service_high_rating = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0

# تحديد الـ Age Group
if age < 30: age_group = '<30'
elif age <= 40: age_group = '30-40'
elif age <= 50: age_group = '40-50'
else: age_group = '>50'

# تجهيز الـ DataFrame بنفس ترتيب الموديل بالظبط
input_data = pd.DataFrame([[
    no_of_trainings, age, previous_year_rating, length_of_service, 
    avg_training_score, high_training_score, has_awards, 
    long_service_high_rating, department, region, education, 
    gender, recruitment_channel, age_group
]], columns=expected_features)

# --- تحويل الأعمدة النصية لـ Category (هذا هو سر حل الأيرور) ---
# الموديل بتاعك متدرب على Category Data مش One-Hot
for col in ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']:
    input_data[col] = input_data[col].astype('category')

# --- 5. زر التوقع والنتيجة ---
if st.button("🔮 Predict Promotion"):
    try:
        # تحويل البيانات لـ DMatrix (الطريقة الأضمن لـ XGBoost Booster)
        dmatrix_input = xgb.DMatrix(input_data, enable_categorical=True)
        
        # التوقع
        prob = model.predict(dmatrix_input)[0]
        
        # عرض النتيجة بشكل احترافي
        st.write("---")
        if prob > 0.5:
            st.markdown(f"""
                <div class='prediction-box' style='background-color: #1e3d24; border: 1px solid #2ecc71;'>
                    <h2 style='color: #2ecc71;'>Promoted! 🚀</h2>
                    <h1 style='color: white;'>Probability: {prob*100:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='prediction-box' style='background-color: #3d1e1e; border: 1px solid #e74c3c;'>
                    <h2 style='color: #e74c3c;'>Not Promoted 😔</h2>
                    <h1 style='color: white;'>Promotion Probability: {prob*100:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if the categorical features in training match the input labels.")
