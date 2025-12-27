import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler

# أقوى theme و layout
st.set_page_config(page_title="Employee Promotion Predictor", page_icon="👔", layout="centered")

# Custom CSS لـ GUI أقوى
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        padding: 2rem;
    }
    .stApp {
        background-color: #0e1117;
    }
    .title {
        font-size: 3rem;
        color: #fa7343;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #8a8a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction {
        font-size: 2.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    .success {
        background-color: #1f4e3d;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #00ff9d;
    }
    .warning {
        background-color: #5e2a2a;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #ff6b6b;
    }
    .slider-label {
        color: #fa7343;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# تحميل الملفات
@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model('employee_promotion_model.json')  # غيري الاسم لو مختلف
    model = xgb.XGBClassifier()
    model._Booster = booster
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_feature_columns():
    with open('feature_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return columns

model = load_model()
scaler = load_scaler()
required_columns = load_feature_columns()

# Title أقوى
st.markdown("<h1 class='title'>👔 Employee Promotion Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>XGBoost model trained on 300k HR records with advanced features</p>", unsafe_allow_html=True)

# Layout أقوى بـ columns و icons
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Info")
    department = st.selectbox("Department 🏢", ['Sales & Marketing', 'Operations', 'Procurement', 'Technology', 'Finance', 'Analytics', 'R&D', 'HR', 'Legal', 'Other'])
    region = st.selectbox("Region 🌍", ['region_2', 'region_22', 'region_7', 'region_15', 'region_13', 'region_4', 'region_26', 'region_16', 'region_27', 'region_10', 'Other'])
    education = st.selectbox("Education Level 🎓", ["Below Secondary", "Bachelor's", "Master's & above", 'Other'])
    gender = st.selectbox("Gender ⚥", ['f', 'm'])
    recruitment_channel = st.selectbox("Recruitment Channel 📩", ['sourcing', 'other', 'referred'])
    kpis_met = st.selectbox("KPIs_met >80%? 📊", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    st.markdown("### 📈 Performance & Experience")
    no_of_trainings = st.slider("Number of Trainings 📚", 1, 10, 1)
    age = st.slider("Age 🎂", 18, 60, 35)
    length_of_service = st.slider("Years of Service ⏳", 1, 37, 5)
    previous_year_rating = st.slider("Previous Year Rating ⭐", 1.0, 5.0, 3.0, step=0.1)
    awards_won = st.selectbox("Awards Won? 🏆", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    avg_training_score = st.slider("Average Training Score 📝", 39, 99, 75)

# Button أقوى
if st.button("🔮 Predict Promotion", type="primary", use_container_width=True):
    with st.spinner("Analyzing employee data..."):
        data = {
            'department': department,
            'region': region,
            'education': education,
            'gender': gender,
            'recruitment_channel': recruitment_channel,
            'no_of_trainings': no_of_trainings,
            'age': age,
            'previous_year_rating': previous_year_rating,
            'length_of_service': length_of_service,
            'awards_won': awards_won,
            'avg_training_score': avg_training_score,
            'KPIs_met >80%': kpis_met
        }
        df = pd.DataFrame([data])

        df['age_log'] = np.log1p(df['age'])
        df['length_of_service_log'] = np.log1p(df['length_of_service'])

        if department in ['Legal', 'Other']:
            df['department'] = 'Other'

        cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        num_cols = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score', 'age_log', 'length_of_service_log']
        df[num_cols] = scaler.transform(df[num_cols])

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[required_columns]

        data_array = df.values.astype('float32')
        dmatrix = xgb.DMatrix(data_array)

        raw_margin = model.get_booster().predict(dmatrix, output_margin=True, validate_features=False)[0]
        prob = 1 / (1 + np.exp(-raw_margin))

    st.markdown(f"<h2 class='prediction'>Promotion Probability: <span style='color:#fa7343'>{prob:.1%}</span></h2>", unsafe_allow_html=True)

    if prob > 0.5:
        st.markdown("<div class='success'>🎉 The employee is likely to be promoted! 🎈</div>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("<div class='warning'>😔 The employee is unlikely to be promoted this year.</div>", unsafe_allow_html=True)

st.caption("Employee Promotion Prediction Project • Developed by Salma • Powered by XGBoost")
