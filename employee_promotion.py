import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model('employee_promotion_model.json')
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

st.set_page_config(page_title="Employee Promotion Prediction", page_icon="👔", layout="centered")
st.title("👔 Employee Promotion Prediction")
st.markdown("### XGBoost model trained on 300,000 HR records")
st.write("Enter employee details to predict the probability of promotion")

col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department",
                              ['Sales & Marketing', 'Operations', 'Procurement', 'Technology',
                               'Finance', 'Analytics', 'R&D', 'HR', 'Legal', 'Other'])
    education = st.selectbox("Education Level",
                             ["Below Secondary", "Bachelor's", "Master's & above", 'Other'])
    gender = st.selectbox("Gender", ['f', 'm'])
    recruitment_channel = st.selectbox("Recruitment Channel", ['sourcing', 'other', 'referred'])
    no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=10, value=1)

with col2:
    region = st.selectbox("Region",
                          ['region_2', 'region_22', 'region_7', 'region_15', 'region_13',
                           'region_4', 'region_26', 'region_16', 'region_27', 'region_10', 'Other'])
    age = st.slider("Age", 18, 60, 35)
    length_of_service = st.slider("Years of Service", 1, 37, 5)
    previous_year_rating = st.selectbox("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    awards_won = st.selectbox("Awards Won?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    avg_training_score = st.slider("Average Training Score", 39, 99, 75)

if st.button("🔮 Predict Promotion", type="primary"):
    with st.spinner("Predicting..."):
        data = {
            'department': department,
            'region': region,
            'education': education,
            'gender': gender,
            'recruitment_channel': recruitment_channel,
            'no_of_trainings': no_of_trainings,
            'age': age,
            'previous_year_rating': float(previous_year_rating),
            'length_of_service': length_of_service,
            'awards_won': awards_won,
            'avg_training_score': avg_training_score
        }
        df = pd.DataFrame([data])

        df['age_log'] = np.log1p(df['age'])
        df['length_of_service_log'] = np.log1p(df['length_of_service'])

        if department in ['Legal', 'Other']:
            df['department'] = 'Other'

        cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        num_cols = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score',
                    'age_log', 'length_of_service_log']
        df[num_cols] = scaler.transform(df[num_cols])

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[required_columns]

        data_array = df.values.astype('float32')
        dmatrix = xgb.DMatrix(data_array, feature_names=None)

        raw_margin = model.get_booster().predict(dmatrix, output_margin=True)[0]
        prob = 1 / (1 + np.exp(-raw_margin))
        pred = 1 if prob > 0.5 else 0

    st.markdown(f"### Promotion Probability: **{prob:.1%}**")
    if pred == 1:
        st.success("🎉 The employee is likely to be promoted!")
        st.balloons()
    else:
        st.warning("😔 The employee is unlikely to be promoted this year.")

    st.info("Model trained on augmented data (300k records) using XGBoost")

st.caption("Employee Promotion Prediction Project • Developed by Salma")
