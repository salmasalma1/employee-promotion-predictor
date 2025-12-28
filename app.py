import streamlit as st
import pandas as pd
import xgboost as xgb

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Promotion Predictor",
    layout="centered"
)

# ================== TITLE ==================
st.markdown("""
<h1 style="text-align:center;">🚀 AI Promotion Predictor</h1>
<p style="text-align:center;color:#9aa4b2;">
Advanced HR Analytics • Explainable AI Decision Support
</p>
""", unsafe_allow_html=True)

# ================== SAMPLE INPUTS ==================
# استبدلهم بالـ Streamlit widgets بتوعك
no_of_trainings = 2
age = 30
previous_year_rating = 3
length_of_service = 10
avg_training_score = 84
department = "Operations"
region = "region_1"
education = "Bachelor's"
gender = "m"
recruitment_channel = "referred"
awards_won = "Yes"

# ================== FEATURE ENGINEERING (UNCHANGED) ==================
has_awards = 1 if awards_won == 'Yes' else 0
high_training_score = 1 if avg_training_score > 80 else 0
long_service_high_rating = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0

if age < 30:
    age_group = '<30'
elif age <= 40:
    age_group = '30-40'
elif age <= 50:
    age_group = '40-50'
else:
    age_group = '>50'

features_order = [
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'avg_training_score',
    'high_training_score', 'has_awards',
    'long_service_high_rating', 'department',
    'region', 'education', 'gender',
    'recruitment_channel', 'age_group'
]

input_df = pd.DataFrame([[
    no_of_trainings, age, previous_year_rating,
    length_of_service, avg_training_score,
    high_training_score, has_awards,
    long_service_high_rating, department,
    region, education, gender,
    recruitment_channel, age_group
]], columns=features_order)

for col in ['department', 'region', 'education', 'gender',
            'recruitment_channel', 'age_group']:
    input_df[col] = input_df[col].astype("category")

# ================== MODEL (Dummy for UI test) ==================
# استبدله بالموديل الحقيقي بتاعك
class DummyModel:
    def predict(self, dmat):
        return [0.72]

model = DummyModel()

# ================== PREDICTION ==================
if st.button("🚀 RUN ANALYSIS"):
    with st.spinner("Analyzing employee profile..."):
        dmat = xgb.DMatrix(input_df, enable_categorical=True)
        prob = model.predict(dmat)[0]

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg, rgba(46,204,113,0.18), rgba(46,204,113,0.05));
        border:2px solid #2ecc71;
        border-radius:18px;
        padding:32px;
        text-align:center;
    ">

        <div style="font-size:3.6rem;">🚀</div>

        <h1 style="color:#2ecc71;font-weight:900;letter-spacing:1px;">
            HIGH PROMOTION POTENTIAL
        </h1>

        <div style="
            font-size:2.8rem;
            font-weight:900;
            margin:14px 0;
            color:#ffffff;
        ">
            {prob*100:.1f}%
        </div>

        <span style="
            display:inline-block;
            padding:6px 16px;
            border-radius:999px;
            font-size:0.85rem;
            font-weight:800;
            background:#2ecc71;
            color:#0b1f14;
        ">
            High Confidence
        </span>

        <div style="
            height:10px;
            border-radius:8px;
            background:rgba(255,255,255,0.15);
            overflow:hidden;
            margin-top:18px;
        ">
            <div style="
                width:{prob*100:.1f}%;
                height:100%;
                background:#2ecc71;
                border-radius:8px;
            "></div>
        </div>

        <p style="margin-top:18px;color:#b9f6ca;font-size:1rem;">
            Employee strongly meets promotion criteria based on historical patterns.
        </p>

    </div>
    """, unsafe_allow_html=True)
