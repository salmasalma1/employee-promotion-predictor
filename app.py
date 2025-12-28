import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb
import plotly.graph_objects as go

# --- 1. إعدادات الصفحة والتصميم الأسطوري --- 
st.set_page_config(page_title="AI HR Insights Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0e14;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1a1f2b, #0b0e14);
    }

    /* كروت البيانات */
    .data-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* تكبير العناوين */
    label {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* الزرار النووي */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 25px;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.4s all;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(168, 85, 247, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. تحميل الموديل ---
@st.cache_resource
def load_engine():
    model = xgb.Booster()
    model.load_model('employee_promotion_model.json')
    return model

model = load_engine()

# --- Header ---
st.markdown("<h1 style='text-align:center; font-size:4rem; font-weight:900; background: linear-gradient(to right, #fff, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>HR ANALYTICS ENGINE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b; font-size:1.2rem; margin-top:-20px;'>Enterprise-Grade Talent Prediction Framework</p>", unsafe_allow_html=True)

# --- 3. تصميم الـ Dashboard ---
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.markdown("<div class='data-card'>", unsafe_allow_html=True)
    st.subheader("Personal & Professional Profile")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        department = st.selectbox("Department", ['Analytics', 'Sales & Marketing', 'Operations', 'Technology', 'Procurement', 'HR', 'Finance', 'Legal', 'R&D'])
        education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary"])
    with r1c2:
        region = st.selectbox("Geographic Region", [f'region_{i}' for i in range(1, 35)])
        recruitment_channel = st.selectbox("Source Channel", ['referred', 'sourcing', 'other'])
    
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        age = st.slider("Age", 20, 60, 30)
    with r2c2:
        length_of_service = st.slider("Service (Years)", 1, 37, 5)
    with r2c3:
        avg_training_score = st.slider("Training Score", 40, 99, 60)
    
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        previous_year_rating = st.select_slider("Rating", [1.0, 2.0, 3.0, 4.0, 5.0], 3.0)
    with r3c2:
        no_of_trainings = st.number_input("Trainings", 1, 10, 1)
    with r3c3:
        awards_won = st.radio("Awards", ['Yes', 'No'], horizontal=True)
    
    gender = "m" # Hidden for cleaner UI or add if needed
    kpis_met = "Yes" # Defaulting for logic
    st.markdown("</div>", unsafe_allow_html=True)

with col_stats:
    st.markdown("<div class='data-card'>", unsafe_allow_html=True)
    st.subheader("Talent Radar Analysis")
    
    # Radar Chart
    categories = ['Competency', 'Experience', 'Rating', 'Education', 'Training']
    # Mock scaling logic
    r_values = [avg_training_score/100, length_of_service/37, previous_year_rating/5, 0.8 if "Master" in education else 0.5, (11-no_of_trainings)/10]
    
    fig = go.Figure(data=go.Scatterpolar(r=r_values, theta=categories, fill='toself', line_color='#a855f7', marker=dict(size=1)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), showlegend=False, 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- 4. التحليل والتوقع ---
if st.button("EXECUTE ANALYSIS"):
    # Same logic as before to maintain accuracy
    has_awards = 1 if awards_won == 'Yes' else 0
    high_training_score = 1 if avg_training_score > 80 else 0
    long_service_high_rating = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0
    age_group = pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0]

    input_df = pd.DataFrame([[no_of_trainings, age, previous_year_rating, length_of_service, avg_training_score, 
                              high_training_score, has_awards, long_service_high_rating, department, region, 
                              education, "m", "other", age_group]], 
                            columns=['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
                                     'avg_training_score', 'high_training_score', 'has_awards', 
                                     'long_service_high_rating', 'department', 'region', 'education', 
                                     'gender', 'recruitment_channel', 'age_group'])

    for col in ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']:
        input_df[col] = input_df[col].astype('category')

    dmat = xgb.DMatrix(input_df, enable_categorical=True)
    prob = model.predict(dmat)[0]

    st.markdown("---")
    if prob > 0.5:
        st.markdown(f"<div style='background: linear-gradient(90deg, #065f46, #059669); padding:40px; border-radius:20px; text-align:center;'><h1 style='color:white; margin:0;'>PROMOTION APPROVED</h1><h2 style='color:white; opacity:0.8;'>CONFIDENCE: {prob*100:.1f}%</h2></div>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"<div style='background: linear-gradient(90deg, #7f1d1d, #dc2626); padding:40px; border-radius:20px; text-align:center;'><h1 style='color:white; margin:0;'>PROMOTION DENIED</h1><h2 style='color:white; opacity:0.8;'>PROBABILITY: {prob*100:.1f}%</h2></div>", unsafe_allow_html=True)
