import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb
import plotly.graph_objects as go

# --- 1. إعدادات الصفحة والتصميم (The Ultra GUI) --- 
st.set_page_config(page_title="AI HR Analytics Pro", layout="wide", initial_sidebar_state="collapsed")

# CSS لتحويل Streamlit لـ Dashboard احترافي
st.markdown("""
    <style>
    .stApp {
        background: #0e1117;
        color: #e6edf3;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    
    /* تصميم الكروت */
    .css-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* العناوين */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: -webkit-linear-gradient(#fff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* الزرار الفاشخ */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff7575);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 43, 43, 0.2);
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
st.markdown("<h1 class='main-title'>AI HR Analytics Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; margin-top:-10px;'>Empowering Data-Driven Talent Decisions</p>", unsafe_allow_html=True)

# --- 3. تصميم الواجهة (Layout) ---
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("### 👤 Employee Demographics")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            department = st.selectbox("Department", ['Analytics', 'Sales & Marketing', 'Operations', 'Technology', 'Procurement', 'HR', 'Finance', 'Legal'])
            education = st.selectbox("Education Level", ["Bachelor's", "Master's & above", "Below Secondary"])
            gender = st.selectbox("Gender", ['m', 'f'])
        with c2:
            region = st.selectbox("Region", [f'region_{i}' for i in range(1, 35)])
            recruitment_channel = st.selectbox("Recruitment Channel", ['referred', 'sourcing', 'other'])
            kpis_met = st.selectbox("KPIs Met >80%?", ['Yes', 'No'])

    st.markdown("### 📊 Performance & Experience")
    with st.container():
        c3, c4 = st.columns(2)
        with c3:
            age = st.slider("Age", 20, 60, 30)
            length_of_service = st.slider("Years of Service", 1, 37, 5)
        with c4:
            no_of_trainings = st.number_input("No. of Trainings", 1, 10, 1)
            previous_year_rating = st.select_slider("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0], value=3.0)
            avg_training_score = st.slider("Average Training Score", 40, 99, 60)
            awards_won = st.radio("Awards Won?", ['Yes', 'No'], horizontal=True)

with col_right:
    st.markdown("### 📈 Talent Insights")
    # Radar Chart (الرسم البياني الفاشخ)
    categories = ['Score', 'Rating', 'Trainings', 'Service', 'KPIs']
    # Normalize values for radar
    radar_values = [
        (avg_training_score-40)/60, 
        previous_year_rating/5, 
        (10-no_of_trainings)/10, 
        length_of_service/37,
        1 if kpis_met == 'Yes' else 0.2
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=radar_values,
        theta=categories,
        fill='toself',
        line_color='#ff4b4b'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    predict_btn = st.button("RUN PREDICTION ANALYTICS")

# --- 4. Logic & Result ---
if predict_btn:
    # Feature Engineering
    has_awards = 1 if awards_won == 'Yes' else 0
    high_training_score = 1 if avg_training_score > 80 else 0
    long_service_high_rating = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0
    age_group = pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0]

    features = [
        'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
        'avg_training_score', 'high_training_score', 'has_awards', 
        'long_service_high_rating', 'department', 'region', 'education', 
        'gender', 'recruitment_channel', 'age_group'
    ]
    
    input_df = pd.DataFrame([[
        no_of_trainings, age, previous_year_rating, length_of_service, 
        avg_training_score, high_training_score, has_awards, 
        long_service_high_rating, department, region, education, 
        gender, recruitment_channel, age_group
    ]], columns=features)

    for col in ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']:
        input_df[col] = input_df[col].astype('category')

    dmat = xgb.DMatrix(input_df, enable_categorical=True)
    prob = model.predict(dmat)[0]

    # عرض النتيجة بكارت فخم
    if prob > 0.5:
        st.balloons()
        st.markdown(f"""
            <div style="background: rgba(46, 204, 113, 0.2); border-radius: 15px; padding: 20px; border: 1px solid #2ecc71; text-align: center;">
                <h2 style="color: #2ecc71; margin:0;">PROMOTED 🚀</h2>
                <h1 style="margin:0;">{prob*100:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background: rgba(231, 76, 60, 0.2); border-radius: 15px; padding: 20px; border: 1px solid #e74c3c; text-align: center;">
                <h2 style="color: #e74c3c; margin:0;">REJECTED 😔</h2>
                <h1 style="margin:0;">{prob*100:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)
