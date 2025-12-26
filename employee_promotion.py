import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Loading the model
@st.cache_resource
def load_model():

return joblib.load('employee_promotion_model.pkl')

model = load_model()

st.set_page_config(page_title="Employee Promotion Prediction", page_icon="👔", layout="centered")

st.title("👔 Employee Promotion Prediction")
st.markdown("### XGBoost model trained on 300,000 HR data records")

st.write("Enter employee data to predict promotion probability")

col1, col2 = st.columns(2)

with col1:

department = st.selectbox("Department",
['Sales & Marketing', 'Operations', 'Procurement', 'Technology',
'Finance', 'Analytics', 'R&D', 'HR', 'Legal', 'Other'])

education = st.selectbox("Qualification",
["Below Secondary", "Bachelor's", "Master's & above", 'Other'])

gender = st.selectbox("Gender", ['f', 'm'])

recruitment_channel = st.selectbox("Recruitment Channel", ['sourcing', 'other', 'referred'])

no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=10, value=1)

with col2:

region = st.selectbox("Region",
['region_2', 'region_22', 'region_7', 'region_15', 'region_13',
'region_4', 'region_26', 'region_16', 'region_27', 'region_10', 'Other'])
age = st.slider("Age", min_value=18, max_value=60, value=35)
length_of_service = st.slider("Years of Service", min_value=1, max_value=37, value=5)
previous_year_rating = st.selectbox("Previous Year Rating", [1,2,3,4,5], index=2)
awards_won = st.selectbox("Won Awards?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
avg_training_score = st.slider("Average Training Score", min_value=39, max_value=99, value=75)

if st.button("🔮 Upgrade prediction", type="primary"): 
with st.spinner("Accounting..."): 
data = { 
'department': department, 'region': region, 'education': education, 
'gender': gender, 'recruitment_channel': recruitment_channel, 
'no_of_trainings': no_of_trainings, 'age': age, 
'previous_year_rating': previous_year_rating, 
'length_of_service':length_of_service, 
'awards_won': awards_won, 'avg_training_score': avg_training_score 
} 
df = pd.DataFrame([data]) 

df['age_log'] = np.log1p(df['age']) 
df['length_of_service_log'] = np.log1p(df['length_of_service']) 

if department in ['Legal', 'Other']: 
df['department'] = 'Other' 

cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel'] 
df = pd.get_dummies(df, columns=cat_cols, drop_first=True) 

num_cols = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score', 'age_log', 'length_of_service_log'] 
scaler = StandardScaler() 
df[num_cols] = scaler.fit_transform(df[num_cols]) 

prob= model.predict_proba(df)[0][1]

pred = model.predict(df)[0]

st.markdown(f"### Probability of promotion: **{prob: .1%}**")

if pred == 1:

st.success("🎉 Expected to be promoted!")

st.balloons()

else:

st.warning("😔 Expected not to be promoted.")

st.info("Model trained on extensive data (300k) using XGBoost")

st.caption("Employee promotion prediction project • Developed by [Your Name]")
