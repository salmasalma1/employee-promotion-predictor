import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

# --- 1. Page Settings ---
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

st.title("🚀 Employee Promotion Predictor")
st.write("Enter employee details to predict their promotion status.")

# --- 2. Loading the Model and Files ---
@st.cache_resource
def load_model_artifacts():

model_path = 'employee_promotion_model.json'

if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:

st.error(f"❌ The file {model_path} does not exist or is corrupt!")

st.stop()

try:

scaler = joblib.load('scaler.pkl')

feature_columns = joblib.load('feature_columns.pkl')

# Attempt to load as a Classifier

model = xgb.XGBClassifier()

model.load_model(model_path)

return model, scaler, feature_columns

except:

# Alternative attempt as a Booster

model = xgb.Booster()

model.load_model(model_path)

return model, scaler, feature_columns

except Exception as e:

st.error(f"❌ Model failed to load completely: {e}")

st.stop()

model, scaler, feature_columns = load_model_artifacts()

# --- 3. Input Interface ---
with st.sidebar:

st.header("Employee Details")

department = st.selectbox("Department", ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'Procurement', 'Other']) 
region = st.selectbox("Region", ['region_2', 'region_7', 'region_22', 'Other']) 
education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Other"]) 
gender = st.selectbox("Gender", ['m', 'f', 'Other']) 
recruitment_channel = st.selectbox("Recruitment Channel", ['other', 'sourcing', 'Other']) 

no_of_trainings = st.slider("Number of Trainings", 1, 10, 1) 
age = st.slider("Age", 20, 60, 30) 
previous_year_rating = st.selectbox("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0]) 
length_of_service = st.slider("Length of Service (Years)", 1, 37, 5) 
awards_won = st.selectbox("Awards Won (0=No, 1=Yes)", [0, 1]) 
avg_training_score = st.slider("Average Training Score", 40, 99, 60)

# --- 4. Data processing ---
# Convert inputs to a DataFrame for scaling
cols_for_scaler = ['age', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
df_num= pd.DataFrame([[float(age), float(no_of_trainings), float(previous_year_rating), float(length_of_service), float(awards_won), float(avg_training_score)]], columns=cols_for_scaler)

try:

# Scaling

scaled_data = scaler.transform(df_num.values)

scaled_dict = dict(zip(cols_for_scaler, scaled_data[0]))

# Adding Additional Features (Algorithm)

scaled_dict['age_log'] = np.log1p(float(age))

scaled_dict['length_of_service_log'] = np.log1p(float(length_of_service))

# Building the Complete Data Before Encoding

input_combined = {
'department': department, 'region': region, 'education': education,

'gender': gender, 'recruitment_channel': recruitment_channel,

'age_group': pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0],

**scaled_dict

}

# Convert to Encoding

df_temp = pd.DataFrame([input_combined])

df_encoded = pd.get_dummies(df_temp)

# Add interactive features (fix AttributeError)

# Use direct int() instead of .astype() to check

df_encoded['high_training_score'] = 1 if avg_training_score > 80 else 0 

df_encoded['has_awards'] = int(awards_won) 
df_encoded['long_service_high_rating'] = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0 

# Match final columns 
final_input = pd.DataFrame(columns=feature_columns) 
for col in feature_columns: 
if col in df_encoded.columns: 
final_input[col] = df_encoded[col] 
else: 
final_input[col] = 0.0

except Exception as e: 
st.error(f"⚠️ Data processing error: {e}") 
st.stop()

# --- 5. Expectation ---
if st.button("Predict Promotion"): 
try: 
if isinstance(model, xgb.XGBClassifier):

prob = model.predict_proba(final_input)[0][1]

else:

dmat = xgb.DMatrix(final_input)

prob = model.predict(dmat)[0]

prediction = 1 if prob > 0.5 else 0

st.divider()

if prediction == 1:

st.success(f"### 🎉 Congratulations! Employee nominated for promotion")

st.write(f"Prediction confidence: **{prob*100:.2f}%**")

else:

st.error(f"### 😔 Unfortunately, promotion is not currently likely")

st.write(f"Probability of promotion based on data: **{prob*100:.2f}%**")

except Exception as e:

st.error(f"❌ Prediction error: {e}")
