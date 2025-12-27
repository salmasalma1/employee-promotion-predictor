import streamlit as st
import pandas as pd
import numpy as np
import joblib # for loading scaler and feature_columns
import xgboost as xgb

# --- Streamlit UI --- 
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

st.title("🚀 Employee Promotion Predictor")
st.write("Enter employee details to predict their promotion status.")

# Load the trained model and artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        model = xgb.XGBClassifier()
        model.load_model('employee_promotion_model.json')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure 'employee_promotion_model.json', 'scaler.pkl', and 'feature_columns.pkl' are in the same directory.")
        st.stop()

model, scaler, feature_columns = load_model_artifacts()


# Input fields
with st.sidebar:
    st.header("Employee Details")
    # Categories based on zIrHZle4eanY output after rare category handling
    department = st.selectbox("Department", ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'Procurement', 'Other'])
    region = st.selectbox("Region", ['region_2', 'region_7', 'region_22', 'Other'])
    education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Other"])
    gender = st.selectbox("Gender", ['m', 'f', 'Other']) # 'other' was mapped to 'Other'
    recruitment_channel = st.selectbox("Recruitment Channel", ['other', 'sourcing', 'Other']) # 'referred' was mapped to 'Other'
    
    no_of_trainings = st.slider("Number of Trainings", 1, 10, 1)
    age = st.slider("Age", 20, 60, 30)
    previous_year_rating = st.selectbox("Previous Year Rating", [1.0, 2.0, 3.0, 4.0, 5.0])
    length_of_service = st.slider("Length of Service (Years)", 1, 37, 5)
    awards_won = st.selectbox("Awards Won (0=No, 1=Yes)", [0, 1])
    avg_training_score = st.slider("Average Training Score", 40, 99, 60)

# Create a DataFrame from inputs
input_data = {
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
    'avg_training_score': avg_training_score
}
df_input = pd.DataFrame([input_data])

# --- Feature Engineering (MUST match training pipeline exactly) ---
# Additional engineered features
df_input['age_group'] = pd.cut(df_input['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)
df_input['high_training_score'] = (df_input['avg_training_score'] > 80).astype(int)
df_input['has_awards'] = df_input['awards_won'] # Assuming awards_won is already 0 or 1
df_input['long_service_high_rating'] = ((df_input['length_of_service'] > 7) & (df_input['previous_year_rating'] >= 4)).astype(int)

# One-Hot Encoding
categorical_features_for_ohe = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']
df_encoded = pd.get_dummies(df_input, columns=categorical_features_for_ohe, drop_first=True)

# Numerical features for scaling (from G6YxFrpd0axj)
numerical_features_to_scale = [
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'avg_training_score',
    'high_training_score', 'has_awards', 'long_service_high_rating'
]

# Ensure all numerical features exist before scaling
for col in numerical_features_to_scale:
    if col not in df_encoded.columns:
        df_encoded[col] = 0.0 # Should not happen with direct input

df_encoded[numerical_features_to_scale] = scaler.transform(df_encoded[numerical_features_to_scale])

# Align columns with the model's expected feature order
final_df = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    if col in df_encoded.columns:
        final_df[col] = df_encoded[col]
    else:
        final_df[col] = 0 # Add missing one-hot encoded columns with 0

# Make prediction
if st.button("Predict Promotion"):
    prediction = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Yes, the employee is likely to be promoted!** 🚀")
        st.write(f"Probability of Promotion: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.error(f"**No, the employee is likely NOT to be promoted.** 😔")
        st.write(f"Probability of Not Being Promoted: **{prediction_proba[0]*100:.2f}%**")
    
    st.info("This prediction is based on the trained XGBoost model and historical data.")
