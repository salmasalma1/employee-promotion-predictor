import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb

# --- إعدادات الصفحة --- 
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

st.title("🚀 Employee Promotion Predictor")
st.write("Enter employee details to predict their promotion status.")

# --- 1. تحميل الموديل والملفات ---
@st.cache_resource
def load_model_artifacts():
    try:
        model = xgb.Booster()
        model.load_model('employee_promotion_model.json')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

model, scaler, feature_columns = load_model_artifacts()

# --- 2. واجهة المدخلات (Sidebar) ---
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

# --- 3. تجهيز البيانات للـ Scaler (الـ 14 عمود) ---
input_raw = {
    'age': age, 'gender': gender, 'department': department, 'region': region, 
    'education': education, 'recruitment_channel': recruitment_channel, 
    'no_of_trainings': no_of_trainings, 'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service, 'awards_won': awards_won, 
    'avg_training_score': avg_training_score, 'is_promoted': 0 # عمود وهمي للسكيلر
}
df_raw = pd.DataFrame([input_raw])
df_raw['age_log'] = np.log1p(df_raw['age'])
df_raw['length_of_service_log'] = np.log1p(df_raw['length_of_service'])

# ترتيب الأعمدة الـ 14 بالظبط كما في كولاب
scaler_features_ordered = [
    'age', 'gender', 'department', 'region', 'education',
    'recruitment_channel', 'no_of_trainings', 'previous_year_rating',
    'length_of_service', 'awards_won', 'avg_training_score', 'is_promoted',
    'age_log', 'length_of_service_log'
]

# عمل الـ Scaling
try:
    temp_df = df_raw[scaler_features_ordered].copy()
    # السكيلر بيحتاج القيم فقط
    scaled_data = scaler.transform(temp_df.values)
    temp_df_scaled = pd.DataFrame(scaled_data, columns=scaler_features_ordered)
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# --- 4. تجهيز البيانات للموديل (One-Hot Encoding) ---
# نستخدم القيم المحجمة للبيانات الرقمية
df_for_model = df_raw.copy()
num_cols = ['age', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'avg_training_score', 'age_log', 'length_of_service_log']
for col in num_cols:
    df_for_model[col] = temp_df_scaled[col].values

# إضافة الـ Feature Engineering اللي الموديل مستنيها
df_for_model['age_group'] = pd.cut(df_for_model['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)
df_for_model['high_training_score'] = (df_raw['avg_training_score'] > 80).astype(int)
df_for_model['has_awards'] = df_raw['awards_won']
df_for_model['long_service_high_rating'] = ((df_raw['length_of_service'] > 7) & (df_raw['previous_year_rating'] >= 4)).astype(int)

# عمل الـ One-Hot Encoding
categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']
df_encoded = pd.get_dummies(df_for_model, columns=categorical_features, drop_first=True)

# محاذاة الأعمدة مع قائمة feature_columns الخاصة بالموديل
final_df = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    final_df[col] = df_encoded[col] if col in df_encoded.columns else 0

# --- 5. التوقع ---
if st.button("Predict Promotion"):
    # استخدام DMatrix للـ Booster
    dmatrix_input = xgb.DMatrix(final_df)
    prob = model.predict(dmatrix_input)[0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Result:")
    if prediction == 1:
        st.success(f"**Promoted!** 🚀 (Probability: {prob*100:.2f}%)")
    else:
        st.error(f"**Not Promoted.** 😔 (Probability: {prob*100:.2f}%)")
