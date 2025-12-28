import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb

# --- إعدادات الصفحة --- 
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

st.title("🚀 Employee Promotion Predictor")
st.write("Enter employee details to predict their promotion status.")

# --- تحميل الموديل والملفات المساعدة ---
@st.cache_resource
def load_model_artifacts():
    try:
        # 1. تحميل الموديل باستخدام Booster لتجنب TypeError
        model = xgb.Booster()
        model.load_model('employee_promotion_model.json')
        
        # 2. تحميل الـ Scaler وأسماء الأعمدة
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

model, scaler, feature_columns = load_model_artifacts()

# --- واجهة المدخلات (Sidebar) ---
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

# تجهيز البيانات
input_data = {
    'department': department, 'region': region, 'education': education,
    'gender': gender, 'recruitment_channel': recruitment_channel,
    'no_of_trainings': no_of_trainings, 'age': age,
    'previous_year_rating': previous_year_rating, 'length_of_service': length_of_service,
    'awards_won': awards_won, 'avg_training_score': avg_training_score
}
df_input = pd.DataFrame([input_data])

# --- Feature Engineering ---
# حساب الـ Log features كما في الكولاب الخاص بك
df_input['age_log'] = np.log1p(df_input['age'])
df_input['length_of_service_log'] = np.log1p(df_input['length_of_service'])

# هندسة ميزات إضافية (اختياري حسب موديلك)
df_input['age_group'] = pd.cut(df_input['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)
df_input['high_training_score'] = (df_input['avg_training_score'] > 80).astype(int)
df_input['has_awards'] = df_input['awards_won']
df_input['long_service_high_rating'] = ((df_input['length_of_service'] > 7) & (df_input['previous_year_rating'] >= 4)).astype(int)

# One-Hot Encoding
categorical_features_for_ohe = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']
df_encoded = pd.get_dummies(df_input, columns=categorical_features_for_ohe, drop_first=True)

# --- Scaling (حل مشكلة ValueError) ---
# الترتيب بناءً على الـ Index اللي بعتهولي
numerical_features_to_scale = [
    'age', 'no_of_trainings', 'previous_year_rating', 
    'length_of_service', 'awards_won', 'avg_training_score',
    'age_log', 'length_of_service_log'
]

# التأكد من وجود كل الأعمدة وترتيبها
for col in numerical_features_to_scale:
    if col not in df_encoded.columns:
        df_encoded[col] = 0.0

# استخدام .values لتخطي فحص الأسماء في السكيلر
scaled_values = scaler.transform(df_encoded[numerical_features_to_scale].values)
df_encoded[numerical_features_to_scale] = scaled_values

# محاذاة الأعمدة مع الموديل
final_df = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    final_df[col] = df_encoded[col] if col in df_encoded.columns else 0

# --- التوقع ---
if st.button("Predict Promotion"):
    # استخدام DMatrix للـ Booster
    dmatrix_input = xgb.DMatrix(final_df)
    prob = model.predict(dmatrix_input)[0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Yes, the employee is likely to be promoted!** 🚀")
        st.write(f"Probability of Promotion: **{prob*100:.2f}%**")
    else:
        st.error(f"**No, the employee is likely NOT to be promoted.** 😔")
        st.write(f"Probability: **{(1-prob)*100:.2f}%**")
