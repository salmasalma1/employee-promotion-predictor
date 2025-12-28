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

# --- 2. واجهة مدخلات المستخدم (Sidebar) ---
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

# --- 3. الـ Scaling (لـ 6 أعمدة رقمية فقط) ---
# الترتيب ده هو اللي السكيلر متوقعه للأرقام
cols_for_scaler = [
    'age', 'no_of_trainings', 'previous_year_rating', 
    'length_of_service', 'awards_won', 'avg_training_score'
]

# تجهيز البيانات الرقمية
df_raw_num = pd.DataFrame([[age, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score]], 
                          columns=cols_for_scaler)

try:
    # عمل الـ Scaling للـ 6 أعمدة فقط باستخدام .values لتجنب أخطاء الأسماء
    scaled_data = scaler.transform(df_raw_num.values)
    scaled_df = pd.DataFrame(scaled_data, columns=cols_for_scaler)
    
    # تحضير القيم المحجمة للاستخدام لاحقاً
    scaled_values = scaled_df.iloc[0].to_dict()
    
    # حساب ميزات الـ Log (غالباً الموديل يحتاجها خارج السكيلر)
    scaled_values['age_log'] = np.log1p(age)
    scaled_values['length_of_service_log'] = np.log1p(length_of_service)
    
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# --- 4. تجهيز البيانات للموديل (Encoding & Alignment) ---
# تجميع كل البيانات في صف واحد
input_for_encoding = {
    'department': department, 'region': region, 'education': education,
    'gender': gender, 'recruitment_channel': recruitment_channel,
    'age_group': pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0],
    **scaled_values
}

df_ready = pd.DataFrame([input_for_encoding])
df_encoded = pd.get_dummies(df_ready)

# إضافة ميزات إضافية للموديل (إن وجدت في feature_columns)
df_encoded['high_training_score'] = (avg_training_score > 80).astype(int)
df_encoded['has_awards'] = awards_won
df_encoded['long_service_high_rating'] = ((length_of_service > 7) & (previous_year_rating >= 4)).astype(int)

# التأكد من مطابقة جميع الأعمدة اللي الموديل اتدرب عليها
final_df = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    final_df[col] = df_encoded[col] if col in df_encoded.columns else 0

# --- 5. زر التوقع ---
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
