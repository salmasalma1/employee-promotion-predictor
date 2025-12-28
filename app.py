import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb
import os

# --- 1. إعدادات الصفحة --- 
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

st.title("🚀 Employee Promotion Predictor")
st.write("Enter employee details to predict their promotion status.")

# --- 2. تحميل الموديل والملفات ---
@st.cache_resource
def load_model_artifacts():
    model_path = 'employee_promotion_model.json'
    
    # التأكد من وجود الملف وحجمه
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        st.error(f"❌ الملف {model_path} غير موجود أو حجمه صفر على السيرفر!")
        st.stop()

    try:
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # محاولة تحميل الموديل بالطريقة القياسية
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, scaler, feature_columns
    except Exception as e:
        try:
            # محاولة بديلة لو الطريقة الأولى فشلت
            model = xgb.Booster()
            model.load_model(model_path)
            return model, scaler, feature_columns
        except Exception as e2:
            st.error(f"❌ فشل تحميل الموديل: {e2}")
            st.stop()

model, scaler, feature_columns = load_model_artifacts()

# --- 3. واجهة مدخلات المستخدم ---
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

# --- 4. معالجة البيانات ---
# السكيلر متوقع 6 أعمدة رقمية
cols_for_scaler = ['age', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
df_num = pd.DataFrame([[age, no_of_trainings, previous_year_rating, length_of_service, awards_won, avg_training_score]], columns=cols_for_scaler)

try:
    scaled_data = scaler.transform(df_num.values)
    scaled_values = dict(zip(cols_for_scaler, scaled_data[0]))
    # إضافة ميزات الـ Log
    scaled_values['age_log'] = np.log1p(age)
    scaled_values['length_of_service_log'] = np.log1p(length_of_service)
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# بناء الـ DataFrame النهائي للموديل
input_combined = {
    'department': department, 'region': region, 'education': education,
    'gender': gender, 'recruitment_channel': recruitment_channel,
    'age_group': pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0],
    **scaled_values
}

df_final = pd.get_dummies(pd.DataFrame([input_combined]))
df_final['high_training_score'] = (avg_training_score > 80).astype(int)
df_final['has_awards'] = awards_won
df_final['long_service_high_rating'] = ((length_of_service > 7) & (previous_year_rating >= 4)).astype(int)

# ضبط الأعمدة
final_input = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    final_input[col] = df_final[col] if col in df_final.columns else 0

# --- 5. التوقع ---
if st.button("Predict Promotion"):
    # تحديد طريقة التوقع بناءً على نوع الموديل المحمل
    if isinstance(model, xgb.XGBClassifier):
        prob = model.predict_proba(final_input)[0][1]
    else:
        dmat = xgb.DMatrix(final_input)
        prob = model.predict(dmat)[0]
    
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Result:")
    if prediction == 1:
        st.success(f"**Yes! Likely to be promoted.** 🚀 (Prob: {prob*100:.2f}%)")
    else:
        st.error(f"**No. Not likely to be promoted.** 😔 (Prob: {prob*100:.2f}%)")
