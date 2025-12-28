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

# --- 2. واجهة مدخلات المستخدم ---
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
    'age': age, 'no_of_trainings': no_of_trainings, 'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service, 'awards_won': awards_won, 'avg_training_score': avg_training_score
}
df_raw = pd.DataFrame([input_data])
df_raw['age_log'] = np.log1p(df_raw['age'])
df_raw['length_of_service_log'] = np.log1p(df_raw['length_of_service'])

# --- 3. الـ Scaling الذكي (حل مشكلة string to float) ---
# السكيلر بتاعك متوقع 14 عمود. إحنا هنديله مصفوفة (Array) فيها 14 عمود أصفار
# ونحط الأرقام بتاعتنا في أماكنها الصح بالظبط
try:
    # إنشاء مصفوفة أصفار (1 صف و 14 عمود)
    X_dummy = np.zeros((1, 14))
    
    # توزيع الأرقام بناءً على الترتيب اللي أنت بعته للأعمدة الـ 14:
    # 'age' هو العمود 0
    # 'no_of_trainings' هو العمود 6
    # 'previous_year_rating' هو العمود 7
    # 'length_of_service' هو العمود 8
    # 'awards_won' هو العمود 9
    # 'avg_training_score' هو العمود 10
    # 'age_log' هو العمود 12
    # 'length_of_service_log' هو العمود 13
    
    X_dummy[0, 0] = df_raw['age'].values[0]
    X_dummy[0, 6] = df_raw['no_of_trainings'].values[0]
    X_dummy[0, 7] = df_raw['previous_year_rating'].values[0]
    X_dummy[0, 8] = df_raw['length_of_service'].values[0]
    X_dummy[0, 9] = df_raw['awards_won'].values[0]
    X_dummy[0, 10] = df_raw['avg_training_score'].values[0]
    X_dummy[0, 12] = df_raw['age_log'].values[0]
    X_dummy[0, 13] = df_raw['length_of_service_log'].values[0]
    
    # عمل الـ Scaling للمصفوفة كلها
    scaled_array = scaler.transform(X_dummy)
    
    # سحب القيم المحجمة اللي تهمنا
    scaled_values = {
        'age': scaled_array[0, 0],
        'no_of_trainings': scaled_array[0, 6],
        'previous_year_rating': scaled_array[0, 7],
        'length_of_service': scaled_array[0, 8],
        'awards_won': scaled_array[0, 9],
        'avg_training_score': scaled_array[0, 10],
        'age_log': scaled_array[0, 12],
        'length_of_service_log': scaled_array[0, 13]
    }
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# --- 4. تجهيز البيانات للموديل (One-Hot Encoding) ---
df_for_model = pd.DataFrame([{
    'department': department, 'region': region, 'education': education,
    'gender': gender, 'recruitment_channel': recruitment_channel,
    'age_group': pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0],
    **scaled_values # دمج القيم المحجمة
}])

df_encoded = pd.get_dummies(df_for_model)

# محاذاة الأعمدة مع قائمة feature_columns الخاصة بالموديل
final_df = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    final_df[col] = df_encoded[col] if col in df_encoded.columns else 0

# --- 5. التوقع ---
if st.button("Predict Promotion"):
    dmatrix_input = xgb.DMatrix(final_df)
    prob = model.predict(dmatrix_input)[0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Result:")
    if prediction == 1:
        st.success(f"**Promoted!** 🚀 (Prob: {prob*100:.2f}%)")
    else:
        st.error(f"**Not Promoted.** 😔 (Prob: {prob*100:.2f}%)")
