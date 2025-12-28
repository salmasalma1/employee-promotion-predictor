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

# تجهيز البيانات الخام
input_data = {
    'age': age, 'gender': gender, 'department': department, 'region': region, 
    'education': education, 'recruitment_channel': recruitment_channel, 
    'no_of_trainings': no_of_trainings, 'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service, 'awards_won': awards_won, 
    'avg_training_score': avg_training_score
}
df_input = pd.DataFrame([input_data])

# --- Feature Engineering ---
df_input['age_log'] = np.log1p(df_input['age'])
df_input['length_of_service_log'] = np.log1p(df_input['length_of_service'])
df_input['is_promoted'] = 0 # عمود وهمي لأن السكيلر قد يطلبه

# --- Scaling (حل مشكلة عدد الأعمدة 14) ---
# القائمة الكاملة اللي السكيلر اتدرب عليها في كولاب بالترتيب
scaler_features_ordered = [
    'age', 'gender', 'department', 'region', 'education',
    'recruitment_channel', 'no_of_trainings', 'previous_year_rating',
    'length_of_service', 'awards_won', 'avg_training_score', 'is_promoted',
    'age_log', 'length_of_service_log'
]

try:
    # إنشاء DataFrame مؤقت فيه الـ 14 عمود بالظبط
    temp_df = pd.DataFrame(columns=scaler_features_ordered)
    for col in scaler_features_ordered:
        temp_df.loc[0, col] = df_input.loc[0, col] if col in df_input.columns else 0
    
    # تحويل القيم (Scaling)
    scaled_values = scaler.transform(temp_df.values)
    temp_df_scaled = pd.DataFrame(scaled_data, columns=scaler_features_ordered)
    
    # تحضير البيانات لـ One-Hot Encoding بعد الـ Scaling
    # (نستخدم البيانات المحجمة للقيم الرقمية)
    df_for_encode = df_input.copy()
    num_cols = ['age', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'avg_training_score', 'age_log', 'length_of_service_log']
    for col in num_cols:
        df_for_encode[col] = temp_df_scaled[col].values

    # One-Hot Encoding
    df_for_encode['age_group'] = pd.cut(df_input['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)
    categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'age_group']
    df_encoded = pd.get_dummies(df_for_encode, columns=categorical_features, drop_first=True)

    # إضافة ميزات إضافية قد يحتاجها الموديل
    df_encoded['high_training_score'] = (df_input['avg_training_score'] > 80).astype(int)
    df_encoded['has_awards'] = df_input['awards_won']
    df_encoded['long_service_high_rating'] = ((df_input['length_of_service'] > 7) & (df_input['previous_year_rating'] >= 4)).astype(int)

    # محاذاة الأعمدة مع الموديل (feature_columns)
    final_df = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        final_df[col] = df_encoded[col] if col in df_encoded.columns else 0

except Exception as e:
    st.error(f"Scaling/Encoding Error: {e}")
    st.stop()

# --- Prediction ---
if st.button("Predict Promotion"):
    dmatrix_input = xgb.DMatrix(final_df)
    prob = model.predict(dmatrix_input)[0]
    prediction = 1 if prob > 0.5 else 0

    st.subheader("Result:")
    if prediction == 1:
        st.success(f"**Promoted!** 🚀 (Prob: {prob*100:.2f}%)")
    else:
        st.error(f"**Not Promoted.** 😔 (Prob: {prob*100:.2f}%)")
