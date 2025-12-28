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

# --- 2. دالة تحميل الموديل والملفات (مظبوطة المسافات) ---
@st.cache_resource
def load_model_artifacts():
    model_path = 'employee_promotion_model.json'
    
    # التأكد من وجود الملف وحجمه
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        st.error(f"❌ الملف {model_path} غير موجود أو تالف على GitHub!")
        st.stop()

    try:
        # تحميل الملفات المساعدة
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # محاولة التحميل كـ Classifier أولاً
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, scaler, feature_columns
    except Exception:
        try:
            # محاولة بديلة كـ Booster لو الطريقة الأولى فشلت
            model = xgb.Booster()
            model.load_model(model_path)
            return model, scaler, feature_columns
        except Exception as e:
            st.error(f"❌ فشل تحميل الموديل تماماً: {e}")
            st.stop()

# استدعاء الدالة لتحميل الموارد
model, scaler, feature_columns = load_model_artifacts()

# --- 3. واجهة المدخلات (Sidebar) ---
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

# --- 4. معالجة البيانات (التحويل والتحجيم) ---
try:
    # 1. تجهيز الـ 6 أعمدة الرقمية للـ Scaler
    cols_for_scaler = ['age', 'no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
    df_num = pd.DataFrame([[float(age), float(no_of_trainings), float(previous_year_rating), 
                            float(length_of_service), float(awards_won), float(avg_training_score)]], 
                          columns=cols_for_scaler)
    
    # عمل الـ Scaling (استخدام .values لتجنب أخطاء الأسماء)
    scaled_data = scaler.transform(df_num.values)
    scaled_dict = dict(zip(cols_for_scaler, scaled_data[0]))
    
    # 2. إضافة ميزات الـ Log والـ Age Group
    scaled_dict['age_log'] = np.log1p(float(age))
    scaled_dict['length_of_service_log'] = np.log1p(float(length_of_service))
    age_group = pd.cut([age], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'], right=False)[0]
    
    # 3. بناء الداتا قبل الـ Encoding
    input_combined = {
        'department': department, 'region': region, 'education': education,
        'gender': gender, 'recruitment_channel': recruitment_channel,
        'age_group': age_group,
        **scaled_dict
    }
    
    # 4. الـ One-Hot Encoding
    df_temp = pd.DataFrame([input_combined])
    df_encoded = pd.get_dummies(df_temp)
    
    # 5. إضافة الميزات التفاعلية (Interactive Features)
    df_encoded['high_training_score'] = 1 if avg_training_score > 80 else 0
    df_encoded['has_awards'] = int(awards_won)
    df_encoded['long_service_high_rating'] = 1 if (length_of_service > 7 and previous_year_rating >= 4) else 0

    # 6. مطابقة الأعمدة النهائية مع الموديل (Alignment)
    final_input = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        final_input[col] = df_encoded[col] if col in df_encoded.columns else 0.0

except Exception as e:
    st.error(f"⚠️ خطأ أثناء معالجة البيانات: {e}")
    st.stop()

# --- 5. التوقع وعرض النتيجة ---
if st.button("Predict Promotion Status"):
    try:
        # التوقع بناءً على نوع الموديل المحمل
        if isinstance(model, xgb.XGBClassifier):
            prob = model.predict_proba(final_input)[0][1]
        else:
            dmat = xgb.DMatrix(final_input)
            prob = model.predict(dmat)[0]
        
        prediction = 1 if prob > 0.5 else 0

        st.divider()
        if prediction == 1:
            st.success(f"### 🎉 مبروك! الموظف مرشح للترقية")
            st.metric("احتمالية الترقية", f"{prob*100:.2f}%")
        else:
            st.error(f"### 😔 الترقية غير محتملة حالياً")
            st.metric("احتمالية الترقية", f"{prob*100:.2f}%")
            
    except Exception as e:
        st.error(f"❌ خطأ أثناء التوقع: {e}")
