# Employee Promotion Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Made with Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

A simple and intelligent web application that predicts the likelihood of an employee's promotion based on their personal and professional data using the **XGBoost** model.
# Try the app directly
👉 [Click here to try the app](https://your-app-name.streamlit.app)

(Replace `your-app-name` with the actual URL after deployment, e.g., employee-promotion-predictor.streamlit.app)

# 📊 About the project
- **Dataset**: HR Analytics Dataset from Kaggle (~55,000 original records).

- **Expansion**: Industrial data was generated to reach **300,000 records** to improve model performance.

- **Technologies used**:

- XGBoost Classifier

- Feature Engineering (log transforms, rare category grouping)

- One-Hot Encoding + Standard Scaling

- Hyperparameter tuning with Hyperopt
- **Model performance**: ROC-AUC ≈ 0.84 on test data

### How to use the app?

Enter employee data:
- Department, Region, Qualification, Age, Years of Service
- Previous year's evaluation, Awards, Average training scores, etc.

The application will instantly calculate:
- Probability of promotion as a percentage
- Final decision: Promotion or not

### 🛠️ Files in the repo
- `app.py` → Main application code (Streamlit)
- `employee_promotion_model.pkl` → Trained model
- `requirements.txt` → Required libraries

### 💻 Run locally
```bash
`git clone https://github.com/username/your-repo-name.git
`cd your-repo-name
`pip install -r requirements.txt
`streamlit run app.py`
