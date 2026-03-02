import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="CreditWise Dashboard", page_icon="💳", layout="wide")

# 2. Custom CSS for Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #2E59FF;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1A3BB5;
        color: white;
    }
    h1 {
        color: #1E3A8A;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Title
st.markdown("<h1>📊 CreditWise: Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.divider()

# 3. Load and Train Model
@st.cache_data
def load_and_train():
    df = pd.read_csv('cleaned_loan_data.csv')
    # Features selection as per your notebook
    X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
    y = df["Loan_Approved"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_scaled, y)
    
    return knn, scaler, X.columns

try:
    model, scaler, feature_names = load_and_train()

    # 4. Input Columns
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("👤 Personal Profile")
        age = st.slider("Applicant Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        dependents = st.number_input("Number of Dependents", 0, 10, 0)
        edu = st.radio("Education Level", ["Not Graduate", "Graduate"], horizontal=True)
        emp_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])

    with col2:
        st.subheader("💰 Financial Indicators")
        app_income = st.number_input("Monthly Income ($)", value=5000)
        co_income = st.number_input("Co-applicant Income ($)", value=0)
        loan_amount = st.number_input("Requested Loan Amount ($)", value=25000)
        savings = st.number_input("Current Savings ($)", value=10000)
        credit_score = st.slider("Credit Score", 300, 900, 700)
        dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)

    st.divider()

    # 5. Preparing Data for Prediction
    input_dict = {col: 0 for col in feature_names}
    
    input_dict['Applicant_Income'] = app_income
    input_dict['Coapplicant_Income'] = co_income
    input_dict['Age'] = age
    input_dict['Dependents'] = dependents
    input_dict['Savings'] = savings
    input_dict['Loan_Amount'] = loan_amount
    input_dict['Education_Level'] = 1 if edu == "Graduate" else 0
    input_dict['Gender_Male'] = 1 if gender == "Male" else 0
    input_dict['Marital_Status_Single'] = 1 if marital == "Single" else 0
    
    # Matching categorical flags
    if emp_status == "Salaried": input_dict['Employment_Status_Salaried'] = 1
    elif emp_status == "Self-employed": input_dict['Employment_Status_Self-employed'] = 1
    else: input_dict['Employment_Status_Unemployed'] = 1
    
    # Polynomial features
    input_dict['DTI_Ratio_sq'] = dti_ratio ** 2
    input_dict['Credit_Score_sq'] = credit_score ** 2

    # 6. Prediction Button
    if st.button("🔍 Analyze Eligibility"):
        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        if prediction[0] == 1:
            st.balloons()
            st.success("### ✅ Status: LOAN APPROVED")
        else:
            st.error("### ❌ Status: LOAN REJECTED")

except Exception as e:
    st.error(f"⚠️ Error: {e}")