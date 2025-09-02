import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import tensorflow as tf
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL AND PREPROCESSOR LOADING ---
@st.cache_resource
def load_resources():
    """Loads all models and preprocessor objects to avoid reloading on every interaction."""
    resources = {}
    try:
        # Load resources for Churn Prediction
        resources['churn_model'] = tf.keras.models.load_model('model.h5')
        
        # Load resources for Salary Prediction
        resources['salary_model'] = tf.keras.models.load_model('regression_model.h5')

        # Load shared preprocessors
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            resources['onehot_encoder_geo'] = pickle.load(file)
        with open('label_encoder_gender.pkl', 'rb') as file:
            resources['label_encoder_gender'] = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            resources['scaler'] = pickle.load(file)
            
    except FileNotFoundError as e:
        st.error(f"Error loading a required file: {e.name}. Please make sure all model and pickle files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during resource loading: {e}")
        st.stop()
    return resources

resources = load_resources()
scaler = resources['scaler']

# --- SIDEBAR FOR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the Prediction Model",
    ["Customer Churn Prediction", "Salary Estimation"]
)

# --- SHARED USER INPUTS ---
st.sidebar.header("Customer Details")
Geography = st.sidebar.selectbox('Geography', resources['onehot_encoder_geo'].categories_[0], key='geo')
gender = st.sidebar.selectbox('Gender', resources['label_encoder_gender'].classes_, key='gender')
age = st.sidebar.slider('Age', 18, 92, 35, key='age')
balance = st.sidebar.number_input('Balance', value=75000.0, format="%.2f", key='balance')
credit_score = st.sidebar.number_input('Credit Score', value=650, key='credit')
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5, key='tenure')
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1, key='products')
has_cr_card = st.sidebar.selectbox('Has Credit Card?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No', key='card')
is_active_member = st.sidebar.selectbox('Is Active Member?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No', key='active')


# =====================================================================================
# --- CHURN PREDICTION APP ---
# =====================================================================================
if app_mode == "Customer Churn Prediction":
    st.title("📊 Customer Churn Prediction")
    st.markdown("This model predicts whether a customer is likely to churn (leave the bank).")

    # Input specific to churn model
    estimated_salary_churn = st.sidebar.number_input('Estimated Salary', value=100000.0, format="%.2f", key='salary_churn')

    if st.sidebar.button("Predict Churn", type="primary"):
        # --- Prepare data for churn prediction ---
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [resources['label_encoder_gender'].transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary_churn]
        })
        
        geo_arr = resources['onehot_encoder_geo'].transform([[Geography]]).toarray()
        geo_cols = resources['onehot_encoder_geo'].get_feature_names_out(['Geography'])
        geo_encoded_df = pd.DataFrame(geo_arr, columns=geo_cols)

        input_data_combined = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Reorder to match scaler's expectation
        try:
            expected_order = scaler.feature_names_in_
            input_data_reordered = input_data_combined.reindex(columns=expected_order, fill_value=0)
            
            # --- Scale and Predict ---
            input_data_scaled = scaler.transform(input_data_reordered)
            prediction = resources['churn_model'].predict(input_data_scaled)
            prediction_proba = prediction[0][0]

            # --- Display Results ---
            st.header("Prediction Result")
            col1, col2 = st.columns(2)
            col1.metric("Churn Probability", f"{prediction_proba:.2%}")

            if prediction_proba > 0.5:
                col2.error("Conclusion: Customer is likely to CHURN.")
            else:
                col2.success("Conclusion: Customer is likely to STAY.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# =====================================================================================
# --- SALARY ESTIMATION APP ---
# =====================================================================================
elif app_mode == "Salary Estimation":
    st.title("💰 Estimated Salary Prediction")
    st.markdown("This model predicts a customer's estimated salary based on their banking behavior.")

    if st.sidebar.button("Predict Salary", type="primary"):
        # --- Prepare data for salary prediction ---
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [resources['label_encoder_gender'].transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [0]  # Placeholder as required by the scaler
        })

        geo_arr = resources['onehot_encoder_geo'].transform([[Geography]]).toarray()
        geo_cols = resources['onehot_encoder_geo'].get_feature_names_out(['Geography'])
        geo_encoded_df = pd.DataFrame(geo_arr, columns=geo_cols)

        input_data_combined = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Reorder to match scaler's expectation
        try:
            expected_order = scaler.feature_names_in_
            input_data_reordered = input_data_combined.reindex(columns=expected_order, fill_value=0)

            # --- Scale and Predict ---
            input_data_scaled = scaler.transform(input_data_reordered)
            prediction = resources['salary_model'].predict(input_data_scaled)
            predicted_salary = prediction[0][0]

            # --- Display Results ---
            st.header("Prediction Result")
            st.metric(label="Predicted Estimated Salary", value=f"${predicted_salary:,.2f}")
            st.success("The prediction is based on the provided customer data.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
