import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import streamlit as st
import tensorflow as tf

# --- LOAD THE TRAINED REGRESSION MODEL AND PREPROCESSORS ---
# IMPORTANT: Make sure 'regression_model.h5' is your trained model for predicting salary.
try:
    model = tf.keras.models.load_model('regression_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# Load the encoders and scaler
# These should be the same encoders used when training your model.
# The scaler should have been fitted on the training data WITHOUT the 'EstimatedSalary' column.
try:
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Could not find the required .pkl files. Please make sure 'onehot_encoder_geo.pkl', 'label_encoder_gender.pkl', and 'scaler.pkl' are in the same directory.")
    st.stop()


# --- STREAMLIT APP INTERFACE ---
st.set_page_config(layout="wide")
st.title('💰 Estimated Salary Prediction')
st.markdown("Enter the customer's details to predict their estimated salary.")

st.sidebar.header("Customer Details")

# --- USER INPUT ---
# We get all features EXCEPT for the one we want to predict (EstimatedSalary)
Geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, 35)
balance = st.sidebar.number_input('Balance', value=0.0, format="%.2f")
credit_score = st.sidebar.number_input('Credit Score', value=600)
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.sidebar.selectbox('Is Active Member?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')


# --- DATA PREPARATION FOR PREDICTION ---
if st.sidebar.button('Predict Salary', type="primary"):
    # 1. Create a DataFrame from the user's input (excluding the target variable)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [0] # Add placeholder for the missing 'Exited' column
        # NOTE: 'EstimatedSalary' is NOT included here
    })

    # 2. One-hot encode the 'Geography' feature
    try:
        geo_arr = onehot_encoder_geo.transform([[Geography]]).toarray()
        geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
        geo_encoded_df = pd.DataFrame(geo_arr, columns=geo_cols)
    except Exception as e:
        st.error(f"Error during one-hot encoding: {e}")
        st.stop()


    # 3. Combine the one-hot encoded columns with the rest of the input data
    input_data_combined = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # 4. Reorder columns to match the exact order the scaler was trained on
    try:
        # Get the feature names from the scaler object
        expected_order = scaler.feature_names_in_
        input_data_reordered = input_data_combined.reindex(columns=expected_order, fill_value=0)
    except AttributeError:
        st.error("The scaler object does not contain feature names. Please retrain and save the scaler with a version of scikit-learn that supports this (0.24+).")
        st.stop()


    # 5. Scale the input data using the pre-fitted scaler
    try:
        input_data_scaled = scaler.transform(input_data_reordered)
    except Exception as e:
        st.error(f"Error during data scaling: {e}")
        st.error("Please ensure the scaler was fitted on data with the same columns and order.")
        st.stop()

    # 6. Make the prediction
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    # --- DISPLAY THE RESULT ---
    st.header("Prediction Result")
    st.metric(label="Predicted Estimated Salary", value=f"${predicted_salary:,.2f}")
    st.success("The prediction is based on the provided customer data.")

else:
    st.info("Please enter customer details in the sidebar and click 'Predict Salary'.")

