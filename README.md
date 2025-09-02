 Churn & EstimatedSalary Prediction
This project is a multi-functional web application built with Streamlit that deploys two distinct machine learning models within a single, unified interface:

Customer Churn Prediction: A classification model that predicts the probability of a customer churning (leaving the bank).

Salary Estimation: A regression model that estimates a customer's salary based on their banking details and behavior.

The application features a sidebar navigation menu to seamlessly switch between the two prediction tasks, providing an interactive and user-friendly experience for making real-time predictions.

🚀 Live Demo
You can try the live, deployed application here:

[**Click Here to Access the Web App**](https://churn-and-estimatedsalary-prediction-pgfustt389qvbsacftkub6.streamlit.app/)

📸 App Screenshot
Customer Churn Prediction
<img width="1905" height="982" alt="Screenshot 2025-09-02 205553" src="https://github.com/user-attachments/assets/7f395676-c4cc-4d6b-92da-d15f38de20c0" />
Estimated Salary Prediction
<img width="1906" height="974" alt="Screenshot 2025-09-02 205613" src="https://github.com/user-attachments/assets/39c459ee-4463-4a67-b079-858e69a33b82" />


✨ Features
Unified Interface: A single Streamlit application serves both the classification and regression models.

Easy Navigation: A clean sidebar menu allows users to select the desired prediction model.

Interactive UI: User-friendly sliders, select boxes, and number inputs for entering customer data.

Real-Time Predictions: The backend, powered by trained TensorFlow/Keras models, provides instant predictions.

Dynamic Inputs: The UI intelligently shows the correct input fields based on the selected model.

Cached Resources: Models and preprocessors are loaded once and cached for improved performance.

🛠️ Technology Stack
Backend & ML: Python, TensorFlow, Keras, Scikit-learn, Pandas

Frontend Web App: Streamlit

Version Control: Git & GitHub

⚙️ How to Run Locally
To set up and run this project on your local machine, follow these steps:

1. Clone the Repository

git clone [https://github.com/pixelbyteee/Churn-And-EstimatedSalary-Prediction.git](https://github.com/pixelbyteee/Churn-And-EstimatedSalary-Prediction.git)
cd Churn-And-EstimatedSalary-Prediction


2. Create and Activate a Virtual Environment
It's highly recommended to create a virtual environment to manage dependencies and avoid conflicts.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

4. Run the Streamlit App
Launch the application using the Streamlit CLI. Make sure you are in the project's root directory.

streamlit run app.py

The application should now be open and running in your default web browser!
