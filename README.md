# Churn & Estimated Salary Prediction

A Streamlit web app that serves two machine learning models in one unified interface:

- Customer Churn Prediction (classification): Estimates the probability that a customer will churn.
- Estimated Salary Prediction (regression): Predicts a customer's estimated salary from banking profile features.

The app provides a clean sidebar to switch between tasks, interactive inputs, and real-time predictions using pre-trained TensorFlow/Keras models and Scikit-learn preprocessors.

## Table of Contents
- Live Demo
- Features
- Tech Stack
- Screenshots
- Run Locally
- License

## Live Demo
Add your deployment link here when available:

- Web App: https://churn-and-estimatedsalary-prediction-pgfustt389qvbsacftkub6.streamlit.app/

## Features
- Unified interface for both classification and regression.
- Simple sidebar navigation between prediction tasks.
- Interactive, user-friendly inputs (sliders, selects, numeric fields).
- Real-time predictions powered by cached, pre-loaded models and encoders.
- Efficient resource loading with caching to improve responsiveness.

## Tech Stack
- Backend/ML: Python, TensorFlow/Keras, Scikit-learn, Pandas
- Frontend: Streamlit
- Version Control: Git & GitHub

## Screenshots

### Customer Churn Prediction
![Customer Churn Prediction Screenshot](https://github.com/user-attachments/assets/7f395676-c4cc-4d6b-92da-d15f38de20c0)

### Estimated Salary Prediction
![Estimated Salary Prediction Screenshot](https://github.com/user-attachments/assets/39c459ee-4463-4a67-b079-858e69a33b82)

## Run Locally
Follow these steps to set up and run the app locally.

1) Clone the repository

```powershell
# Windows PowerShell
git clone <your-repo-url>
cd <your-repo-folder>
```

2) Create and activate a virtual environment

```powershell
# Windows PowerShell
python -m venv venv
venv\Scripts\activate
```

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3) Install dependencies

```powershell
pip install -r requirements.txt
```

4) Run the Streamlit app

```powershell
streamlit run app.py
```

Your browser should open automatically. If not, copy the local URL printed in the terminal into your browser.

## License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.