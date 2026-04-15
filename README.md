# ❤️ Heart Disease Risk Predictor

An AI-powered web application that predicts heart disease risk based on lifestyle and medical history using a Random Forest model trained on CDC 2020 data.
local host link --- https://healthpredictor9.streamlit.app/

## 📊 Overview

This project uses machine learning to assess heart disease risk by analyzing 17 health indicators from the CDC's 2020 Behavioral Risk Factor Surveillance System (BRFSS) dataset. The Random Forest model achieves **~91% accuracy** and provides real-time risk assessments.

### Key Features

- ✅ **Real-time predictions** - Instant risk assessment based on user inputs
- 🎯 **High accuracy** - 91% accuracy with 0.83 ROC AUC
- 📈 **Feature importance** - Understand which factors most influence your risk
- 🎨 **User-friendly interface** - Available in both Streamlit and Gradio
- 📊 **Visual feedback** - Risk meters, probability scores, and personalized recommendations

## 🚀 Live Demos



## 📋 Features Analyzed

### Demographics
- Age Category (18-24 to 80+)
- Sex (Male/Female)
- Race

### Physical Health Metrics
- BMI (Body Mass Index)
- Physical Health (days feeling unwell)
- Mental Health (days feeling unwell)
- Sleep Hours per night

### Lifestyle Factors
- Smoking Status
- Alcohol Drinking (heavy drinkers)
- Physical Activity (past 30 days)
- Difficulty Walking

### Medical History
- Diabetes
- Stroke
- Asthma
- Kidney Disease
- Skin Cancer
- General Health Status

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Scikit-learn | Random Forest model training |
| Pandas/NumPy | Data processing |
| Streamlit | Web application framework |
| Gradio | Alternative UI framework |
| Joblib | Model serialization |
| Matplotlib/Seaborn | Data visualization |

## 📁 Project Structure
├── heart-disease-predictor

├── app_streamlit.py # Streamlit web application

├── health_care_predictor.ipynb # Jupyter notebook with model training

│
├── models/ # Saved model artifacts

│ ├── random_forest_model.pkl

│ ├── scaler.pkl

│ ├── label_encoders.pkl

│ ├── feature_names.pkl

│ └── numeric_cols.pkl

│
├── data/

│ └── heart_2020_cleaned.csv # Dataset (not included in repo)

│
├── README.md # Project documentation





🎯 How It Works

Data Preprocessing

One-hot encoding for categorical variables
Standard scaling for numeric features
Label encoding for target variable

Model Training

Random Forest Classifier with 100 estimators
Train/validation/test split: 70%/15%/15%
Stratified sampling to handle class imbalance

Prediction Pipeline

User inputs → Feature engineering → Scaling → Model prediction → Risk assessment

📱 Usage Examples
Low Risk Patient
Age: 25-29

BMI: 22.5

Non-smoker

Regular exercise

Good general health

Result: Low Risk (<30%)

High Risk Patient
Age: 65-69

BMI: 32.0

Smoker

Diabetic

Poor general health

Result: High Risk (>70%)

👥 Authors
Your Name - **Brainy bytes**

🙏 Acknowledgments
CDC for providing the BRFSS dataset

Scikit-learn team for the machine learning tools

Streamlit  the amazing frameworks

