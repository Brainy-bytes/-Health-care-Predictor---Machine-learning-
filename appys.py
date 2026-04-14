import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("❤️ AI Health Predictor - Heart Disease Risk Assessment")
st.markdown("""
This application uses a **Random Forest model** to predict heart disease risk based on lifestyle and medical history.
Enter your information below to get a personalized risk assessment.
""")

# --- FIXED LOADING LOGIC ---
# Define the path to your models folder (use forward slashes for cross-platform compatibility)
MODEL_PATH = 'models/random_forest_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
TARGET_ENCODER_PATH = 'models/target_encoder.pkl'

# Initialize variables
model = None
scaler = None
encoders = None
target_encoder = None

# Try multiple possible paths
possible_paths = [
    'models/random_forest_model.pkl',
    'model/random_forest_model.pkl',
    'random_forest_model.pkl',
    '../models/random_forest_model.pkl'
]

for path in possible_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        SCALER_PATH = path.replace('random_forest_model.pkl', 'scaler.pkl')
        ENCODER_PATH = path.replace('random_forest_model.pkl', 'label_encoders.pkl')
        TARGET_ENCODER_PATH = path.replace('random_forest_model.pkl', 'target_encoder.pkl')
        break

try:
    # Check if files exist before loading
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
        if os.path.exists(TARGET_ENCODER_PATH):
            target_encoder = joblib.load(TARGET_ENCODER_PATH)
        st.success("✅ Model loaded successfully!")
    else:
        st.error(f"❌ Model files not found. Please ensure you've run your Jupyter Notebook to train and save the model.")
        st.info("Expected files in 'models' folder: random_forest_model.pkl, scaler.pkl, label_encoders.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Define column order (must match training data) ---
COLUMN_ORDER = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 
    'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 
    'Asthma', 'KidneyDisease', 'SkinCancer'
]

# --- Input Forms ---
st.header("📋 1. Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, help="Normal range: 18.5-24.9")
    physical_health = st.slider("Physical Health (Days feeling unwell in past 30 days)", 0, 30, 0)
    mental_health = st.slider("Mental Health (Days feeling unwell in past 30 days)", 0, 30, 0)
    sleep_time = st.number_input("Average Sleep Hours per night", min_value=1, max_value=24, value=7)

with col2:
    st.subheader("Demographics")
    sex = st.selectbox("Sex", ["Female", "Male"])
    age_category = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    race = st.selectbox("Race", ['White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaskan Native', 'Other'])
    gen_health = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])

st.header("🏃 2. Lifestyle & Medical History")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Lifestyle Factors")
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Drinking (Heavy drinkers)", ["No", "Yes"])
    physical_activity = st.selectbox("Physical Activity (Past 30 days)", ["No", "Yes"])
    
    st.subheader("Medical History")
    stroke = st.selectbox("History of Stroke", ["No", "Yes"])
    diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])

with col4:
    st.subheader("Chronic Conditions")
    diabetic = st.selectbox("Diabetic", ["No", "Yes"])
    asthma = st.selectbox("Asthma", ["No", "Yes"])
    kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])

# --- Prediction Logic ---
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True)

if predict_button:
    if model is None or encoders is None:
        st.error("❌ Model files not loaded. Please run your training script first.")
    else:
        # Preprocess inputs using same logic as training
        input_data = {
            'BMI': bmi, 
            'Smoking': smoking, 
            'AlcoholDrinking': alcohol,
            'Stroke': stroke, 
            'PhysicalHealth': physical_health, 
            'MentalHealth': mental_health,
            'DiffWalking': diff_walking, 
            'Sex': sex, 
            'AgeCategory': age_category,
            'Race': race, 
            'Diabetic': diabetic, 
            'PhysicalActivity': physical_activity,
            'GenHealth': gen_health, 
            'SleepTime': sleep_time, 
            'Asthma': asthma,
            'KidneyDisease': kidney_disease, 
            'SkinCancer': skin_cancer
        }
        input_df = pd.DataFrame([input_data])

        # Reorder columns to match training data
        input_df = input_df[COLUMN_ORDER]

        # Apply label encoding to categorical fields
        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    # Handle unknown categories
                    st.warning(f"Unknown value for {col}. Using default.")
                    input_df[col] = 0

        # Scale numeric features if scaler exists
        if scaler is not None:
            numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Make prediction
        try:
            prediction_proba = model.predict_proba(input_df)[0][1]
            prediction_class = model.predict(input_df)[0]
            
            # Determine risk level
            if prediction_proba < 0.3:
                risk_level = "Low"
                risk_color = "🟢"
            elif prediction_proba < 0.7:
                risk_level = "Medium"
                risk_color = "🟡"
            else:
                risk_level = "High"
                risk_color = "🔴"
            
            # Display Results
            st.markdown("---")
            st.header("📊 Results")
            
            # Create metrics row
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Risk Level", f"{risk_color} {risk_level}")
            
            with col_r2:
                st.metric("Probability Score", f"{prediction_proba:.2%}")
            
            with col_r3:
                status = "⚠️ High Risk - Please Consult a Doctor" if prediction_proba > 0.5 else "✅ Low Risk - Maintain Healthy Lifestyle"
                st.metric("Status", status)
            
            # Progress bar for risk level
            st.subheader("Risk Assessment Meter")
            st.progress(prediction_proba)
            
            # Risk interpretation
            if prediction_proba > 0.5:
                st.error(f"""
                ### ⚠️ High Risk Assessment ({prediction_proba:.1%})
                Based on the information provided, you have a **HIGH risk** of heart disease.
                
                **Recommendations:**
                - Consult with a healthcare provider
                - Monitor blood pressure and cholesterol levels
                - Consider lifestyle modifications
                - Regular health check-ups recommended
                """)
            else:
                st.success(f"""
                ### ✅ Low Risk Assessment ({prediction_proba:.1%})
                Based on the information provided, you have a **LOW risk** of heart disease.
                
                **Keep up the good work!**
                - Maintain regular physical activity
                - Continue healthy eating habits
                - Regular health monitoring is still recommended
                """)
            
            # Show key risk factors based on input
            st.subheader("📈 Your Key Health Indicators")
            
            risk_factors = []
            if bmi > 30:
                risk_factors.append(("High BMI", f"{bmi:.1f} (Obese range)", "⚠️"))
            elif bmi > 25:
                risk_factors.append(("BMI", f"{bmi:.1f} (Overweight)", "🟡"))
            else:
                risk_factors.append(("BMI", f"{bmi:.1f} (Healthy)", "✅"))
            
            if smoking == "Yes":
                risk_factors.append(("Smoking", "Current smoker", "⚠️"))
            
            if diabetic != "No":
                risk_factors.append(("Diabetes", diabetic, "⚠️"))
            
            if physical_activity == "No":
                risk_factors.append(("Physical Activity", "Inactive lifestyle", "🟡"))
            
            if physical_health > 10:
                risk_factors.append(("Physical Health", f"{physical_health} poor days/month", "🟡"))
            
            for factor, detail, icon in risk_factors:
                st.write(f"{icon} **{factor}:** {detail}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- Feature Importance Visualization (if model loaded) ---
if model is not None:
    st.markdown("---")
    st.header("📊 Model Insights")
    
    with st.expander("View Feature Importance Analysis"):
        st.markdown("""
        ### Top Factors Influencing Heart Disease Risk
        
        Based on the trained Random Forest model, these are the most important predictors:
        """)
        
        # Create feature importance chart
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = COLUMN_ORDER
            
            # Create DataFrame for importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Display top 10
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = importance_df.head(10)
            colors = plt.cm.RdYlGn_r(top_features['importance'].values)
            bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Most Important Features for Heart Disease Prediction')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, val in zip(bars, top_features['importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("💡 **Insight:** Age, General Health, BMI, and Difficulty Walking are the strongest predictors of heart disease risk in this model.")
        else:
            st.info("Feature importance data not available for this model.")

# --- Sidebar with additional info ---
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Training Data: CDC 2020 Heart Disease Dataset
    - Features: 17 health indicators
    
    **Risk Factors Considered:**
    - Demographics (Age, Sex, Race)
    - Lifestyle (Smoking, Alcohol, Exercise)
    - Medical History (Diabetes, Stroke, Kidney Disease)
    - Physical Health Metrics (BMI, Sleep, Mental Health)
    
    **Important Note:**
    This is a predictive tool based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for medical decisions.
    """)
    
    st.divider()
    
    st.header("📈 Understanding Your Results")
    st.markdown("""
    - **Low Risk (<30%)**: Your lifestyle factors suggest lower probability of heart disease
    - **Medium Risk (30-70%)**: Some risk factors present - consider lifestyle modifications
    - **High Risk (>70%)**: Multiple risk factors detected - consult a healthcare provider
    """)
    
    st.divider()
    
    st.caption("Built with ❤️ using Streamlit | Data Source: CDC 2020")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Always consult a qualified healthcare professional for medical advice.")