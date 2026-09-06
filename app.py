import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart_model.pkl')
scaler=joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

st.title("Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease based on user input.")


age=st.slider("Age", 18, 100, 40)
sex=st.selectbox("Sex", ["Male", "Female"])
chest_pain=st.selectbox("Chest Pain Type", ["Typical Angina(TA)", "Atypical Angina(ATA)", "Non-Anginal Pain(NAP)", "Asymptomatic(ASY)"])
resting_bp=st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol=st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0 , 1])
resting_ecg=st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality(ST)", "Left Ventricular Hypertrophy(LVH)"])
max_hr=st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina=st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak=st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope=st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    raw_input={
        'Age': age,
        'Sex': 1 if sex == "Male" else 0,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': 1 if exercise_angina == "Yes" else 0,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope      
    }
    input_df = pd.DataFrame([raw_input])


    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that you are likely to have heart disease.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease.")      
