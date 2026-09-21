import streamlit as st
import joblib
import pandas as pd

model = joblib.load('heart_disease_model.pkl')
preprocessor = joblib.load('heart_disease_preprocessor.pkl')

st.title("Heart Disease Risk Predictor")
st.write("Educational tool only — not a diagnostic device.")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [1, 2, 3])

input_df = pd.DataFrame([{
    'age': age, 'sex': 1 if sex == "Male" else 0, 'cp': cp,
    'trestbps': trestbps, 'chol': chol, 'fbs': 1 if fbs == "Yes" else 0,
    'restecg': restecg, 'thalach': thalach, 'exang': 1 if exang == "Yes" else 0,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}])

if st.button("Predict"):
    processed = preprocessor.transform(input_df)
    probability = model.predict_proba(processed)[0][1]
    prediction = "High Risk" if probability >= 0.5 else "Low Risk"
    st.metric("Prediction", prediction)
    st.write(f"Probability of heart disease: {probability:.2%}")