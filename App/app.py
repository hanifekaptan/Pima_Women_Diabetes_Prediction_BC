import streamlit as st
import joblib
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the Keras model
model = load_model("model.keras")

# Load the scaler for standardizing input features
scaler = joblib.load("scaler.pkl")

def prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # Create a DataFrame with the user input data
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age]
    })
    # Scale the input data using the loaded scaler
    scaled = scaler.transform(input_data)
    
    # Make a prediction using the model
    prediction = model.predict(scaled)[0][0]
    return prediction

# Set the title of the Streamlit app
st.title("Pima Women Diabetes Prediction")

# Create input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17)  # Number of pregnancies
glucose = st.number_input("Glucose", min_value=0, max_value=200)  # Glucose level
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122)  # Blood pressure
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)  # Skin thickness
insulin = st.number_input("Insulin", min_value=0, max_value=846)  # Insulin level
bmi = st.number_input("BMI", min_value=5.0, max_value=67.1)  # Body mass index
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42)  # Diabetes pedigree function
age = st.number_input("Age", min_value=21, max_value=81)  # Age

# Create a button to trigger prediction
if st.button("Predict"):
    # Get the prediction result based on user input
    result = prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    # Display the prediction result
    st.write(f"Prediction: {result}")
