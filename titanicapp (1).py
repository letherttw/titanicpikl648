import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("Titanic Survival Predictor")

st.write("Enter passenger details below:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=50.0)

# Convert to dataframe (IMPORTANT — same as churn notebook)
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Fare': [fare]
})

# Prediction button (same logic as notebook)
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.success("This passenger is likely to SURVIVE")
    else:
        st.error("This passenger is NOT likely to survive")
