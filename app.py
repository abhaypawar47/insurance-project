import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("insurance_model.pkl", "rb"))

st.title("🏥 Medical Insurance Cost Predictor")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0)
children = st.number_input("Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict"):
    sex = 0 if sex == "male" else 1
    smoker = 1 if smoker == "yes" else 0
    region_map = {
        "southwest": [1,0,0],
        "southeast": [0,1,0],
        "northwest": [0,0,1],
        "northeast": [0,0,0]
    }

    final = [age, sex, bmi, children, smoker] + region_map[region]
    final = np.array(final).reshape(1, -1)

    result = model.predict(final)
    st.success(f"Predicted Insurance Price: ₹ {round(result[0],2)}")
