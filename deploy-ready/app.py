import streamlit as st
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

# Load dataset to generate lat/long per zipcode
df = pd.read_csv("house_data.csv")
zip_lat_long = df.groupby("zipcode")[["lat", "long"]].mean().round(6).to_dict(orient="index")

# Load TensorFlow model and scaler (trained with 12 features)
model = tf.keras.models.load_model("HP_model")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit page setup
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the details below to predict the house price.")

# Input form
with st.form("prediction_form"):
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Sqft Living", min_value=0, value=2000)
    sqft_lot = st.number_input("Sqft Lot", min_value=0, value=5000)
    floors = st.number_input("Floors", min_value=0.0, value=1.0, step=0.5)
    condition = st.selectbox("Condition (1–5)", list(range(1, 6)))
    grade = st.selectbox("Grade (1–13)", list(range(1, 14)))
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990)
    sqft_living15 = st.number_input("Sqft Living15", min_value=0, value=1800)
    zipcode = st.selectbox("Region (Zipcode)", sorted(zip_lat_long.keys()))

    submit = st.form_submit_button("Predict Price 💰")

# On submit
if submit:
    if zipcode not in zip_lat_long:
        st.error("❌ Selected zipcode is missing coordinates. Please try another.")
    else:
        lat = zip_lat_long[zipcode]['lat']
        long = zip_lat_long[zipcode]['long']

        input_data = np.array([[  # Must match training order
            bedrooms, bathrooms, sqft_living, sqft_lot, floors,
            condition, grade, yr_built, zipcode, lat, long, sqft_living15
        ]])

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model(scaled_data).numpy()[0][0]
        prediction = np.expm1(prediction)  # reverse log1p from training

        st.success(f"🏡 Estimated House Price: **${round(prediction, 2):,.2f}**")

