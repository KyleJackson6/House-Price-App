"""
Streamlit Web App — House Price Prediction

Features:
- Loads pretrained Keras model (HP.keras) and scaler (scaler.pkl)
- Allows user to input features for single prediction
- Supports CSV upload for bulk predictions
- Displays predicted prices, summary metrics, and chart visualization
- Uses multiple Streamlit components: file_uploader, number_input, selectbox, metric, dataframe, download_button, altair_chart, map

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ---------------------- Configuration ----------------------
MODEL_PATH = './models/HP.keras'
SCALER_PATH = './models/scaler.pkl'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Load Model and Scaler ----------------------
@st.cache_resource
def load_house_model():
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        raise

# ---------------------- Prediction Function ----------------------
def predict_price(model, scaler, input_df):
    df_scaled = scaler.transform(input_df)
    prediction = model.predict(df_scaled)
    return np.exp(prediction[0][0])  # reverse log-transform if model trained on log(price)

# ---------------------- App Layout ----------------------
def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    st.title("House Price Prediction App")
    st.markdown("This app uses a pretrained neural network model to estimate house prices.")

    # Load model and scaler
    model, scaler = load_house_model()

    # Sidebar - Single Prediction Inputs
    st.sidebar.header("Enter House Features")

    # Input fields (you can adjust based on your model features)
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=0, max_value=10, value=3)
    bathrooms = st.sidebar.number_input('Bathrooms', min_value=0.0, max_value=10.0, value=2.0)
    sqft_living = st.sidebar.number_input('Living Area (sqft)', min_value=500, max_value=10000, value=2000)
    sqft_lot = st.sidebar.number_input('Lot Size (sqft)', min_value=500, max_value=50000, value=5000)
    floors = st.sidebar.number_input('Floors', min_value=1.0, max_value=3.5, value=1.0)
    condition = st.sidebar.slider('Condition (1–5)', 1, 5, 3)
    grade = st.sidebar.slider('Grade (1–13)', 1, 13, 7)
    yr_built = st.sidebar.number_input('Year Built', min_value=1900, max_value=2025, value=2000)
    zipcode = st.sidebar.number_input('Zipcode', min_value=98000, max_value=99999, value=98178)
    lat = st.sidebar.number_input('Latitude', format="%.6f", value=47.5112)
    long = st.sidebar.number_input('Longitude', format="%.6f", value=-122.257)
    sqft_living15 = st.sidebar.number_input('sqft_living15', min_value=500, max_value=10000, value=2000)

    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'condition': [condition],
        'grade': [grade],
        'yr_built': [yr_built],
        'zipcode': [zipcode],
        'lat': [lat],
        'long': [long],
        'sqft_living15': [sqft_living15]
    })

    if st.sidebar.button('Predict Price'):
        try:
            predicted_price = predict_price(model, scaler, input_data)
            st.metric(label="Predicted House Price", value=f"${predicted_price:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ---------------------- Bulk Predictions ----------------------
    st.markdown("---")
    st.header("Bulk Predictions (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV with house features", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if st.button('Generate Predictions for Uploaded Data'):
                df_scaled = scaler.transform(df)
                preds = model.predict(df_scaled)
                df['predicted_price'] = np.exp(preds)

                st.subheader("Predicted Prices")
                st.dataframe(df.head(50))

                # Chart visualization
                chart = alt.Chart(df.head(100)).mark_circle(size=60).encode(
                    x='sqft_living', y='predicted_price', color='grade', tooltip=['bedrooms','bathrooms','sqft_living','predicted_price']
                )
                st.altair_chart(chart, use_container_width=True)

                # Download predictions
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "predicted_prices.csv", "text/csv")

                # Optional map visualization if lat/lon present
                if {'lat', 'long'}.issubset(df.columns):
                    st.subheader("Map of Properties")
                    st.map(df.rename(columns={'lat': 'latitude', 'long': 'longitude'}).dropna(subset=['latitude','longitude']))

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Footer
    st.markdown("---")
    st.caption("House Price Prediction App using pretrained Keras model.")


if __name__ == '__main__':
    main()