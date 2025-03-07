from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import joblib  # To load scaler if saved as .pkl
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = load_model('/Users/kylejackson/Documents/House Prices Code/HP.keras')
scaler = joblib.load('/Users/kylejackson/Documents/House Prices Code/scaler.pkl') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve inputs from the form
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        sqft_lot = float(request.form['sqft_lot'])
        floors = float(request.form['floors'])
        condition = int(request.form['condition'])
        grade = int(request.form['grade'])
        sqft_above = int(request.form['sqft_above'])
        sqft_basement = int(request.form['sqft_basement'])
        yr_built = int(request.form['yr_built'])
        yr_renovated = int(request.form['yr_renovated'])
        sqft_living15 = float(request.form['sqft_living15'])
        sqft_lot15 = float(request.form['sqft_lot15'])
        
        # Create a DataFrame with the correct feature names
        features = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'condition': condition,
            'grade': grade,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'sqft_living15': sqft_living15,
            'sqft_lot15': sqft_lot15

        }])
        
        print("Features received from form:", features)

        # Scale the features
        scaled_features = scaler.transform(features)
        print("Scaled features:", scaled_features)

        # Predict price
        predicted_price = model.predict(scaled_features)[0][0]
        print("Predicted price:", predicted_price)

        # Return the result
        return render_template('results.html', result=f"The estimated price is: ${predicted_price:,.2f}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Log the full traceback
        return render_template('index.html', result="Error occurred during prediction. Please check inputs.")

if __name__ == '__main__':
    app.run(debug=True)
