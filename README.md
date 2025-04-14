# House Price Prediction App

This repository contains a machine learning model for predicting house prices in **King County, Washington** (Seattle area) based on 12 key home features. The model is built using **TensorFlow/Keras** and deployed via a clean **Streamlit** web interface.

---

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Training (HP.py)](#training-hppy)
- [Prediction UI (app.py)](#prediction-ui-apppy)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Structure](#structure)
- [License](#license)

---

##  Installation

Create a Python 3.10 virtual environment (required for TensorFlow compatibility):

```bash
python3.10 -m venv venv310
source venv310/bin/activate  # or .\venv310\Scripts\activate on Windows
```

Install required libraries:

```bash
pip install -r requirements.txt
```

### requirements.txt:
```txt
streamlit
tensorflow==2.15.0
scikit-learn==1.3.2
numpy
pandas
```

---

## Data

- Dataset: `house_data.csv`
- Location: **King County, Washington, USA**
- Features used:  
  `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `condition`, `grade`, `yr_built`, `zipcode`, `lat`, `long`, `sqft_living15`
- Target: `price` (log-transformed during training)

---

## Training (`HP.py`)

This script preprocesses the data, trains the neural network model using:
- **Batch Size**: 128
- **Epochs**: 300
- **Dropout Rate**: 0.2
- **Learning Rate**: 0.005

```python
# Highlight: Feature list
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'condition', 'grade', 'yr_built', 'zipcode', 'lat', 'long', 'sqft_living15'
]
```

- Target values are log-transformed using `np.log1p(y)`
- Final evaluation inverts the transform using `np.expm1()`
- Model and scaler are saved as `HP.keras` and `scaler.pkl`

To retrain the model locally:
```bash
python HP.py
```
This will regenerate `HP.keras`, `scaler.pkl`, and `training_loss_plot.png`.

---

## Prediction UI (`app.py`)

- Built with **Streamlit**
- Users enter home features in a form
- `zipcode` is selected from a dropdown
- `lat` and `long` are **auto-filled** based on selected `zipcode`
- Output: estimated home price in USD

```python
# Auto-fill lat/long based on zipcode
lat, long = zip_lat_long[zipcode]['lat'], zip_lat_long[zipcode]['long']
```

To launch the app locally:
```bash
streamlit run app.py
```

---

## Model Architecture
- Input layer with 12 features
- Dense (128), ReLU
- Dropout (0.2)
- Dense (64), ReLU
- Output: Dense (1) for regression

---

## Evaluation

| Metric | Value |
|--------|-------|
| Best Cross-Validated MAE | ~74,870 |
| Final Test MAE | ~89,638.35 |
| Final Test R² Score | ~0.7739 |

Plot: `training_loss_plot.png` shows training vs validation loss.

---

## Structure
```
House-Price-App/
├── app.py                  # ✅ Main Streamlit UI
├── HP.py                   # ✅ Model training script
├── house_data.csv          # ✅ Dataset
├── HP.keras                # ✅ Trained model
├── scaler.pkl              # ✅ Scaler for user input
├── training_loss_plot.png  # ✅ Visual of training performance
├── requirements.txt        # ✅ Cleaned dependency list
├── README.md               # ✅ documentation
└── Dockerfile              # ✅ For deployment
```
---

## License

MIT License. See `LICENSE` file.

---

For questions or contributions, contact **Kyle Jackson** at **Kyjack66@gmail.com**.

