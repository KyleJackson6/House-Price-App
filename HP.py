import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
import joblib


# Load and preprocess data
data = pd.read_csv('/Users/kylejackson/Documents/House Prices Code/house_data.csv')
data.drop(columns=['id','date','waterfront','view','zipcode','lat','long',], inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)


# Define features (X) and target (y)
X = data.drop(columns=['price'])  # Dropping the target variable
y = data['price']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

def build_model(dropout_rate=0.2, learning_rate=0.005):
    model = Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Wrap with KerasRegressor
model = KerasRegressor(model=build_model, verbose=1)

param_grid = {
    'batch_size': [64, 128, 256],
    'epochs': [100, 200, 300, 500],
    'model__dropout_rate': [0.1, 0.2, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.005, 0.01, 0.02]
}

# Train with GridSearchCV to find best parameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_result = grid.fit(X, y)

# Display best parameters and score
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best cross-validated MAE: {-grid_result.best_score_}")

# Step 5: Final evaluation on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = grid_result.best_estimator_
best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'], verbose=1)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE) on test set: {mae}")
print(f"Root Mean Squared Error (RMSE) on test set: {rmse}")

# Save the best model
best_model.model_.save('/Users/kylejackson/Documents/House Prices Code/HP.keras')
scaler_path = '/Users/kylejackson/Documents/House Prices Code/scaler.pkl'
joblib.dump(scaler, scaler_path)

# Define a function for user input and price estimation
def estimate_price(model, scaler):
    input_data = {}
    print("Please enter the following details about the house:")
    for col in data.drop(columns=['price']).columns:
        value = float(input(f"{col}: "))
        input_data[col] = value

    # Create a DataFrame for user input and scale it
    user_df = pd.DataFrame([input_data])
    user_df_scaled = scaler.transform(user_df)

    # Predict price
    predicted_price = model.predict(user_df_scaled)
    print(f"Estimated House Price: ${predicted_price[0]:,.2f}")

# Estimate house price based on user input
estimate_price(best_model, scaler)


#Best parameters: {'batch_size': 128, 'epochs': 300, 'model__dropout_rate': 0.2, 'model__learning_rate': 0.005}
#Best cross-validated MAE: 74870.21772848831