import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("house_data.csv")

# Use 12 features (including zipcode, lat, long)
features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'condition', 'grade', 'yr_built', 'zipcode', 'lat', 'long', 'sqft_living15'
]
X = df[features]
y = np.log1p(df['price'])  # log-transform target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=128,
    verbose=0
)

# Evaluate using inverse-transformed values
y_pred = model.predict(X_test_scaled).flatten()
y_pred_real = np.expm1(y_pred)
y_true_real = np.expm1(y_test)

mae = mean_absolute_error(y_true_real, y_pred_real)
r2 = r2_score(y_true_real, y_pred_real)

print(f"✅ MAE: ${mae:,.2f}")
print(f"✅ R² Score: {r2:.4f}")

# Save model
model.save("HP.keras")
print("✅ Model saved as HP.keras")

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_plot.png")
plt.show()




#Best parameters: {'batch_size': 128, 'epochs': 300, 'model__dropout_rate': 0.2, 'model__learning_rate': 0.005}
#Best cross-validated MAE: 74870.21772848831