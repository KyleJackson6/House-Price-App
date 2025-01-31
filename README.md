# House Price Prediction Model

This repository contains a machine learning model for predicting house prices based on various features. The model is built using TensorFlow and Keras, and it utilizes a neural network for regression tasks. The project includes data preprocessing, model training, hyperparameter tuning, and a function for estimating house prices based on user input.

Table of Contents

Installation
Usage
Model Details
Results
License
Installation

To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
scikit-learn
tensorflow
scikeras
joblib


You can install the required libraries using pip:
'pip install pandas numpy scikit-learn tensorflow scikeras joblib'

Data Preparation:
The dataset house_data.csv should be placed in the specified directory.
The script preprocesses the data by dropping unnecessary columns and filling missing values with the median.
Model Training:
The script defines a neural network model using TensorFlow and Keras.
Hyperparameter tuning is performed using GridSearchCV to find the best parameters for the model.
Model Evaluation:
The model is evaluated on a test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Price Estimation:
The script includes a function estimate_price that allows users to input house features and get a predicted price.
To run the script, use the following command:

Create venv and activate
'pip install pandas numpy scikit-learn tensorflow scikeras joblib'
Run 'python HP.py' and let it train
Then run 'house_price_app.py'

Model Architecture:
Input layer with shape corresponding to the number of features.
Four hidden layers with ReLU activation and dropout for regularization.
Output layer with a single neuron for regression.
Hyperparameters:

Best parameters found by GridSearchCV:
batch_size: 128
epochs: 300
dropout_rate: 0.2
learning_rate: 0.005

Evaluation Metrics:
Best cross-validated MAE: 74870.22
MAE on test set: [Value from your run]
RMSE on test set: [Value from your run]

Results

The model achieved a cross-validated MAE of 74870.22. The final evaluation on the test set yielded the following results:

Mean Absolute Error (MAE): [Value from your run]
Root Mean Squared Error (RMSE): [Value from your run]
License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to contribute to this project by submitting issues or pull requests. For any questions, please contact [Kyle Jackson] at [Kyjack66@gmail.com].
