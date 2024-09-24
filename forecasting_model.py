import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv(r'Projects/Python/Project1/bank-full.csv', sep=';')

# Handle missing values by using forward fill
data.ffill(inplace=True)

# Remove outliers in the 'balance' column using Z-score (values beyond 3 standard deviations are considered outliers)
data = data[(np.abs(stats.zscore(data['balance'])) < 3)]

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Feature engineering: convert 'month' to numeric (0 for January, 11 for December)
data['Month'] = pd.to_datetime(data['month'], format='%b').dt.month

# Define features (X) and target variable (y)
X = data[['age', 'Month', 'campaign', 'pdays', 'previous']]
y = data['balance']  # Predicting balance

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)

# Model 2: Decision Tree Regressor
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)
predictions_dt = model_dt.predict(X_test)

# Model 3: TensorFlow Neural Network
model_tf = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(32, activation='relu'),                                  # Second hidden layer
    Dense(1)                                                       # Output layer
])

# Compile the model with Adam optimizer and Mean Squared Error as the loss function
model_tf.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (50 epochs, batch size of 32, use 10% of training data for validation)
history = model_tf.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Predict balance for the test set
predictions_tf = model_tf.predict(X_test).flatten()

# Visualization of the actual vs predicted balance values for all three models
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Balance', color='blue')
plt.plot(predictions_lr, label='Linear Regression Predictions', linestyle='--', color='red')
plt.plot(predictions_dt, label='Decision Tree Predictions', linestyle='--', color='green')
plt.plot(predictions_tf, label='TensorFlow Predictions', linestyle='--', color='orange')
plt.legend()
plt.title('Balance Forecasting')
plt.xlabel('Data Points')
plt.ylabel('Balance')
plt.show()

# Evaluate Model 1: Linear Regression
mae_lr = mean_absolute_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)
print(f'Linear Regression - MAE: {mae_lr:.2f}, R2: {r2_lr:.2f}')

# Evaluate Model 2: Decision Tree Regressor
mae_dt = mean_absolute_error(y_test, predictions_dt)
r2_dt = r2_score(y_test, predictions_dt)
print(f'Decision Tree - MAE: {mae_dt:.2f}, R2: {r2_dt:.2f}')

# Evaluate Model 3: TensorFlow Neural Network
mae_tf = mean_absolute_error(y_test, predictions_tf)
r2_tf = r2_score(y_test, predictions_tf)
print(f'TensorFlow Model - MAE: {mae_tf:.2f}, R2: {r2_tf:.2f}')