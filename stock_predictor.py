# Stock Price Predictor using LSTM
# This script predicts the next day's stock price using LSTM and visualizes the results.
# It also saves the model and can predict the next 7 days' prices.
# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("TSLA.csv")
df = df[['Date', 'Close']]  # Only keep Date and Close price
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot the closing price
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Tesla Stock Closing Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Normalize the Close prices (LSTM loves values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Create sequences
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape X to be [samples, time_steps, features] for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into training and testing sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # Output layer (predicting 1 value)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# ==== PREDICT AND VISUALIZE ====
import matplotlib.pyplot as plt

# Predict on the test data
predicted_prices = model.predict(X_test)

# Inverse transform the predicted and actual prices to get real values
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price', linewidth=2)
plt.plot(predicted_prices, label='Predicted Price', linestyle='--')
plt.title("Stock Price Prediction vs Actual (Test Data)")
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output_plot.png")  # Save to file
plt.show()

# ==== SAVE THE TRAINED MODEL ====
model.save("stock_price_lstm_model.h5")
print("✅ Model saved as stock_price_lstm_model.h5")
