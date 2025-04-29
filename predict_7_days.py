import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load model
model = load_model("stock_price_lstm_model.h5")

# Load and scale data
df = pd.read_csv("TSLA.csv")
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Use last 60 days for prediction
input_seq = scaled_data[-60:].tolist()  # shape: (60, 1)

predicted_prices_scaled = []

# Predict next 7 days
for _ in range(7):
    x_input = np.array(input_seq[-60:]).reshape(1, 60, 1)
    predicted = model.predict(x_input, verbose=0)
    predicted_prices_scaled.append(predicted[0][0])
    input_seq.append([predicted[0][0]])  # add to sequence for next prediction

# Inverse transform to get actual price
predicted_prices_scaled = np.array(predicted_prices_scaled).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

# Plot last 60 + next 7
last_60_real = scaler.inverse_transform(scaled_data[-60:])
forecast_days = list(range(1, 68))  # 60 actual + 7 forecast

plt.figure(figsize=(12, 6))
plt.plot(forecast_days[:60], last_60_real, label='Last 60 Days')
plt.plot(forecast_days[60:], predicted_prices, 'ro-', label='Next 7 Days Prediction')
plt.title("LSTM Forecast - Next 7 Days")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("7_day_forecast.png")
plt.show()

print("📈 Next 7 day predictions:")
for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i}: ${price[0]:.2f}")
