import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# === Load the trained model ===
model = load_model("stock_price_lstm_model.h5")
print("✅ Model loaded successfully")

# === Load and preprocess the data ===
df = pd.read_csv("TSLA.csv")
data = df[['Close']].values  # Only use 'Close' prices

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Get last 60 days and predict the next day
last_60_days = scaled_data[-60:]
X_input = np.reshape(last_60_days, (1, 60, 1))
predicted_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_scaled)

print(f"📈 Predicted next closing price: ${predicted_price[0][0]:.2f}")

# === Plot the last 60 days + predicted day ===
actual_last_60 = scaler.inverse_transform(last_60_days)

# Build full series: last 60 actual + 1 predicted
full_series = np.append(actual_last_60, predicted_price)
days = list(range(1, 62))  # 60 actual + 1 prediction

plt.figure(figsize=(10, 6))
plt.plot(days[:60], actual_last_60, label="Last 60 Days", linewidth=2)
plt.plot(days[60:], predicted_price, 'ro', label="Predicted Next Day Price")
plt.title("Stock Price Prediction: Next Day Forecast")
plt.xlabel("Day")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("next_day_prediction_plot.png")
plt.show()
