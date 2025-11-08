import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:\\Users\\HP\\Downloads\\Forecast_Project\\energydata.csv")

df.columns.tolist()
df.shape
df.head()

date_col = [c for c in df.columns if 'date' in c.lower()][0]
df[date_col] = pd.to_datetime(df[date_col])

df = df.sort_values(by=date_col)

df = df.set_index(date_col)

target_col = 'Appliances' 
data = df[target_col]

data = data.resample('H').mean().interpolate()

data.index.min()
data.index.max()

plt.figure(figsize=(14,5))
plt.plot(data, color='royalblue')
plt.title("Energy Consumption Over Time", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Appliance Energy Consumption")
plt.grid(True)
plt.show()

rolling_mean = data.rolling(window=24).mean()
rolling_std = data.rolling(window=24).std()

plt.figure(figsize=(14,5))
plt.plot(data, label='Original', color='blue')
plt.plot(rolling_mean, label='Rolling Mean (24h)', color='orange')
plt.plot(rolling_std, label='Rolling Std (24h)', color='green')
plt.title("Rolling Mean & Standard Deviation")
plt.legend()
plt.show()

split_ratio = 0.8
split_point = int(len(data) * split_ratio)
train, test = data[:split_point], data[split_point:]

train.shape
test.shape

fig, axes = plt.subplots(1, 2, figsize=(15,4))
plot_acf(train, lags=40, ax=axes[0])
plot_pacf(train, lags=40, ax=axes[1])
plt.show()

model = ARIMA(train, order=(1,1,1))
fitted_model = model.fit()

fitted_model.summary()

forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps)
forecast.index = test.index

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"\nModel Performance:\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}")

plt.figure(figsize=(14,6))
plt.plot(train[-500:], label="Train Data", color='gray')
plt.plot(test, label="Actual Energy", color='blue')
plt.plot(forecast, label="Forecast", color='red')
plt.title("ARIMA Forecast vs Actual Energy Consumption")
plt.legend()
plt.show()

future_steps = 48
future_forecast = fitted_model.forecast(steps=future_steps)

plt.figure(figsize=(12,4))
plt.plot(future_forecast, color='purple')
plt.title("Forecast for Next 48 Hours")
plt.xlabel("Time Steps Ahead")
plt.ylabel("Predicted Energy Consumption")
plt.show()
