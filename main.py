import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# Load data
# Assuming you have a CSV file with columns 'date' and 'lpg_price'
data = pd.read_csv('lpg_price_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check for stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

test_stationarity(data['lpg_price'])

# If the series is not stationary, difference it
if adfuller(data['lpg_price'], autolag='AIC')[1] > 0.05:
    data['lpg_price_diff'] = data['lpg_price'].diff()
    data = data.dropna()
    test_stationarity(data['lpg_price_diff'])
else:
    data['lpg_price_diff'] = data['lpg_price']

# Automatically find the optimal ARIMA parameters
model = auto_arima(data['lpg_price_diff'], start_p=1, start_q=1,
                   test='adf', max_p=3, max_q=3, m=12,
                   start_P=0, seasonal=True, d=None, D=1, 
                   trace=True, error_action='ignore',  
                   suppress_warnings=True, stepwise=True)

# Fit the ARIMA model
arima_model = ARIMA(data['lpg_price'], order=model.order)
results = arima_model.fit()

# Make predictions
forecast_steps = 36  # 3 years of monthly data
forecast = results.forecast(steps=forecast_steps)

# Create future dates for plotting
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(data.index, data['lpg_price'], label='Historical')
plt.plot(future_dates, forecast, color='red', label='Forecast')
plt.fill_between(future_dates, 
                 forecast - 1.96 * results.forecast(steps=forecast_steps).se_mean,
                 forecast + 1.96 * results.forecast(steps=forecast_steps).se_mean,
                 color='pink', alpha=0.3)
plt.legend()
plt.title('LPG Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Print model summary
print(results.summary())