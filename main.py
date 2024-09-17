

# Load data
# Assuming you have a CSV file with columns 'date', 'lpg_price', 'gdp', and 'co2_emissions'
data = pd.read_csv('lpg_price_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Function to check stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

# Check stationarity for each series
for column in ['lpg_price', 'gdp', 'co2_emissions']:
    print(f"\nTesting stationarity for {column}")
    test_stationarity(data[column])

# Difference series if not stationary
for column in ['lpg_price', 'gdp', 'co2_emissions']:
    if adfuller(data[column], autolag='AIC')[1] > 0.05:
        data[f'{column}_diff'] = data[column].diff()
        print(f"\n{column} differenced. Testing stationarity:")
        test_stationarity(data[f'{column}_diff'].dropna())
    else:
        data[f'{column}_diff'] = data[column]

# Drop NaN values created by differencing
data = data.dropna()

# Normalize exogenous variables
scaler = StandardScaler()
exog_vars = ['gdp_diff', 'co2_emissions_diff']
data[exog_vars] = scaler.fit_transform(data[exog_vars])

# Automatically find the optimal ARIMAX parameters
model = auto_arima(data['lpg_price_diff'], exogenous=data[exog_vars],
                   start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                   start_P=0, seasonal=True, d=None, D=1, trace=True,
                   error_action='ignore', suppress_warnings=True, stepwise=True)

# Fit the ARIMAX model
arimax_model = SARIMAX(data['lpg_price'], exog=data[exog_vars],
                       order=model.order, seasonal_order=model.seasonal_order)
results = arimax_model.fit()

# Make predictions
forecast_steps = 36  # 3 years of monthly data

# Create future exogenous variables (you may want to use more sophisticated forecasting for these)
future_exog = pd.DataFrame({
    'gdp_diff': [data['gdp_diff'].mean()] * forecast_steps,
    'co2_emissions_diff': [data['co2_emissions_diff'].mean()] * forecast_steps
})

forecast = results.forecast(steps=forecast_steps, exog=future_exog)

# Create future dates for plotting
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(data.index, data['lpg_price'], label='Historical')
plt.plot(future_dates, forecast, color='red', label='Forecast')
plt.fill_between(future_dates, 
                 forecast - 1.96 * results.forecast(steps=forecast_steps, exog=future_exog).se_mean,
                 forecast + 1.96 * results.forecast(steps=forecast_steps, exog=future_exog).se_mean,
                 color='pink', alpha=0.3)
plt.legend()
plt.title('LPG Price Forecast with ARIMAX')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Print model summary
print(results.summary())

# Print the impact of exogenous variables
print("\nImpact of exogenous variables:")
print(results.pvalues[['gdp_diff', 'co2_emissions_diff']])