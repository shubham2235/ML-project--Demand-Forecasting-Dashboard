# src/forecasting.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import os

warnings.filterwarnings("ignore")

def decompose_time_series(series, product_id, period=52):
    if len(series) >= 2:
        period = min(period, len(series) // 2)
        result = seasonal_decompose(series, model="additive", period=period)
        result.plot()
        plt.suptitle(f"Seasonal Decomposition for Product {product_id}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not enough data to decompose for {product_id}")

def plot_acf_pacf(series, product_id, lags=20):
    max_lags = min(lags, len(series) // 2 - 1)  # Adjust lags based on available data
    if max_lags < 1:
        print(f"Not enough data to plot ACF/PACF for Product {product_id}")
        return

    plot_acf(series, lags=max_lags)
    plt.title(f"ACF for Product {product_id}")
    plt.tight_layout()
    plt.show()

    plot_pacf(series, lags=max_lags)
    plt.title(f"PACF for Product {product_id}")
    plt.tight_layout()
    plt.show()

def arima_forecast(series, order=(5, 1, 0), steps=15):
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast

def forecast_top_products(weekly_data, top_products, steps=15):
    forecasts = {}
    for product in top_products:
        product_series = weekly_data[weekly_data["StockCode"] == product].set_index("Week")["Quantity"]
        if len(product_series) > steps:
            try:
                forecast = arima_forecast(product_series[:-steps], steps=steps)
                forecasts[product] = forecast.values
            except Exception as e:
                print(f"ARIMA failed for Product {product} with error: {e}")
    return forecasts

def save_forecast_csv(forecasts_dict, filename="forecasted_quantities_top_products.csv"):
    df = pd.DataFrame(forecasts_dict)
    df.to_csv(filename, index=False)
    print(f"Forecasts saved to {filename}")

def plot_forecast(series, forecast, product_id, code_to_name=None):
    if forecast is None or len(forecast) == 0:
        print(f"Skipping plot for {product_id} due to empty forecast.")
        return

    # Convert code to name if mapping provided
    product_label = code_to_name.get(product_id, product_id) if code_to_name else product_id

    train = series[:-len(forecast)]
    test = series[-len(forecast):]

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training Data")
    plt.plot(test.index, test, label="Actual", color="blue")
    plt.plot(test.index, forecast, label="Forecast", color="red")
    plt.title(f"Forecasting for Product: {product_label}")
    plt.xlabel("Week")
    plt.ylabel("Quantity Sold")
    plt.legend()
    plt.tight_layout()
    plt.show()

