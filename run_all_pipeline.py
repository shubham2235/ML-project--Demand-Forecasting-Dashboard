import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data, merge_data
from src.preprocessing import (
    preprocess_transactional_data,
    preprocess_product_info,
    preprocess_customer_data,
    feature_engineering
)
from src.eda import (
    customer_level_analysis,
    item_level_analysis,
    transaction_level_analysis,
    plot_top_products,
    plot_monthly_revenue_trends,
    plot_correlation_heatmap
)
from src.visualization import (
    plot_top_selling_products,
    plot_top_revenue_products,
    plot_monthly_revenue_trends as vis_monthly,
    plot_correlation_heatmap as vis_corr
)
from src.ml_modeling import prepare_ml_features, train_models, evaluate_models
from src.forecasting import (
    decompose_time_series, plot_acf_pacf,
    forecast_top_products, plot_forecast
)
from src.final_forecast import generate_final_forecast, visualize_forecast
from src.evaluation import evaluate_model, compare_models
import time

# ------------------ Load and Preprocess Data ------------------ #
print("\nLoading and preprocessing data...")
product_info, trans_1, trans_2, customer_demo = load_data()

trans_1 = preprocess_transactional_data(trans_1)
trans_2 = preprocess_transactional_data(trans_2)
product_info = preprocess_product_info(product_info)
customer_demo = preprocess_customer_data(customer_demo)

merged_data = merge_data(product_info, trans_1, trans_2, customer_demo)
final_data = feature_engineering(merged_data)

# ------------------ EDA ------------------ #
print("\nPerforming Exploratory Data Analysis...")
customer_level_analysis(final_data)
top_selling, top_revenue = item_level_analysis(final_data)
transaction_level_analysis(final_data)

plot_top_products(top_selling, top_revenue)
plot_monthly_revenue_trends(final_data)
plot_correlation_heatmap(final_data)

plot_top_selling_products(top_selling)
plot_top_revenue_products(top_revenue)
vis_monthly(final_data)
vis_corr(final_data)

plt.show(block=False)
time.sleep(30)

# ------------------ ML Modeling ------------------ #
print("\nTraining ML models...")
features, target = prepare_ml_features(final_data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

decision_tree, xgb = train_models(X_train, y_train)
metrics = evaluate_models(X_test, y_test, decision_tree, xgb)

# ------------------ Time Series Forecasting ------------------ #
print("\nForecasting top products using ARIMA...")
weekly_data = final_data.groupby(["StockCode", "Week"])["Quantity"].sum().reset_index()
top_products = top_selling.head(3).index.tolist()

for product_id in top_products:
    series = weekly_data[weekly_data["StockCode"] == product_id].set_index("Week")["Quantity"]
    decompose_time_series(series, product_id)
    plot_acf_pacf(series, product_id)
    forecast = forecast_top_products(weekly_data, top_products, steps=15)
    if product_id in forecast:
        plot_forecast(series, forecast[product_id], product_id)

plt.show(block=False)
time.sleep(30)

# ------------------ Final Forecasts (XGB & DT Models) ------------------ #
print("\nGenerating final forecast using ML models...")
top_df = final_data[final_data["StockCode"].isin(top_products)]
forecast_df = generate_final_forecast(final_data, top_df)
print(forecast_df.head())

# Optional: visualize one product forecast
example_product = top_products[0]
product_target = final_data[final_data["StockCode"] == example_product]["Quantity"]
forecast_for_example = forecast_df[forecast_df["StockCode"] == example_product]
visualize_forecast(example_product, product_target, forecast_for_example)

plt.show()
