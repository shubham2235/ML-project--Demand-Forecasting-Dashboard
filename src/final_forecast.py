""" import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def generate_final_forecast(merged_data, top_products, forecast_horizon=15):
    forecast_results = pd.DataFrame()
    top_products = top_products["StockCode"].unique().tolist()

    for product_id in top_products:
        product_data = merged_data[merged_data["StockCode"] == product_id]

        if product_data.empty:
            continue

        if "CustomerAge" not in product_data.columns:
            product_data["CustomerAge"] = np.random.randint(18, 65, size=len(product_data))
        if "CustomerIncome" not in product_data.columns:
            product_data["CustomerIncome"] = np.random.randint(20000, 100000, size=len(product_data))

        product_features = product_data[["CustomerID", "TotalRevenue", "UnitPrice", "CustomerAge", "CustomerIncome"]]
        product_target = product_data["Quantity"]

        if product_features.empty or len(product_target) == 0:
            continue

        decision_tree = DecisionTreeRegressor(random_state=42)
        decision_tree.fit(product_features, product_target)

        xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
        xgb.fit(product_features, product_target)

        future_features = product_features.iloc[-forecast_horizon:].copy()
        future_features.index = range(product_features.shape[0], product_features.shape[0] + forecast_horizon)

        if len(future_features) < forecast_horizon:
            future_features = pd.concat(
                [future_features] * (forecast_horizon // len(future_features) + 1), ignore_index=True
            )[:forecast_horizon]

        dt_forecast = decision_tree.predict(future_features)
        xgb_forecast = xgb.predict(future_features)

        forecast_df = pd.DataFrame({
            "StockCode": product_id,
            "Week": range(1, forecast_horizon + 1),
            "DecisionTree_Forecast": dt_forecast,
            "XGBoost_Forecast": xgb_forecast
        })

        forecast_results = pd.concat([forecast_results, forecast_df], ignore_index=True)

    return forecast_results


def visualize_forecast(example_product, product_target, forecast_df):
    forecast_horizon = len(forecast_df)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(product_target)), product_target, label="Historical Data", color="blue")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             forecast_df["DecisionTree_Forecast"], label="Decision Tree Forecast", color="green")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             forecast_df["XGBoost_Forecast"], label="XGBoost Forecast", color="red")
    plt.xlabel("Weeks")
    plt.ylabel("Quantity")
    plt.title(f"Forecast for Product {example_product}")
    plt.legend()
    plt.tight_layout()
    plt.show()
 """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def generate_final_forecast(merged_data, top_products, forecast_horizon=15):
    forecast_results = pd.DataFrame()
    top_products = top_products["StockCode"].unique().tolist()

    for product_id in top_products:
        product_data = merged_data[merged_data["StockCode"] == product_id]

        if product_data.empty:
            continue

        if "CustomerAge" not in product_data.columns:
            product_data["CustomerAge"] = np.random.randint(18, 65, size=len(product_data))
        if "CustomerIncome" not in product_data.columns:
            product_data["CustomerIncome"] = np.random.randint(20000, 100000, size=len(product_data))

        product_features = product_data[["CustomerID", "TotalRevenue", "UnitPrice", "CustomerAge", "CustomerIncome"]]
        product_target = product_data["Quantity"]

        if product_features.empty or len(product_target) == 0:
            continue

        decision_tree = DecisionTreeRegressor(random_state=42)
        decision_tree.fit(product_features, product_target)

        xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
        xgb.fit(product_features, product_target)

        # Take last rows for future features
        future_features = product_features.iloc[-forecast_horizon:].copy()

        # If fewer rows than forecast horizon, repeat rows to extend length
        if len(future_features) < forecast_horizon:
            future_features = pd.concat(
                [future_features] * (forecast_horizon // len(future_features) + 1),
                ignore_index=True
            )[:forecast_horizon]

        # Now assign index after length is fixed to avoid mismatch
        future_features.index = range(product_features.shape[0], product_features.shape[0] + forecast_horizon)

        dt_forecast = decision_tree.predict(future_features)
        xgb_forecast = xgb.predict(future_features)

        forecast_df = pd.DataFrame({
            "StockCode": product_id,
            "Week": range(1, forecast_horizon + 1),
            "DecisionTree_Forecast": dt_forecast,
            "XGBoost_Forecast": xgb_forecast
        })

        forecast_results = pd.concat([forecast_results, forecast_df], ignore_index=True)

    return forecast_results

def visualize_forecast(example_product, product_target, forecast_df, code_to_name=None):
    forecast_horizon = len(forecast_df)
    
    # Use product name if mapping available
    product_label = code_to_name.get(example_product, example_product) if code_to_name else example_product

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(product_target)), product_target, label="Historical Data", color="blue")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             forecast_df["DecisionTree_Forecast"], label="Decision Tree Forecast", color="green")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             forecast_df["XGBoost_Forecast"], label="XGBoost Forecast", color="red")
    plt.xlabel("Weeks")
    plt.ylabel("Quantity")
    plt.title(f"Forecast for Product: {product_label}")
    plt.legend()
    plt.tight_layout()
    plt.show()
