import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


def prepare_ml_features(merged_data):
    # Select relevant columns
    features = merged_data.copy()
    features = features[["CustomerID", "Quantity", "TotalRevenue", "UnitPrice", "Description", "TransactionDate"]]

    # Simulate demographic and category data
    features["CustomerAge"] = np.random.randint(18, 65, size=len(features))
    features["CustomerIncome"] = np.random.randint(20000, 100000, size=len(features))
    features["ProductCategory"] = np.random.choice(["Electronics", "Clothing", "Furniture"], size=len(features))

    # Limit unique product names
    top_descriptions = features["Description"].value_counts().nlargest(50).index
    features["Description"] = features["Description"].where(features["Description"].isin(top_descriptions), "Other")

    # Label encode categorical features
    label_encoder = LabelEncoder()
    features["Description"] = label_encoder.fit_transform(features["Description"])
    features["ProductCategory"] = label_encoder.fit_transform(features["ProductCategory"])

    # Define target and drop unused columns
    target = features["Quantity"]
    features.drop(["Quantity", "TransactionDate"], axis=1, inplace=True)

    return features, target


def train_models(X_train, y_train):
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(X_train, y_train)

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    return decision_tree, xgb


def evaluate_models(X_test, y_test, decision_tree, xgb):
    dt_predictions = decision_tree.predict(X_test)
    xgb_predictions = xgb.predict(X_test)

    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
    dt_mae = mean_absolute_error(y_test, dt_predictions)

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)

    print("Decision Tree - RMSE:", dt_rmse, "MAE:", dt_mae)
    print("XGBoost - RMSE:", xgb_rmse, "MAE:", xgb_mae)

    if xgb_rmse < dt_rmse:
        print("XGBoost performed better based on RMSE.")
    else:
        print("Decision Tree performed better based on RMSE.")

    return {
        "DecisionTree": {"RMSE": dt_rmse, "MAE": dt_mae},
        "XGBoost": {"RMSE": xgb_rmse, "MAE": xgb_mae},
    }

