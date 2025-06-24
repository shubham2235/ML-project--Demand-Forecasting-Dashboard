import pandas as pd

# Load the datasets
product_info = pd.read_csv("ProductInfo.csv")
transactional_data_1 = pd.read_csv("Transactional_data_retail_01.csv")
transactional_data_2 = pd.read_csv("Transactional_data_retail_02.csv")
customer_demographics = pd.read_csv("CustomerDemographics.csv")

# Combine the two transactional datasets
transactional_data = pd.concat([transactional_data_1, transactional_data_2], ignore_index=True)

# Standardize column names to ensure consistency
transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
customer_demographics.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

# Inspect data
print("Product Info:")
print(product_info.head())

print("\nCombined Transactional Data:")
print(transactional_data.head())

print("\nCustomer Demographics:")
print(customer_demographics.head())

# Handle missing values
product_info.dropna(inplace=True)  # Drop rows with missing descriptions
transactional_data.dropna(subset=["CustomerID"], inplace=True)  # Drop rows where CustomerID is missing
customer_demographics.dropna(inplace=True)

# Drop duplicates
product_info.drop_duplicates(inplace=True)
transactional_data.drop_duplicates(inplace=True)
customer_demographics.drop_duplicates(inplace=True)

# Merge transactional data with product info
merged_data = pd.merge(transactional_data, product_info, on="StockCode", how="inner")

# Merge with customer demographics
final_data = pd.merge(merged_data, customer_demographics, on="CustomerID", how="inner")

# Add total revenue column
final_data["TotalRevenue"] = final_data["Quantity"] * final_data["UnitPrice"]

# Convert transaction date to datetime
final_data["TransactionDate"] = pd.to_datetime(final_data["InvoiceDate"], errors="coerce", format="%d %B %Y")

# Drop rows with invalid dates
final_data.dropna(subset=["TransactionDate"], inplace=True)

# Extract year, month, week
final_data["Year"] = final_data["TransactionDate"].dt.year
final_data["Month"] = final_data["TransactionDate"].dt.month
final_data["Week"] = final_data["TransactionDate"].dt.isocalendar().week

# Inspect the final dataset
print("\nFinal Merged Dataset with Feature Engineering:")
print(final_data.head())



# STEP 2 EDA 

import matplotlib.pyplot as plt
import seaborn as sns

# Customer-Level Analysis
print("\nCustomer-Level Summary:")
total_customers = final_data["CustomerID"].nunique()
avg_spending_per_customer = final_data.groupby("CustomerID")["TotalRevenue"].sum().mean()
print(f"Total Customers: {total_customers}")
print(f"Average Spending per Customer: {avg_spending_per_customer:.2f}")

# Item-Level Analysis
print("\nItem-Level Summary:")
total_items = final_data["StockCode"].nunique()
top_selling_products = final_data.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(10)
top_revenue_products = final_data.groupby("StockCode")["TotalRevenue"].sum().sort_values(ascending=False).head(10)
print(f"Total Unique Items: {total_items}")
print("\nTop-Selling Products by Quantity:")
print(top_selling_products)
print("\nTop Revenue-Generating Products:")
print(top_revenue_products)

# Transaction-Level Analysis
print("\nTransaction-Level Summary:")
total_transactions = final_data["Invoice"].nunique()
avg_revenue_per_transaction = final_data.groupby("Invoice")["TotalRevenue"].sum().mean()
print(f"Total Transactions: {total_transactions}")
print(f"Average Revenue per Transaction: {avg_revenue_per_transaction:.2f}")

# Visualizations
# Top 10 Selling Products by Quantity
top_selling_products.plot(kind="bar", figsize=(10, 5), title="Top 10 Selling Products by Quantity")
plt.ylabel("Total Quantity Sold")
plt.xlabel("StockCode")
plt.show()

# Top 10 Revenue-Generating Products
top_revenue_products.plot(kind="bar", color="orange", figsize=(10, 5), title="Top 10 Revenue-Generating Products")
plt.ylabel("Total Revenue")
plt.xlabel("StockCode")
plt.show()

# Monthly Revenue Trends
monthly_revenue = final_data.groupby(["Year", "Month"])["TotalRevenue"].sum().reset_index()
monthly_revenue["Month_Year"] = monthly_revenue["Year"].astype(str) + "-" + monthly_revenue["Month"].astype(str)
plt.figure(figsize=(15, 5))
plt.plot(monthly_revenue["Month_Year"], monthly_revenue["TotalRevenue"], marker="o")
plt.title("Monthly Revenue Trends")
plt.xticks(rotation=45)
plt.ylabel("Total Revenue")
plt.xlabel("Month-Year")
plt.show()

# Correlation Heatmap
correlation_matrix = final_data[["Quantity", "UnitPrice", "TotalRevenue"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


####

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
product_info = pd.read_csv("ProductInfo.csv")
transactional_data_1 = pd.read_csv("Transactional_data_retail_01.csv")
transactional_data_2 = pd.read_csv("Transactional_data_retail_02.csv")

# Combine transactional datasets
transactional_data = pd.concat([transactional_data_1, transactional_data_2], ignore_index=True)

# Standardize column names
transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)

# Merge transactional data with product info
merged_data = pd.merge(transactional_data, product_info, on="StockCode", how="inner")

# Add a revenue column
merged_data["Revenue"] = merged_data["Quantity"] * merged_data["UnitPrice"]

# Step 1: Top 10 Best-Selling Products by Quantity
top_selling_products = (
    merged_data.groupby(["StockCode", "Description"])["Quantity"]
    .sum()
    .reset_index()
    .sort_values(by="Quantity", ascending=False)
    .head(10)
)

# Visualization: Top 10 Best-Selling Products
plt.figure(figsize=(12, 6))
plt.bar(top_selling_products["Description"], top_selling_products["Quantity"], color="skyblue")
plt.title("Top 10 Best-Selling Products by Quantity")
plt.ylabel("Total Quantity Sold")
plt.xlabel("Product Name")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Step 2: Top 10 Revenue-Generating Products
top_revenue_products = (
    merged_data.groupby(["StockCode", "Description"])["Revenue"]
    .sum()
    .reset_index()
    .sort_values(by="Revenue", ascending=False)
    .head(10)
)

# Visualization: Top 10 Revenue-Generating Products
plt.figure(figsize=(12, 6))
plt.bar(top_revenue_products["Description"], top_revenue_products["Revenue"], color="orange")
plt.title("Top 10 Revenue-Generating Products")
plt.ylabel("Total Revenue")
plt.xlabel("Product Name")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Optional: Save the results to CSV files for further analysis
top_selling_products.to_csv("top_10_best_sellers_with_names.csv", index=False)
top_revenue_products.to_csv("top_10_revenue_products_with_names.csv", index=False)

# Inspect results
print("Top 10 Best-Selling Products by Quantity:")
print(top_selling_products)

print("\nTop 10 Revenue-Generating Products:")
print(top_revenue_products)


#### step 4 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load and merge datasets (adjust file paths as needed)
product_info = pd.read_csv("ProductInfo.csv")
transactional_data_1 = pd.read_csv("Transactional_data_retail_01.csv")
transactional_data_2 = pd.read_csv("Transactional_data_retail_02.csv")
transactional_data = pd.concat([transactional_data_1, transactional_data_2], ignore_index=True)

# Standardize column names and merge with product info
transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
merged_data = pd.merge(transactional_data, product_info, on="StockCode", how="inner")
merged_data["Revenue"] = merged_data["Quantity"] * merged_data["UnitPrice"]

# Ensure TransactionDate is in datetime format
merged_data["TransactionDate"] = pd.to_datetime(merged_data["InvoiceDate"], errors="coerce", format="%d %B %Y")

# Step 4a: Data Preparation
# Filter the top 10 products by total quantity sold
top_products = (
    merged_data.groupby(["StockCode", "Description"])["Quantity"]
    .sum()
    .reset_index()
    .sort_values(by="Quantity", ascending=False)
    .head(10)
)

# Aggregate weekly data for each top product
merged_data["Week"] = merged_data["TransactionDate"].dt.to_period("W").dt.start_time
weekly_data = merged_data.groupby(["Week", "StockCode", "Description"])["Quantity"].sum().reset_index()

# Plot weekly sales for each top product
for product in top_products["StockCode"]:
    product_data = weekly_data[weekly_data["StockCode"] == product]
    plt.figure(figsize=(12, 6))
    plt.plot(product_data["Week"], product_data["Quantity"], marker="o")
    plt.title(f"Weekly Sales for Product: {product}")
    plt.xlabel("Week")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 4b: ACF and PACF Analysis
# Select a single top product for demonstration
example_product = top_products.iloc[0]["StockCode"]
example_product_data = weekly_data[weekly_data["StockCode"] == example_product]

# Decompose the time series with dynamic period adjustment
if len(example_product_data["Quantity"]) >= 2:
    dynamic_period = min(52, len(example_product_data["Quantity"]) // 2)  # Use the smaller of 52 or half the data length
    decomposed = seasonal_decompose(example_product_data["Quantity"], model="additive", period=dynamic_period)
    decomposed.plot()
    plt.show()
else:
    print(f"Insufficient data for seasonal decomposition of product {example_product}.")

# Plot ACF and PACF
plot_acf(example_product_data["Quantity"], lags=20)
plt.title(f"ACF for Product: {example_product}")
plt.show()

plot_pacf(example_product_data["Quantity"], lags=20)
plt.title(f"PACF for Product: {example_product}")
plt.show()

# Step 4c: Model Selection and Forecasting
# Train an ARIMA model for the selected product
train_data = example_product_data["Quantity"][:-15]
test_data = example_product_data["Quantity"][-15:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5, 1, 0))  # Adjust the order based on ACF and PACF
arima_result = arima_model.fit()

# Forecast for the next 15 weeks
forecast = arima_result.forecast(steps=15)
forecast.index = test_data.index

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label="Training Data")
plt.plot(test_data.index, test_data, label="Actual Data", color="blue")
plt.plot(forecast.index, forecast, label="Forecast", color="red")
plt.title(f"Forecasting for Product: {example_product}")
plt.xlabel("Week")
plt.ylabel("Quantity Sold")
plt.legend()
plt.tight_layout()
plt.show()

# Step 4d: Forecast Next 15 Weeks for All Top Products
predictions = {}
for product in top_products["StockCode"]:
    product_data = weekly_data[weekly_data["StockCode"] == product]["Quantity"]
    if len(product_data) > 15:
        arima_model = ARIMA(product_data[:-15], order=(5, 1, 0))
        arima_result = arima_model.fit()
        predictions[product] = arima_result.forecast(steps=15)

# Save the forecasts for all products
forecast_df = pd.DataFrame(predictions)
forecast_df.to_csv("forecasted_quantities_top_products.csv", index=False)
print("Forecast saved to 'forecasted_quantities_top_products.csv'")




####step 5 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Ensure merged_data and top_products are already defined from the previous steps
# Include customer demographics and product information for feature selection

# Step 5a: Feature Selection
features = merged_data.copy()
features = features[["CustomerID", "Quantity", "Revenue", "UnitPrice", "Description", "TransactionDate"]]

# Generate example customer demographics and product categories
features["CustomerAge"] = np.random.randint(18, 65, size=len(features))
features["CustomerIncome"] = np.random.randint(20000, 100000, size=len(features))
features["ProductCategory"] = np.random.choice(["Electronics", "Clothing", "Furniture"], size=len(features))

# Limit the number of unique categories for one-hot encoding
# Group less frequent categories into "Other"
top_descriptions = features["Description"].value_counts().nlargest(50).index  # Keep top 50
features["Description"] = features["Description"].where(features["Description"].isin(top_descriptions), "Other")

# Label encode categorical features with many categories
label_encoder = LabelEncoder()
features["Description"] = label_encoder.fit_transform(features["Description"])
features["ProductCategory"] = label_encoder.fit_transform(features["ProductCategory"])

# Define target variable
target = features["Quantity"]

# Drop unnecessary columns
features.drop(["Quantity", "TransactionDate"], axis=1, inplace=True)

# Step 5b: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 5c: Train Machine Learning Models
# Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
dt_predictions = decision_tree.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
xgb.fit(X_train, y_train)
xgb_predictions = xgb.predict(X_test)

# Step 5d: Evaluate Models
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_mae = mean_absolute_error(y_test, dt_predictions)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

# Step 5e: Compare Model Performance
print("Decision Tree - RMSE:", dt_rmse, "MAE:", dt_mae)
print("XGBoost - RMSE:", xgb_rmse, "MAE:", xgb_mae)

if xgb_rmse < dt_rmse:
    print("XGBoost performed better based on RMSE.")
else:
    print("Decision Tree performed better based on RMSE.")



##### step 6 
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure merged_data and top_products are already defined from the previous steps
# Include customer demographics and product information for feature selection

# Step 6a: Time-Based Cross-Validation
# Initialize TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Store results for each fold
dt_rmse_scores = []
xgb_rmse_scores = []

dt_mae_scores = []
xgb_mae_scores = []

for train_index, test_index in tscv.split(features):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    # Train Decision Tree
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(X_train, y_train)
    dt_predictions = decision_tree.predict(X_test)

    # Evaluate Decision Tree
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
    dt_mae = mean_absolute_error(y_test, dt_predictions)
    dt_rmse_scores.append(dt_rmse)
    dt_mae_scores.append(dt_mae)

    # Train XGBoost
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)

    # Evaluate XGBoost
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse_scores.append(xgb_rmse)
    xgb_mae_scores.append(xgb_mae)

# Step 6b: Calculate Average Performance Across Folds
avg_dt_rmse = np.mean(dt_rmse_scores)
avg_dt_mae = np.mean(dt_mae_scores)

avg_xgb_rmse = np.mean(xgb_rmse_scores)
avg_xgb_mae = np.mean(xgb_mae_scores)

# Step 6c: Compare Model Performance
print("Time-Based Cross-Validation Results:")
print("Decision Tree - Avg RMSE:", avg_dt_rmse, "Avg MAE:", avg_dt_mae)
print("XGBoost - Avg RMSE:", avg_xgb_rmse, "Avg MAE:", avg_xgb_mae)

if avg_xgb_rmse < avg_dt_rmse:
    print("XGBoost performed better based on average RMSE.")
else:
    print("Decision Tree performed better based on average RMSE.")


#### step 7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assume merged_data and top_products are already defined

# Debug: Print the content of top_products
print("Top Products List:", top_products)

# Debug: Check the columns of merged_data
print("Merged Data Columns:", merged_data.columns)

# Step 7a: Forecast for Top 10 Products
forecast_horizon = 15  # Forecast next 15 weeks

# Initialize a DataFrame to store forecasts
forecast_results = pd.DataFrame()

# Extract only the unique StockCode values from the top_products DataFrame
top_products = top_products["StockCode"].unique().tolist()

# Debug: Print the corrected top_products list
print("Corrected Top Products List:", top_products)


for product_id in top_products:  # Loop through the top 10 products
    # Debug: Log the current product being processed
    print(f"Processing Product ID: {product_id}")

    # Filter data for the current product
    product_data = merged_data[merged_data["StockCode"] == product_id]

    # Debug: Check if the filtered product_data has rows
    print(f"Product Data for {product_id}: {len(product_data)} rows")

    # Skip if no data is available for the product
    if product_data.empty:
        print(f"No data available for Product {product_id}. Skipping...")
        continue

    # Check and generate missing columns if required
    if "CustomerAge" not in product_data.columns:
        product_data["CustomerAge"] = np.random.randint(18, 65, size=len(product_data))
    if "CustomerIncome" not in product_data.columns:
        product_data["CustomerIncome"] = np.random.randint(20000, 100000, size=len(product_data))

    # Prepare features and target
    product_features = product_data[["CustomerID", "Revenue", "UnitPrice", "CustomerAge", "CustomerIncome"]]
    product_target = product_data["Quantity"]

    # Skip if there are no rows left after filtering
    if product_features.empty or len(product_target) == 0:
        print(f"Insufficient data for Product {product_id}. Skipping...")
        continue

    # Train models on the entire dataset
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(product_features, product_target)

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=100, learning_rate=0.1)
    xgb.fit(product_features, product_target)

    # Create future feature inputs (simulate features for the next 15 weeks)
    future_features = product_features.iloc[-forecast_horizon:].copy()
    future_features.index = range(product_features.shape[0], product_features.shape[0] + forecast_horizon)

    # Handle edge case: Not enough historical rows to simulate future features
    if len(future_features) < forecast_horizon:
        future_features = pd.concat(
            [future_features] * (forecast_horizon // len(future_features) + 1), ignore_index=True
        )[:forecast_horizon]

    # Generate forecasts
    dt_forecast = decision_tree.predict(future_features)
    xgb_forecast = xgb.predict(future_features)

    # Combine results
    forecast_df = pd.DataFrame({
        "StockCode": product_id,
        "Week": range(1, forecast_horizon + 1),
        "DecisionTree_Forecast": dt_forecast,
        "XGBoost_Forecast": xgb_forecast
    })

    # Append to the results DataFrame
    forecast_results = pd.concat([forecast_results, forecast_df], ignore_index=True)

# Step 7b: Save Results to CSV
if not forecast_results.empty:
    forecast_results.to_csv("forecasted_quantities_next_15_weeks.csv", index=False)
    print("Forecasting completed. Results saved to 'forecasted_quantities_next_15_weeks.csv'.")

# Step 7c: Visualize Forecasts for One Product (Example: First Product)
if not forecast_results.empty:
    example_product = top_products[0]
    example_forecast = forecast_results[forecast_results["StockCode"] == example_product]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(product_target)), product_target, label="Historical Data", color="blue")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             example_forecast["DecisionTree_Forecast"], label="Decision Tree Forecast", color="green")
    plt.plot(range(len(product_target), len(product_target) + forecast_horizon),
             example_forecast["XGBoost_Forecast"], label="XGBoost Forecast", color="red")
    plt.xlabel("Weeks")
    plt.ylabel("Quantity")
    plt.title(f"Forecast for Product {example_product}")
    plt.legend()
    plt.show()
else:
    print("No forecast results to visualize.")



#### step 8 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
product_info = pd.read_csv("ProductInfo.csv")
transactional_data_1 = pd.read_csv("Transactional_data_retail_01.csv")
transactional_data_2 = pd.read_csv("Transactional_data_retail_02.csv")
forecast_data = pd.read_csv("forecasted_quantities_next_15_weeks.csv")  # Forecast data from Step 7
errors_data = pd.read_csv("forecast_insights.csv")  # Insights from Step 5

# Combine transactional data
transactional_data = pd.concat([transactional_data_1, transactional_data_2], ignore_index=True)
transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
transactional_data["TransactionDate"] = pd.to_datetime(transactional_data["InvoiceDate"], errors="coerce")
transactional_data["Week"] = transactional_data["TransactionDate"].dt.to_period("W").dt.start_time

# Merge transactional data with product info
merged_data = pd.merge(transactional_data, product_info, on="StockCode", how="inner")

# Aggregate weekly data
weekly_data = merged_data.groupby(["Week", "StockCode", "Description"])["Quantity"].sum().reset_index()

# Streamlit App
st.title("Sales Forecasting App")
st.sidebar.header("Select Stock Code")

# Input field for stock code
stock_code = st.sidebar.text_input("Enter Stock Code", value="85123A")  # Default stock code

# Display Historical Data and Forecast
st.subheader(f"Demand Analysis for Stock Code: {stock_code}")

if stock_code in weekly_data["StockCode"].unique():
    # Filter historical data for the selected stock code
    historical = weekly_data[weekly_data["StockCode"] == stock_code]
    
    # Filter forecast data for the selected stock code
    forecast = forecast_data[forecast_data["StockCode"] == stock_code]
    
    # Combine historical and forecast data
    combined = historical.merge(forecast, on="Week", how="outer").fillna(0)
    
    # Plot historical and forecasted demand
    st.line_chart(combined.set_index("Week")[["Quantity", "DecisionTree_Forecast", "XGBoost_Forecast"]])
    
    # Display the data
    st.write("Historical and Forecasted Demand:")
    st.dataframe(combined)
else:
    st.error("Stock code not found in the historical data!")

# Model Performance
st.subheader("Model Performance")

if stock_code in errors_data["Product"].unique():
    # Get error data for the selected stock code
    product_errors = errors_data[errors_data["Product"] == stock_code]
    
    # Display RMSE and MAE
    st.write(f"RMSE: {product_errors['RMSE'].values[0]:.2f}")
    st.write(f"MAE: {product_errors['MAE'].values[0]:.2f}")
    
    # Plot error histogram if available
    if "ErrorDistribution" in product_errors.columns:
        fig, ax = plt.subplots()
        ax.hist(eval(product_errors["ErrorDistribution"].values[0]), bins=20, color="skyblue", edgecolor="black")
        ax.set_title("Error Distribution (Training & Testing)")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.write("Error distribution data not available for this product.")
else:
    st.error("Model performance data not available for the selected stock code!")

