import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_data():
    """
    Load and return product info, transactional data, and customer demographics.
    """
    product_info = pd.read_csv(os.path.join(data_dir, 'ProductInfo.csv'))
    transactional_data_1 = pd.read_csv(os.path.join(data_dir, 'Transactional_data_retail_01.csv'))
    transactional_data_2 = pd.read_csv(os.path.join(data_dir, 'Transactional_data_retail_02.csv'))
    customer_demographics = pd.read_csv(os.path.join(data_dir, 'CustomerDemographics.csv'))

    return product_info, transactional_data_1, transactional_data_2, customer_demographics


def merge_data(product_info, transactional_data_1, transactional_data_2, customer_demographics):
    """
    Clean, preprocess, and merge the datasets into a single DataFrame.
    """
    # Combine transactional data
    transactional_data = pd.concat([transactional_data_1, transactional_data_2], ignore_index=True)

    # Standardize column names
    transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
    customer_demographics.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

    # Drop missing and duplicate values
    product_info.dropna(inplace=True)
    product_info.drop_duplicates(inplace=True)

    transactional_data.dropna(subset=["CustomerID"], inplace=True)
    transactional_data.drop_duplicates(inplace=True)

    customer_demographics.dropna(inplace=True)
    customer_demographics.drop_duplicates(inplace=True)

    # Merge datasets
    merged_data = pd.merge(transactional_data, product_info, on="StockCode", how="inner")
    final_data = pd.merge(merged_data, customer_demographics, on="CustomerID", how="inner")

    # Add revenue column and parse dates
    final_data["TotalRevenue"] = final_data["Quantity"] * final_data["UnitPrice"]
    final_data["TransactionDate"] = pd.to_datetime(final_data["InvoiceDate"], errors="coerce", format="%d %B %Y")
    final_data.dropna(subset=["TransactionDate"], inplace=True)

    # Feature engineering
    final_data["Year"] = final_data["TransactionDate"].dt.year
    final_data["Month"] = final_data["TransactionDate"].dt.month
    final_data["Week"] = final_data["TransactionDate"].dt.isocalendar().week

    return final_data
