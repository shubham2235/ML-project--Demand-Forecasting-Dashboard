import pandas as pd

def preprocess_transactional_data(transactional_data: pd.DataFrame) -> pd.DataFrame:
    transactional_data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
    transactional_data.dropna(subset=["CustomerID"], inplace=True)
    transactional_data.drop_duplicates(inplace=True)
    return transactional_data

def preprocess_product_info(product_info: pd.DataFrame) -> pd.DataFrame:
    product_info.dropna(inplace=True)
    product_info.drop_duplicates(inplace=True)
    return product_info

def preprocess_customer_data(customer_demographics: pd.DataFrame) -> pd.DataFrame:
    customer_demographics.rename(columns={"Customer ID": "CustomerID"}, inplace=True)
    customer_demographics.dropna(inplace=True)
    customer_demographics.drop_duplicates(inplace=True)
    return customer_demographics

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["TransactionDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", format="%d %B %Y")
    df.dropna(subset=["TransactionDate"], inplace=True)
    df["Year"] = df["TransactionDate"].dt.year
    df["Month"] = df["TransactionDate"].dt.month
    df["Week"] = df["TransactionDate"].dt.isocalendar().week
    return df
