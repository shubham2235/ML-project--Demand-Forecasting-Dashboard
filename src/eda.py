import streamlit as st
import pandas as pd

def customer_level_analysis(df):
    st.subheader("Customer-Level Summary")
    total_customers = df["CustomerID"].nunique()
    avg_spending_per_customer = df.groupby("CustomerID")["TotalRevenue"].sum().mean()

    st.write(f"Total Customers: {total_customers}")
    st.write(f"Average Spending per Customer: ₹{avg_spending_per_customer:.2f}")

def item_level_analysis(df):
    st.subheader("Item-Level Summary")
    total_items = df["StockCode"].nunique()
    st.write(f"Total Unique Items: {total_items}")

    top_selling_products = df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(10)
    top_revenue_products = df.groupby("StockCode")["TotalRevenue"].sum().sort_values(ascending=False).head(10)

    st.write("Top 10 Selling Products by Quantity")
    st.dataframe(top_selling_products)

    st.write("Top 10 Revenue-Generating Products")
    st.dataframe(top_revenue_products)

    return top_selling_products, top_revenue_products

def transaction_level_analysis(df):
    st.subheader("Transaction-Level Summary")
    total_transactions = df["Invoice"].nunique()
    avg_revenue_per_transaction = df.groupby("Invoice")["TotalRevenue"].sum().mean()

    st.write(f"Total Transactions: {total_transactions}")
    st.write(f"Average Revenue per Transaction: ₹{avg_revenue_per_transaction:.2f}")
