""" import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_top_selling_products(top_selling_products):
    fig, ax = plt.subplots(figsize=(10, 5))
    top_selling_products.plot(kind="bar", ax=ax, title="Top 10 Selling Products by Quantity")
    ax.set_ylabel("Total Quantity Sold")
    ax.set_xlabel("StockCode")
    plt.tight_layout()
    st.pyplot(fig)

def plot_top_revenue_products(top_revenue_products):
    fig, ax = plt.subplots(figsize=(10, 5))
    top_revenue_products.plot(kind="bar", color="orange", ax=ax, title="Top 10 Revenue-Generating Products")
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel("StockCode")
    plt.tight_layout()
    st.pyplot(fig)

def plot_monthly_revenue_trends(final_data):
    monthly_revenue = final_data.groupby(["Year", "Month"])["TotalRevenue"].sum().reset_index()
    monthly_revenue["Month_Year"] = monthly_revenue["Year"].astype(str) + "-" + monthly_revenue["Month"].astype(str)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(monthly_revenue["Month_Year"], monthly_revenue["TotalRevenue"], marker="o")
    ax.set_title("Monthly Revenue Trends")
    ax.set_xticks(range(len(monthly_revenue["Month_Year"])))
    ax.set_xticklabels(monthly_revenue["Month_Year"], rotation=45)
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel("Month-Year")
    plt.tight_layout()
    st.pyplot(fig)

def plot_correlation_heatmap(final_data):
    correlation_matrix = final_data[["Quantity", "UnitPrice", "TotalRevenue"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
 """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_top_selling_products(top_selling_products, code_to_name):
    # Convert StockCode index to Product Name using the mapping
    product_names = [code_to_name.get(code, str(code)) for code in top_selling_products.index]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(product_names, top_selling_products.values)
    ax.set_title("Top 10 Selling Products by Quantity")
    ax.set_ylabel("Total Quantity Sold")
    ax.set_xlabel("Product Name")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def plot_top_revenue_products(top_revenue_products, code_to_name):
    # Convert StockCode index to Product Name using the mapping
    product_names = [code_to_name.get(code, str(code)) for code in top_revenue_products.index]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(product_names, top_revenue_products.values, color="orange")
    ax.set_title("Top 10 Revenue-Generating Products")
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel("Product Name")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def plot_monthly_revenue_trends(final_data):
    monthly_revenue = final_data.groupby(["Year", "Month"])["TotalRevenue"].sum().reset_index()
    monthly_revenue["Month_Year"] = monthly_revenue["Year"].astype(str) + "-" + monthly_revenue["Month"].astype(str)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(monthly_revenue["Month_Year"], monthly_revenue["TotalRevenue"], marker="o")
    ax.set_title("Monthly Revenue Trends")
    ax.set_xticks(range(len(monthly_revenue["Month_Year"])))
    ax.set_xticklabels(monthly_revenue["Month_Year"], rotation=45)
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel("Month-Year")
    plt.tight_layout()
    st.pyplot(fig)

def plot_correlation_heatmap(final_data):
    correlation_matrix = final_data[["Quantity", "UnitPrice", "TotalRevenue"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
