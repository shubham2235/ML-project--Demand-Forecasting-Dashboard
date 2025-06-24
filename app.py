import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data, merge_data
from src.preprocessing import (
    preprocess_transactional_data,
    preprocess_product_info,
    preprocess_customer_data,
    feature_engineering,
)
from src.eda import (
    customer_level_analysis,
    item_level_analysis,
    transaction_level_analysis,
)
from src.visualization import (
    plot_top_selling_products,
    plot_top_revenue_products,
    plot_monthly_revenue_trends,
    plot_correlation_heatmap,
)
from src.ml_modeling import (
    prepare_ml_features,
    train_models,
    evaluate_models,
)
from src.forecasting import (
    forecast_top_products,
    plot_forecast,
)
from src.final_forecast import (
    generate_final_forecast,
    visualize_forecast,
)
from src.evaluation import evaluate_model

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization Dashboard",
    layout="wide",
    page_icon="📦"
)

# Project overview (brief)
st.title("📦 Demand Forecasting & Inventory Optimization Dashboard")
st.markdown(
    """
    This dashboard analyzes product demand using **transactional, product, and customer data**.
    It leverages **statistical (ARIMA)** and **machine learning models (Decision Tree, XGBoost)** 
    to provide accurate forecasts and actionable insights for inventory optimization.
    Use the sidebar to navigate through sections and filter products.
    """
)
st.markdown("---")

# Load & preprocess data
@st.cache_data(show_spinner=False)
def load_all_data():
    product_info, t1, t2, customer_demographics = load_data()
    t1_proc = preprocess_transactional_data(t1)
    t2_proc = preprocess_transactional_data(t2)
    prod_info_proc = preprocess_product_info(product_info)
    cust_demo_proc = preprocess_customer_data(customer_demographics)
    merged = merge_data(prod_info_proc, t1_proc, t2_proc, cust_demo_proc)
    final = feature_engineering(merged)
    return final

final_data = load_all_data()

# Create mapping for StockCode ↔ ProductName
code_to_name = final_data.dropna(subset=["Description"]).drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()
name_to_code = {v: k for k, v in code_to_name.items()}

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2630/2630434.png", width=100)
    st.title("Demand Forecasting")
    st.markdown("**Version:** 1.0.0")
    st.markdown("---")

    st.header("Navigation")
    section = st.radio(
        "Go to",
        [
            "Exploratory Data Analysis",
            "Machine Learning Forecasting",
            "ARIMA Forecasting",
            "Final Product-Level Forecasts",
            "Raw Data View",
        ],
        index=0,
    )

    st.markdown("---")
    st.header("Filters")

    all_product_names = list(name_to_code.keys())
    selected_product_names = st.multiselect(
        "Select Product(s)",
        all_product_names,
        default=all_product_names[:5],
        help="Select products to analyze or forecast. You can select multiple.",
    )

    # Convert names back to codes
    selected_products = [name_to_code[name] for name in selected_product_names if name in name_to_code]

    forecast_horizon = st.slider(
        "Forecast Horizon (weeks)",
        min_value=5,
        max_value=30,
        value=15,
        help="Select how many weeks ahead to forecast the demand.",
    )

# Main content
if section == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹{final_data['TotalRevenue'].sum():,.0f}")
    col2.metric("Unique Customers", f"{final_data['CustomerID'].nunique():,}")
    col3.metric("Total Transactions", f"{final_data['Invoice'].nunique():,}")

    st.markdown("---")

    with st.expander("Customer Level Analysis"):
        customer_level_analysis(final_data)
        st.markdown("""
        **Summary:**  
        - Total customers represent distinct buyers.  
        - Average spending per customer shows typical purchase size.  
        These insights help identify customer value and segmentation opportunities.
        """)

    with st.expander("Item Level Analysis"):
        top_selling, top_revenue = item_level_analysis(final_data)
        st.markdown("""
        **Summary:**  
        - Top-selling products drive volume sales.  
        - Top revenue products contribute most to total income, which may differ from volume leaders.  
        These insights guide inventory prioritization and marketing focus.
        """)

    with st.expander("Transaction Level Analysis"):
        transaction_level_analysis(final_data)
        st.markdown("""
        **Summary:**  
        - Total transactions represent sales events.  
        - Average revenue per transaction indicates typical transaction value.  
        This helps evaluate sales effectiveness and pricing strategies.
        """)

    st.subheader("Visualizations")

    plot_top_selling_products(top_selling, code_to_name)
    st.markdown("""
    **What does this show?**  
    These products contribute the most to sales volume.  
    **Actionable Insight:** Prioritize stocking and marketing for these items.
    """)

    plot_top_revenue_products(top_revenue, code_to_name)
    st.markdown("""
    **What does this show?**  
    These products generate the highest revenue, not always matching volume leaders.  
    **Actionable Insight:** Focus on maximizing profitability from these products.
    """)

    plot_monthly_revenue_trends(final_data)
    st.markdown("""
    **What does this show?**  
    Monthly revenue trends reveal seasonality and growth patterns.  
    **Actionable Insight:** Use seasonal spikes to plan inventory and promotions.
    """)

    plot_correlation_heatmap(final_data)
    st.markdown("""
    **What does this show?**  
    Strong positive correlation between quantity sold and revenue.  
    **Actionable Insight:** Quantity sold drives revenue more than unit price.
    """)

elif section == "Machine Learning Forecasting":
    st.header("🤖 ML-Based Forecasting")
    with st.spinner("Preparing data and training models..."):
        features, target = prepare_ml_features(final_data)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        decision_tree, xgb = train_models(X_train, y_train)
        metrics = evaluate_models(X_test, y_test, decision_tree, xgb)

    st.subheader("Model Performance")
    st.success("Models trained successfully!")
    st.table(metrics)

    st.markdown("""
    **Interpretation:**  
    These models use historical data and customer/product features to predict future demand.  
    Decision Tree and XGBoost offer complementary strengths for robust forecasting.
    """)

elif section == "ARIMA Forecasting":
    st.header("📈 ARIMA Forecasting")

    weekly_data = final_data.groupby(["Week", "StockCode"])["Quantity"].sum().reset_index()
    forecasts = forecast_top_products(weekly_data, selected_products, steps=forecast_horizon)

    if not selected_products:
        st.info("Please select at least one product to view forecasts.")
    else:
        for product in selected_products:
            product_name = code_to_name.get(product, product)
            series = weekly_data[weekly_data["StockCode"] == product].set_index("Week")["Quantity"]
            if len(series) > forecast_horizon:
                plot_forecast(series, forecasts.get(product, []), product, code_to_name)

        if forecasts:
            # Reshape forecast dict into long-form DataFrame
            forecast_rows = []
            for stock_code, values in forecasts.items():
                for week, forecast in enumerate(values, 1):
                    forecast_rows.append({
                        "StockCode": stock_code,
                        "ProductName": code_to_name.get(stock_code, stock_code),
                        "Week": week,
                        "ForecastedQuantity": forecast
                    })

            df_forecast = pd.DataFrame(forecast_rows)

            st.subheader("Forecast Data")
            st.dataframe(df_forecast)

            csv = df_forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Forecast CSV",
                csv,
                "arima_forecasts.csv",
                "text/csv",
                help="Download the ARIMA forecast results as a CSV file.",
            )

    st.markdown("""
    **Conclusion:**  
    ARIMA forecasts predict weekly demand based on historical sales trends.  
    **Actionable Insight:** Use these forecasts to manage inventory and reduce stockouts or overstocks.
    """)

elif section == "Final Product-Level Forecasts":
    st.header("📦 Final Forecasts (DecisionTree & XGBoost)")
    forecast_df = generate_final_forecast(
        final_data, final_data[final_data["StockCode"].isin(selected_products)], forecast_horizon
    )
    forecast_df["ProductName"] = forecast_df["StockCode"].map(code_to_name)
    st.dataframe(forecast_df)

    for product in selected_products:
        product_name = code_to_name.get(product, product)
        product_data = final_data[final_data["StockCode"] == product]
        product_target = product_data["Quantity"]
        product_forecast = forecast_df[forecast_df["StockCode"] == product]
        visualize_forecast(product_name, product_target, product_forecast,code_to_name)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Final Forecast CSV",
        csv,
        "final_forecasts.csv",
        "text/csv",
        help="Download the combined ML forecast results.",
    )

    st.markdown("""
    **Conclusion:**  
    ML-based forecasts integrate multiple data features to estimate demand.  
    **Actionable Insight:** Use these predictions to make informed stocking and sales decisions.
    """)

elif section == "Raw Data View":
    st.header("🧾 Raw Merged Dataset")
    st.dataframe(final_data.head(100))
    raw_csv = final_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Raw Data",
        raw_csv,
        "merged_data.csv",
        "text/csv",
        help="Download the raw merged dataset as CSV.",
    )

# Footer
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by Shubham Chitlangya | Data Science & ML Intern</center>",
    unsafe_allow_html=True,
)
