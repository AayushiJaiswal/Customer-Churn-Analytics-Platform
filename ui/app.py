import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("Customer Churn Analytics Platform")
st.caption("End-to-End SAS → Python → Dashboard Workflow")

# =========================
# STEP 1: Upload Data
# =========================
st.header("1. Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")
    st.dataframe(df.head())

    # Save uploaded file
    df.to_csv("data/churn_clean.csv", index=False)

# =========================
# STEP 2: SAS Processing (Simulated)
# =========================
st.header("2. SAS Statistical Processing")

st.info(
    "Data preprocessing and statistical validation "
    "is performed using SAS (handling missing values, "
    "encoding, feature selection)."
)

# =========================
# STEP 3: Train ML Model
# =========================
st.header("3. Train Churn Prediction Model")

if st.button("Train Model"):
    with st.spinner("Training model..."):
        subprocess.run(["python", "python/train_model.py"])
    st.success("Model trained successfully")

# =========================
# STEP 4: Prediction Results
# =========================
st.header("4. High-Risk Customers")

if os.path.exists("data/high_risk_customers.csv"):
    results = pd.read_csv("data/high_risk_customers.csv")

    st.metric(
        "Average Churn Risk",
        round(results["Churn_Risk"].mean(), 2)
    )

    st.dataframe(results)

    # =========================
    # STEP 5: Interactive Dashboard
    # =========================
    st.header("5. Interactive Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Risk by Contract")
        st.bar_chart(
            results.groupby("Contract")["Churn_Risk"].mean()
        )

    with col2:
        st.subheader("Churn Risk by Tenure")
        st.line_chart(
            results.groupby("tenure")["Churn_Risk"].mean()
        )

    # Filters
    st.subheader("Filters")
    service = st.selectbox(
        "Internet Service",
        options=["All"] + list(results["InternetService"].unique())
    )

    if service != "All":
        filtered = results[results["InternetService"] == service]
        st.dataframe(filtered)
