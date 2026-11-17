
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_prediction_model.joblib"
DATA_PATH = BASE_DIR / "outputs" / "cleaned_telco_churn.csv"

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Load data and model
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or cleaned dataset not found.")
    st.stop()

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
threshold = saved_data["threshold"]

df = pd.read_csv(DATA_PATH)

# Header
st.title("📊 Customer Churn Analytics Dashboard")
st.caption("Telecom Customer Retention | Machine Learning Project")

# KPI calculations
total_customers = len(df)
churned = (df["Churn"] == "Yes").sum()
retained = (df["Churn"] == "No").sum()
churn_rate = churned / total_customers * 100

st.subheader("Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned:,}")
c3.metric("Retained Customers", f"{retained:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# Charts
st.subheader("Customer Churn Insights")

left, right = st.columns(2)

with left:
    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Customer Status")
        .reset_index(name="Customers")
    )

    fig = px.pie(
        churn_counts,
        names="Customer Status",
        values="Customers",
        title="Churn vs Retained Customers",
        hole=0.45
    )

    st.plotly_chart(fig, width="stretch")

with right:
    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    contract_data["Churn Percentage"] = (
        contract_data["Customers"]
        / contract_data.groupby("Contract")["Customers"].transform("sum")
        * 100
    )

    fig = px.bar(
        contract_data,
        x="Contract",
        y="Churn Percentage",
        color="Churn",
        barmode="group",
        title="Churn Percentage by Contract",
        labels={"Churn Percentage": "Percentage (%)"}
    )

    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)

with left:
    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=24,
        title="Customer Tenure Distribution",
        labels={"tenure": "Tenure (Months)"}
    )

    st.plotly_chart(fig, width="stretch")

with right:
    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn",
        labels={"MonthlyCharges": "Monthly Charges"}
    )

    st.plotly_chart(fig, width="stretch")

st.divider()
# =====================================
# HIGH-RISK CUSTOMER ANALYSIS
# =====================================

st.divider()
st.header("🚨 High-Risk Customer Analysis")

# Calculate churn probability
risk_features = df.drop(
    columns=["customerID", "Churn"]
)

risk_probabilities = model.predict_proba(
    risk_features
)[:, 1]

# Create risk dataframe
risk_df = df[
    [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "Churn"
    ]
].copy()

risk_df["Risk Percentage"] = (
    risk_probabilities * 100
).round(2)

# Filter high-risk customers
high_risk_df = risk_df[
    risk_probabilities >= threshold
].copy()

high_risk_df = high_risk_df.sort_values(
    by="Risk Percentage",
    ascending=False
)

# Display metrics
col1, col2 = st.columns(2)

col1.metric(
    "High-Risk Customers",
    f"{len(high_risk_df):,}"
)

col2.metric(
    "Risk Threshold",
    f"{threshold:.0%}"
)

# Display table
st.subheader("High-Risk Customer List")

if not high_risk_df.empty:

    st.dataframe(
        high_risk_df,
        width="stretch",
        hide_index=True
    )

    csv_data = high_risk_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download High-Risk Customer Report",
        data=csv_data,
        file_name="high_risk_customers.csv",
        mime="text/csv",
        width="stretch"
    )

else:
    st.info("No customers found above the selected risk threshold.")

st.caption(
    "Note: These are exploratory risk estimates on historical "
    "records, not validated predictions for future customers."
)

# Prediction form
st.subheader("🤖 Predict Individual Customer Churn")
st.write("Enter customer details to estimate their churn risk.")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", 0, 72, 5)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=85.50
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=427.50
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    customer_df = pd.DataFrame([customer])
    probability = model.predict_proba(customer_df)[0][1]
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Customer Prediction Result")

    result1, result2 = st.columns(2)

    result1.metric("Churn Probability", f"{probability:.2%}")
    result2.metric("Decision Threshold", f"{threshold:.0%}")

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Customer may churn")
        st.write(
            "Suggested action: Contact the customer, investigate "
            "their concerns and consider a suitable retention offer."
        )
    else:
        st.success("✅ Customer likely to stay")
        st.write(
            "Suggested action: Continue normal engagement "
            "and monitor customer satisfaction."
        )

    st.info(
        "Prediction is an estimate based on historical data, "
        "not a guarantee of future customer behavior."
    )

st.divider()
st.caption("Customer Churn Analytics | Machine Learning Project")
