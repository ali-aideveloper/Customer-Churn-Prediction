
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_prediction_model.joblib"

print("Loading trained churn prediction model...")

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
threshold = saved_data["threshold"]

# Example customer data
customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 427.50,
}

customer_df = pd.DataFrame([customer])

# Predict churn probability
churn_probability = model.predict_proba(customer_df)[0][1]

# Apply selected threshold
prediction = int(churn_probability >= threshold)

print("\n" + "=" * 40)
print("CUSTOMER CHURN PREDICTION")
print("=" * 40)

print(f"Churn Probability: {churn_probability:.2%}")
print(f"Selected Threshold: {threshold}")

if prediction == 1:
    print("\nPREDICTION: CUSTOMER MAY CHURN")
    print("Business Action: Consider a customer retention offer.")
else:
    print("\nPREDICTION: CUSTOMER LIKELY TO STAY")
    print("Business Action: Continue normal customer engagement.")

print("=" * 40)
