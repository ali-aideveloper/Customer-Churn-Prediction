import pandas as pd
from pathlib import Path


def clean_customer_data():

    # Project directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Input and output paths
    input_file = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_file = BASE_DIR / "outputs" / "cleaned_telco_churn.csv"

    print("Loading customer dataset...")

    # Read original dataset
    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    # Remove extra spaces from TotalCharges
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()

    # Convert TotalCharges into numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace("", pd.NA),
        errors="coerce"
    )

    # Identify missing TotalCharges
    missing_mask = df["TotalCharges"].isna()

    # Check whether missing values belong to new customers
    invalid_missing = missing_mask & (df["tenure"] != 0)

    if invalid_missing.any():
        raise ValueError(
            "Missing TotalCharges found for customers with tenure above zero."
        )

    # New customers with zero tenure have zero accumulated charges
    df.loc[missing_mask & (df["tenure"] == 0), "TotalCharges"] = 0

    # Remove duplicate records if any
    df.drop_duplicates(inplace=True)

    # Verify customer IDs
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    # Save cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("\nDATA CLEANING COMPLETED!")
    print("Final shape:", df.shape)
    print("Missing values remaining:", df.isnull().sum().sum())
    print("TotalCharges datatype:", df["TotalCharges"].dtype)
    print("Cleaned file saved at:", output_file)


if __name__ == "__main__":
    clean_customer_data()
