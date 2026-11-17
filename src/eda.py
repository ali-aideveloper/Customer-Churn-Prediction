import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "outputs" / "cleaned_telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)

print("CUSTOMER CHURN EDA")
print("=" * 40)

# 1. Overall churn analysis
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 2. Churn by contract
print("\nChurn by Contract:")
print(pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
).mul(100).round(2))

# 3. Average tenure
print("\nAverage Tenure by Churn:")
print(df.groupby("Churn")["tenure"].mean().round(2))

# 4. Average monthly charges
print("\nAverage Monthly Charges by Churn:")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

# Chart 1: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png")
plt.close()

# Chart 2: Churn by contract
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set2"
)
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_by_contract.png")
plt.close()

# Chart 3: Tenure comparison
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="tenure",
    hue="Churn",
    palette="Set2",
    legend=False
)
plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tenure_vs_churn.png")
plt.close()

# Chart 4: Monthly charges comparison
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_charges_vs_churn.png")
plt.close()

print("\nEDA COMPLETED!")
print("Charts saved in:", OUTPUT_DIR)
