# Customer Churn Prediction System

## Project Overview

The Customer Churn Prediction System is a Machine Learning project designed to help a telecom company identify customers who are likely to discontinue their services.

The system analyzes historical customer data, identifies churn patterns, predicts customer churn probability, and highlights high-risk customers to support customer retention strategies.

## Business Problem

Customer churn is a major challenge for subscription-based businesses.

The telecom company wants to understand:

- Which customers are likely to leave?
- What customer patterns are associated with churn?
- How can the company identify high-risk customers?
- How can the business improve customer retention?

## Project Objectives

- Clean and preprocess customer data.
- Perform Exploratory Data Analysis (EDA).
- Identify important churn patterns.
- Build a Machine Learning classification model.
- Evaluate model performance.
- Predict individual customer churn probability.
- Develop an interactive Streamlit dashboard.
- Generate a high-risk customer report in CSV format.

## Dataset

**Dataset:** IBM Telco Customer Churn

**Source:** Kaggle  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset information:

- Total Records: 7,043 customers
- Total Features: 21 columns
- Target Variable: Churn
- Problem Type: Binary Classification

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Preprocessing
5. Train, Validation and Test Split
6. Machine Learning Model Training
7. Threshold Optimization
8. Model Evaluation
9. Customer Churn Prediction
10. Interactive Dashboard Development

## Data Analysis Insights

Key findings from the dataset:

- Overall customer churn rate: 26.54%
- Month-to-month contract churn rate: 42.71%
- One-year contract churn rate: 11.27%
- Two-year contract churn rate: 2.83%
- Average tenure of retained customers: 37.57 months
- Average tenure of churned customers: 17.98 months

Customers with month-to-month contracts, shorter tenure, and higher monthly charges showed stronger associations with churn.

## Machine Learning Model

**Model:** Logistic Regression

The model uses preprocessing pipelines for numerical and categorical features.

A validation-based threshold optimization process was used to select a classification threshold of 0.55.

## Model Performance

Final test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 75.73% |
| Precision | 52.94% |
| Recall | 77.01% |
| F1 Score | 62.75% |
| ROC-AUC | 83.21% |

The model identifies approximately 77% of actual churn customers in the held-out test set.

## Streamlit Dashboard

The interactive dashboard includes:

- Customer overview and KPI metrics
- Churn vs retained customer visualization
- Contract-wise churn analysis
- Customer tenure analysis
- Monthly charges analysis
- Individual customer churn prediction
- High-risk customer analysis
- Downloadable CSV report

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── train_model.py
│   └── predict_customer.py
│
├── app.py
├── README.md
└── requirements.txt
```

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Customer_Churn_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

**Important:** Download the dataset from Kaggle and place the CSV file inside the `data/` directory before running the data-processing scripts.

## Business Recommendations

Based on the exploratory analysis, the company could:

- Review retention strategies for month-to-month customers.
- Monitor customers with short subscription tenure.
- Evaluate pricing and service concerns among higher-charge customers.
- Contact customers flagged as high risk with suitable retention offers.
- Track retention campaign outcomes before changing business policies.

These are recommendations based on observed patterns, not proof of causation.

## Important Notes

The model is a baseline predictive system. Its predictions are estimates, not guarantees. The historical high-risk dashboard ranking should be treated as exploratory because it scores the same historical records used during model development.

Further validation on unseen, future customer data is recommended before production deployment.

## Author

Data Science & Machine Learning Project

Skills demonstrated: Python, Data Analysis, EDA, Machine Learning, Model Evaluation, Streamlit and Data Visualization.