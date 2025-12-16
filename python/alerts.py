import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

df = pd.read_csv(DATA_PATH)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

model = joblib.load(MODEL_PATH)

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
df['Churn_Risk'] = model.predict_proba(X)[:, 1]

high_risk = df[df['Churn_Risk'] > 0.7]

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "high_risk_customers.csv")
high_risk.to_csv(OUTPUT_PATH, index=False)

print("High-risk customers file created")
