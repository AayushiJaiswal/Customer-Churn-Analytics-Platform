import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_clean.csv")

df = pd.read_csv(DATA_PATH)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
joblib.dump(model, MODEL_PATH)

print("Model trained and saved successfully")
