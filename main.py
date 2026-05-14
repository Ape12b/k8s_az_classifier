from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

# 1. LOAD EVERYTHING AT STARTUP
# This ensures the pod is ready to work immediately
preprocessor = joblib.load("preprocessor.pkl")
model_f = joblib.load("final_stack_f.pkl")
model_m = joblib.load("final_stack_m.pkl")

# Pre-calculated medians (from your training) to avoid data leakage/errors
AGE_MEDIANS = {"Mr": 30, "Mrs": 35, "Miss": 21, "Master": 4, "Rare": 48}
FARE_MEDIAN = 14.45

@app.post("/predict")
def predict(passenger_data: dict):
    df = pd.DataFrame([passenger_data])
    
    # 2. SIMPLIFIED PREPROCESSING (Real-time compatible)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Use pre-calculated medians instead of calculating from the request
    df['Age'] = df.apply(lambda r: AGE_MEDIANS.get(r['Title'], 28) if pd.isnull(r['Age']) else r['Age'], axis=1)
    df['Fare'] = df['Fare'].fillna(FARE_MEDIAN)
    
    # Feature Engineering
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['Family_Size'] == 1).astype(int)

    # 3. MALE/FEMALE BRANCHING
    if df["Sex"].iloc[0] == "female":
        # Prepare data (drop columns not used by the female model)
        X = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1, errors='ignore')
        X_proc = preprocessor.transform(X)
        pred = model_f.predict(X_proc)
    else:
        # Prepare data for the male model
        X = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1, errors='ignore')
        X_proc = preprocessor.transform(X)
        pred = model_m.predict(X_proc)

    return {"survived": int(pred[0])}