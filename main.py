from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

preprocessor = joblib.load("preprocessor.pkl")
model_f = joblib.load("final_stack_f.pkl")
model_m = joblib.load("final_stack_m.pkl")

AGE_MEDIANS = {
    "Mr": 30,
    "Mrs": 35,
    "Miss": 21,
    "Master": 4,
    "Rare": 48
}

FARE_MEDIAN = 14.45

FINAL_COLUMNS = [
    "Embarked",
    "Pclass",
    "Title",
    "Ticket_Group_Size",
    "Family_Survival",
    "Fare_Bin",
    "Age_Bin",
    "Family_Size",
    "IsAlone",
    "Deck",
]


@app.get("/")
def root():
    return {"message": "Titanic survival API is running"}


@app.post("/predict")
def predict(passenger_data: dict):
    df = pd.DataFrame([passenger_data])

    # Defaults for missing API fields
    if "Pclass" not in df.columns:
        df["Pclass"] = 3

    if "Embarked" not in df.columns:
        df["Embarked"] = "S"

    if "Age" not in df.columns:
        df["Age"] = np.nan

    if "Fare" not in df.columns:
        df["Fare"] = FARE_MEDIAN

    if "SibSp" not in df.columns:
        df["SibSp"] = 0

    if "Parch" not in df.columns:
        df["Parch"] = 0

    if "Cabin" not in df.columns:
        df["Cabin"] = None

    # Title extraction
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
         "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    df["Title"] = df["Title"].fillna("Rare")

    # Imputation
    df["Age"] = df.apply(
        lambda r: AGE_MEDIANS.get(r["Title"], 28) if pd.isnull(r["Age"]) else r["Age"],
        axis=1
    )
    df["Fare"] = df["Fare"].fillna(FARE_MEDIAN)

    # Feature engineering
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["Family_Size"] == 1).astype(int)

    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "U")

    # Real-time-safe approximations
    df["Ticket_Group_Size"] = 1
    df["Family_Survival"] = 0.5

    # Match training-time binning approximately
    df["Fare_Bin"] = pd.cut(
        df["Fare"],
        bins=[-1, 7.85, 10.5, 21.0, 39.7, 600],
        labels=False
    ).astype(int)

    df["Age_Bin"] = pd.cut(
        df["Age"],
        bins=[-1, 16, 32, 48, 64, 100],
        labels=False
    ).astype(int)

    sex = df["Sex"].iloc[0].lower()

    X = df[FINAL_COLUMNS]

    X_proc = preprocessor.transform(X)

    if sex == "female":
        pred = model_f.predict(X_proc)
    else:
        pred = model_m.predict(X_proc)

    return {
        "survived": int(pred[0]),
        "sex_model_used": sex
    }