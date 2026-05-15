from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
import gradio as gr

app = FastAPI()

# 1. LOAD MODELS
preprocessor = joblib.load("preprocessor.pkl")
model_f = joblib.load("final_stack_f.pkl")
model_m = joblib.load("final_stack_m.pkl")

AGE_MEDIANS = {"Mr": 30, "Mrs": 35, "Miss": 21, "Master": 4, "Rare": 48}
FARE_MEDIAN = 14.45
FINAL_COLUMNS = [
    "Embarked", "Pclass", "Title", "Ticket_Group_Size",
    "Family_Survival", "Fare_Bin", "Age_Bin", "Family_Size",
    "IsAlone", "Deck"
]

@app.get("/")
def root():
    return {"message": "Titanic survival API is running. Visit /gui for the web interface."}

@app.post("/predict")
def predict(passenger_data: dict):
    df = pd.DataFrame([passenger_data])

    # Defaults and Preprocessing (keeping your existing logic)
    for col, val in [("Pclass", 3), ("Embarked", "S"), ("Age", np.nan), ("Fare", FARE_MEDIAN), ("SibSp", 0), ("Parch", 0), ("Cabin", None)]:
        if col not in df.columns:
            df[col] = val

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs").fillna("Rare")

    df["Age"] = df.apply(lambda r: AGE_MEDIANS.get(r["Title"], 28) if pd.isnull(r["Age"]) else r["Age"], axis=1)
    df["Fare"] = df["Fare"].fillna(FARE_MEDIAN)
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["Family_Size"] == 1).astype(int)
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "U")
    df["Ticket_Group_Size"] = 1
    df["Family_Survival"] = 0.5
    df["Fare_Bin"] = pd.cut(df["Fare"], bins=[-1, 7.85, 10.5, 21.0, 39.7, 600], labels=False).astype(int)
    df["Age_Bin"] = pd.cut(df["Age"], bins=[-1, 16, 32, 48, 64, 100], labels=False).astype(int)

    sex = df["Sex"].iloc[0].lower()
    X = df[FINAL_COLUMNS]
    X_proc = preprocessor.transform(X)

    pred = model_f.predict(X_proc) if sex == "female" else model_m.predict(X_proc)

    return {"survived": int(pred[0]), "sex_model_used": sex}

# 2. GRADIO INTERFACE LOGIC
def gradio_predict(name, sex, age, pclass, fare, sibsp, parch, embarked, cabin):
    # Map the Gradio inputs back into the dictionary format your predict function expects
    payload = {
        "Name": name, "Sex": sex, "Age": age, "Pclass": pclass,
        "Fare": fare, "SibSp": sibsp, "Parch": parch,
        "Embarked": embarked, "Cabin": cabin
    }
    result = predict(payload)
    return "Survived" if result["survived"] == 1 else "Perished"

# 3. BUILD THE WEB UI
titanic_ui = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(label="Passenger Name", placeholder="e.g. Mr. Owen Harris Braund"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(0, 100, value=22, label="Age"),
        gr.Dropdown([1, 2, 3], label="Passenger Class (Pclass)", value=3),
        gr.Number(label="Fare", value=7.25),
        gr.Number(label="Siblings/Spouses Aboard (SibSp)", value=0),
        gr.Number(label="Parents/Children Aboard (Parch)", value=0),
        gr.Dropdown(["S", "C", "Q"], label="Port of Embarkation", value="S"),
        gr.Textbox(label="Cabin Number (Optional)", placeholder="e.g. C123")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Titanic Survival Predictor",
    description="Fill in the passenger details to predict survival using the MLOps pipeline."
)

# 4. MOUNT GRADIO TO FASTAPI
app = gr.mount_gradio_app(app, titanic_ui, path="/gui")