from fastapi import FastAPI
import joblib
import pandas as pd

from app.schema import TransactionData

app = FastAPI(
    title="Credit Risk API"
)

model = joblib.load(
    "models/best_model.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Credit Risk Model API"
    }


app = FastAPI()

model = joblib.load(
    "models/best_model.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Credit Risk API Running"
    }


@app.post("/predict")
def predict(data: TransactionData):

    df = pd.DataFrame(
        [data.model_dump()]
    )

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }
