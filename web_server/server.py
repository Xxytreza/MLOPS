from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 
app = FastAPI()
import os
MODEL_PATH = "/app/regression.joblib"
DATA_PATH = "/app/houses.csv"

print(os.getcwd())

pipe = joblib.load(MODEL_PATH) 
expected_columns = ["size", "nb_rooms", "garden"]

class House(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.get("/predict")
async def predict():
    return {"y_predict" : 2}

@app.post("/predict")
async def predict(house: House):
    input_df = pd.DataFrame([house.dict()])[expected_columns]

    prediction = pipe.predict(input_df)

    return {"y_predict": prediction.tolist()} 


