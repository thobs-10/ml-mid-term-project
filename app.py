from fastapi import FastAPI, status
from dotenv import load_dotenv
import joblib
import os
import uvicorn
from src.exception import AppException
import pandas as pd
import numpy as np


load_dotenv()

app = FastAPI()

model_path = 'artifacts/staged_model'

async def preprocess_input(model_input: dict)-> pd.DataFrame:
    df = pd.DataFrame(model_input)
    df['seasons'] = df['seasons'].map({'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4})
    df['functioning_day'] = np.where(df['functioning_day'] =='Yes', 1, 0)
    return df

async def load_model_pipeline():
    full_path = os.path.join(model_path, 'model_pipeline.joblib')
    if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found at {full_path}")
    try:
        model = joblib.load(full_path)
        return model
    except FileNotFoundError as e:
        raise AppException("Failed to load model", e)

async def get_request(request: dict)-> dict:
    mode_input_dict= {
    'temperature' : [request['temperature']],
    'hour' : [request['hour']],
    'solar_radiation' : [request['solar_radiation']],
    'seasons' : [request['seasons']],
    'dew_point_temperature' : [request['dew_point_temperature']],
    'functioning_day' : [request['functioning_day']],
    'rainfall' : [request['rainfall']]
    }
    return mode_input_dict

@app.get('/', status_code= status.HTTP_200_OK)
async def health():
    return {"message ": "welcome to the bike sharing API ML project"}

@app.post("/predict", status_code= status.HTTP_202_ACCEPTED)
async def predict(request: dict):
   
    model_request = await get_request(request)
    input = await preprocess_input(model_request)
    model = await load_model_pipeline()
    prediction = model.predict(input)

    return {"message": "Number of bikes needed",
            "data": prediction[0]}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True,)