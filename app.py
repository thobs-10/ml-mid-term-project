from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Literal, Any
import joblib
import uvicorn
from src.exception import AppException

app = FastAPI()

class PredictionRequest(BaseModel):
    temperature : float
    hour: int
    solar_radication : float
    seasons: str
    dew_point_temperature : float
    functioning_day: bool
    rainfall: float

    def __post__init__(self):
        if self.seasons not in ['Winter', 'Spring', 'Summer', 'Fall']:
            raise HTTPException(status_code=400, detail="Invalid seasons. Expected 'Winter', 'Spring', 'Summer', or 'Fall'")
        if not (0 <= self.hour <= 23):
            raise HTTPException(status_code=400, detail="Invalid hour. Expected a value between 0 and 23")
        if not (0 <= self.solar_radiation <= 10):
            raise HTTPException(status_code=400, detail="Invalid solar radiation. Expected a value between 0 and 10")
        if not (0 <= self.dew_point_temperature <= 30):
            raise HTTPException(status_code=400, detail="Invalid dew point temperature. Expected a value between 0 and 30")
        if not (0 <= self.rainfall <= 100):
            raise HTTPException(status_code=400, detail="Invalid rainfall. Expected a value between 0 and 100")
        if self.functioning_day not in [True, False]:
            raise HTTPException(status_code=400, detail="Invalid functioning day. Expected a boolean value")


class PredictionResponse(BaseModel):
    prediction: float
    message : Union[Dict[dict, str], Any]

class StatusResponse(BaseModel):
    message: str
    status : Dict[dict, str]

async def load_model_pipeline():
    model_path = ''
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError as e:
        raise AppException("Failed to load model", e)

async def get_request(request: PredictionRequest)-> dict:
    mode_input_dict= {
    'temperature' : request.temperature,
    'hour' : request.hour,
    'solar_radiation' : request.solar_radiation,
    'seasons' : request.seasons,
    'dew_point_temperature' : request.dew_point_temperature,
    'functioning_day' : request.functioning_day,
    'rainfall' : request.rainfall
    }
    return mode_input_dict

@app.get('/', response_model=PredictionResponse, status_code= status.HTTP_200_OK)
async def health():
    return {"message": {"status": "OK, welcome to the bike sharing API ML project"}}

@app.post("/predict", response_model=PredictionResponse, status_code= status.HTTP_202_ACCEPTED)
async def predict(request: PredictionRequest):
   
    model_request = get_request(request)
    model = await load_model_pipeline()
    prediction = model.predict(**model_request)

    return { prediction: {"message": "Number of bikes needed"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)