import pickle

from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn




class LeadScore(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictResponse(BaseModel):
    convert_probability: float
    converted: bool


app = FastAPI(title="converted-prediction")

with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)


def predict_single(lead_score):
    result = pipeline.predict_proba(lead_score)[0, 1]
    return float(result)


@app.post("/predict")
def predict(lead_score: LeadScore) -> PredictResponse:
    prob = predict_single(lead_score.model_dump())

    return PredictResponse(
        convert_probability=prob,
        converted=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
