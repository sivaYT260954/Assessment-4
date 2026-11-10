
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI()

model = pipeline(
    "text-classification",
    model="./saved_model",
    tokenizer="./saved_model"
)

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    result = model(request.text)[0]

    return {
        "label": result["label"],
        "confidence": round(float(result["score"]), 4)
    }

     
