from fastapi import FastAPI

from app.schemas import PredictRequest

from app.predictors.fear_predictor import predict_fear
from app.predictors.fear_level_predictor import predict_fear_level
from app.predictors.difficulty_predictor import predict_difficulty
from app.predictors.difficulty_level_predictor import predict_difficulty_level
from app.predictors.format_predictor import predict_format
from app.predictors.category_predictor import predict_category


app = FastAPI(
    title="Smart Filter API"
)


@app.get("/")
def healthcheck():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: PredictRequest):

    text = request.text

    fear_result = predict_fear(text)

    if fear_result["prediction"] == 1:

        fear_level_result = predict_fear_level(text)

    else:

        fear_level_result = "unknown"

    difficulty_result = predict_difficulty(text)
    
    if difficulty_result["prediction"] == 1:

        difficulty_level_result = predict_difficulty_level(text)

    else:

        difficulty_level_result = "unknown"

    category_result = predict_category(text)

    format_result = predict_format(text)

    return {

        "text": text,

        "has_fear": fear_result,

        "fear_level": fear_level_result,

        "has_difficulty": difficulty_result,

        "difficulty_level": difficulty_level_result,

        "has_format": format_result,

        "category": category_result
    }