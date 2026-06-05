from io import BytesIO

import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import StreamingResponse

from app.predictors.fear_predictor import predict_fear
from app.predictors.fear_level_predictor import predict_fear_level

from app.predictors.difficulty_predictor import predict_difficulty
from app.predictors.difficulty_level_predictor import predict_difficulty_level

from app.predictors.has_format_predictor import predict_has_format
from app.predictors.type_of_format_predictor import predict_format

from app.predictors.category_predictor import predict_category

from app.predictors.has_age_predictor import predict_has_age
from app.predictors.age_predictor import predict_age


router = APIRouter()

TEXT_COL = "query_text"


@router.post("/predict/excel")
async def predict_excel(
    file: UploadFile = File(...)
):

    if not file.filename.endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="Загрузи файл в формате .xlsx"
        )

    contents = await file.read()

    try:
        df = pd.read_excel(
            BytesIO(contents)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать Excel: {e}"
        )

    if TEXT_COL not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"В файле нет колонки '{TEXT_COL}'"
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    has_fear_values = []
    fear_level_values = []

    has_difficulty_values = []
    difficulty_level_values = []

    has_format_values = []
    format_type_values = []

    has_age_values = []
    age_values = []

    category_values = []

    for text in df[TEXT_COL]:

        # =========================
        # FEAR
        # =========================

        fear_result = predict_fear(text)

        has_fear = fear_result["prediction"]

        if has_fear == 1:
            fear_level_result = predict_fear_level(text)
        else:
            fear_level_result = {
                "prediction": "unknown"
            }

        fear_level = fear_level_result["prediction"]

        # =========================
        # DIFFICULTY
        # =========================

        difficulty_result = predict_difficulty(text)

        has_difficulty = difficulty_result["prediction"]

        if has_difficulty == 1:
            difficulty_level_result = predict_difficulty_level(text)
        else:
            difficulty_level_result = {
                "prediction": "unknown"
            }

        difficulty_level = difficulty_level_result["prediction"]

        if isinstance(difficulty_level, dict):
            difficulty_level = ",".join(
                key
                for key, value in difficulty_level.items()
                if value == 1
            )

            if difficulty_level == "":
                difficulty_level = "unknown"

        # =========================
        # FORMAT
        # =========================

        has_format_result = predict_has_format(text)

        has_format = has_format_result["prediction"]

        if has_format == 1:
            format_result = predict_format(text)
        else:
            format_result = {
                "prediction": "unknown"
            }

        format_type = format_result["prediction"]

        # =========================
        # AGE
        # =========================

        has_age_result = predict_has_age(text)

        has_age = has_age_result["prediction"]

        if has_age == 1:
            age_result = predict_age(text)
        else:
            age_result = {
                "age": "unknown"
            }

        age = age_result["age"]

        # =========================
        # CATEGORY
        # =========================

        category_result = predict_category(text)

        categories = category_result["prediction"]

        if isinstance(categories, list):
            categories = ", ".join(
                map(str, categories)
            )

        # =========================
        # APPEND RESULTS
        # =========================

        has_fear_values.append(has_fear)
        fear_level_values.append(fear_level)

        has_difficulty_values.append(has_difficulty)
        difficulty_level_values.append(difficulty_level)

        has_format_values.append(has_format)
        format_type_values.append(format_type)

        has_age_values.append(has_age)
        age_values.append(age)

        category_values.append(categories)

    df["has_fear"] = has_fear_values
    df["fear_level"] = fear_level_values

    df["has_difficulty"] = has_difficulty_values
    df["difficulty_level"] = difficulty_level_values

    df["has_format"] = has_format_values
    df["format_type"] = format_type_values

    df["has_age"] = has_age_values
    df["age"] = age_values

    df["category"] = category_values

    output = BytesIO()

    df.to_excel(
        output,
        index=False
    )

    output.seek(0)

    return StreamingResponse(
        output,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": (
                "attachment; filename=marked_predictions.xlsx"
            )
        }
    )