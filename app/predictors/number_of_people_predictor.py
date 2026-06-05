from app.predictors.has_number_predictor import (
    predict_has_number
)

from app.services.number_of_people_extractor import (
    extract_number_of_people
)


def predict_number_of_people(text: str):

    has_number_result = predict_has_number(
        text
    )

    has_number = (
        has_number_result["prediction"]
    )

    number_of_people = (
        extract_number_of_people(
            text=text,
            has_number=has_number
        )
    )

    return {
        "prediction": number_of_people
    }