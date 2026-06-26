from app.predictors.has_number_predictor import (
    predict_has_number
)

from app.services.number_of_people_extractor import (
    extract_number_of_people
)


UNKNOWN = "unknown"


def predict_number_of_people(text: str):
    """
    Предсказывает количество людей в запросе.

    Логика:
    1. Сначала модель predict_has_number определяет,
       есть ли в тексте информация о количестве людей.
    2. Если has_number = 0, возвращаем unknown.
    3. Если has_number = 1, применяем rule-based extractor
       и достаём конкретное количество людей.
    """

    has_number_result = predict_has_number(text)

    has_number = has_number_result["prediction"]

    try:
        has_number = int(has_number)
    except Exception:
        has_number = 0

    if has_number != 1:
        return {
            "prediction": UNKNOWN,
            "has_number": has_number
        }

    number_of_people = extract_number_of_people(
        text=text,
        has_number=has_number
    )

    return {
        "prediction": number_of_people,
        "has_number": has_number
    }