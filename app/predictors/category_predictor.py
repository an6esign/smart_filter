import numpy as np

from app.model_loader import (
    category_model,
    category_names
)


def predict_category(text: str):

    # raw prediction
    raw_pred = category_model.predict([text])[0]

    # probabilities
    probs = category_model.predict_proba([text])[0]

    # max confidence
    confidence = float(np.max(probs))

    # one-hot -> category names
    predicted_categories = [
        category_names[i]
        for i, value in enumerate(raw_pred)
        if int(value) == 1
    ]

    return {
        "prediction": predicted_categories,
        "confidence": round(confidence, 4)
    }