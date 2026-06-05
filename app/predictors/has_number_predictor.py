import numpy as np

from app.model_loader import has_number_model


def predict_has_number(text: str):

    pred = int(
        has_number_model.predict([text])[0]
    )

    probs = has_number_model.predict_proba([text])[0]

    confidence = float(
        np.max(probs)
    )

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }