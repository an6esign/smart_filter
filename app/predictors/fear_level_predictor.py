import numpy as np

from app.model_loader import fear_level_model


def predict_fear_level(text: str):

    pred = fear_level_model.predict([text])[0].tolist()

    probs = fear_level_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }