import numpy as np

from app.model_loader import format_model



def predict_format(text: str):

    pred = int(
        format_model.predict([text])[0]
    )

    probs = format_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }