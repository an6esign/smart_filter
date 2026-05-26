import numpy as np

from app.model_loader import has_format_model



def predict_has_format(text: str):

    pred = int(
        has_format_model.predict([text])[0]
    )

    probs = has_format_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    return {
        "prediction": pred,
        "confidence": round(confidence, 4)
    }