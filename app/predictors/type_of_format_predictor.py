import numpy as np

from app.model_loader import (
    format_model,
    format_label_encoder
)


def predict_format(text: str):

    probs = format_model.predict_proba(
        [text]
    )[0]

    pred = format_model.predict(
        [text]
    )[0]

    label = (
        format_label_encoder
        .inverse_transform([pred])[0]
    )

    confidence = float(np.max(probs))

    class_probs = {}

    for class_name, prob in zip(
        format_label_encoder.classes_,
        probs
    ):

        class_probs[class_name] = round(
            float(prob),
            4
        )

    return {

        "prediction": label,

        "confidence": round(
            confidence,
            4
        ),

        "probabilities": class_probs
    }