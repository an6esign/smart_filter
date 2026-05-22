import numpy as np

from app.model_loader import difficulty_level_model


def predict_difficulty_level(text: str):

    pred = difficulty_level_model.predict([text])[0].tolist()

    probs = difficulty_level_model.predict_proba([text])[0]

    positive_probs = []

    for p, prob in zip(pred, probs):

        if p == 1:
            positive_probs.append(prob)

    if len(positive_probs) == 0:

        confidence = float(np.max(probs))

    else:

        confidence = float(np.mean(positive_probs))

    return {

        "prediction": {

            "0": pred[0],
            "1": pred[1],
            "2": pred[2],
            "3": pred[3],
            "4": pred[4]
        },

        "confidence": round(confidence, 4)
    }