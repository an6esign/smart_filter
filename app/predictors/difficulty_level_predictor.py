import numpy as np

from app.model_loader import difficulty_level_model


THRESHOLD = 0.6


def make_final_difficulty(pred, proba):

    active = [
        i
        for i, value in enumerate(pred)
        if value == 1
    ]

    # ничего не прошло threshold
    if len(active) == 0:
        return str(int(np.argmax(proba)))

    # если классы идут подряд (например 0,1 или 1,2,3)
    # оставляем их как есть
    if max(active) - min(active) + 1 == len(active):
        return ",".join(map(str, active))

    # если есть разрыв (например 0,3 или 1,4)
    # берём самый вероятный класс
    return str(int(np.argmax(proba)))


def predict_difficulty_level(text: str):

    probs = difficulty_level_model.predict_proba([text])[0]

    pred = (
        probs >= THRESHOLD
    ).astype(int).tolist()

    final_label = make_final_difficulty(
        pred=pred,
        proba=probs
    )

    return {

        "prediction": {
            "0": pred[0],
            "1": pred[1],
            "2": pred[2],
            "3": pred[3],
            "4": pred[4]
        },

        "pred_difficulty_level": final_label,

        "probabilities": {
            "0": round(float(probs[0]), 4),
            "1": round(float(probs[1]), 4),
            "2": round(float(probs[2]), 4),
            "3": round(float(probs[3]), 4),
            "4": round(float(probs[4]), 4)
        },

        "confidence": round(
            float(np.max(probs)),
            4
        )
    }