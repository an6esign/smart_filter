import numpy as np

from app.model_loader import fear_level_model


def make_final_fear_level(pred, proba):
    active = [
        i
        for i, value in enumerate(pred)
        if value == 1
    ]

    # ничего не прошло threshold
    if len(active) == 0:
        return str(int(np.argmax(proba)))

    # если классы идут подряд, например 0,1 или 1,2,3
    # оставляем их как есть
    if max(active) - min(active) + 1 == len(active):
        return ",".join(map(str, active))

    # если есть разрыв, например 0,3 или 1,3
    # берём самый вероятный класс
    return str(int(np.argmax(proba)))


def predict_fear_level(text: str):
    pred = fear_level_model.predict([text])[0].tolist()

    probs = fear_level_model.predict_proba([text])[0]

    confidence = float(np.max(probs))

    final_prediction = make_final_fear_level(
        pred=pred,
        proba=probs
    )

    return {
        "prediction": final_prediction,
        "raw_prediction": pred,
        "confidence": round(confidence, 4)
    }