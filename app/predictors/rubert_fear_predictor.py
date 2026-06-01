import torch

from transformers import AutoTokenizer

from app.model_loader import (
    rubert_model,
    RUBERT_MODEL_DIR,
    DEVICE
)


tokenizer = AutoTokenizer.from_pretrained(
    str(RUBERT_MODEL_DIR)
)


def decode_fear_onehot(onehot):
    labels = []

    for idx, value in enumerate(onehot):
        if value == 1:
            labels.append(str(idx))

    if len(labels) == 0:
        return "unknown"

    return ",".join(labels)


def predict_rubert_fear(text: str):
    text = str(text).strip()

    if text == "":
        return {
            "has_fear": 0,
            "fear_level": "unknown",
            "fear_onehot": [0, 0, 0, 0],
            "has_fear_confidence": 0.0,
            "fear_level_confidence": 0.0
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        has_fear_logits, fear_level_logits = rubert_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    has_fear_prob = torch.sigmoid(
        has_fear_logits
    )

    has_fear_pred = int(
        has_fear_prob.item() >= 0.5
    )

    has_fear_confidence = float(
        has_fear_prob.item()
        if has_fear_pred == 1
        else 1 - has_fear_prob.item()
    )

    if has_fear_pred == 0:
        return {
            "has_fear": 0,
            "fear_level": "unknown",
            "fear_onehot": [0, 0, 0, 0],
            "has_fear_confidence": round(has_fear_confidence, 4),
            "fear_level_confidence": 0.0
        }

    fear_level_probs = torch.sigmoid(
        fear_level_logits
    )

    fear_onehot = (
        fear_level_probs >= 0.5
    ).int().cpu().numpy()[0].tolist()

    fear_level = decode_fear_onehot(
        fear_onehot
    )

    selected_probs = [
        float(prob)
        for prob, label in zip(
            fear_level_probs.cpu().numpy()[0],
            fear_onehot
        )
        if label == 1
    ]

    if len(selected_probs) == 0:
        fear_level_confidence = float(
            torch.max(fear_level_probs).item()
        )
    else:
        fear_level_confidence = sum(selected_probs) / len(selected_probs)

    return {
        "has_fear": 1,
        "fear_level": fear_level,
        "fear_onehot": fear_onehot,
        "has_fear_confidence": round(has_fear_confidence, 4),
        "fear_level_confidence": round(fear_level_confidence, 4)
    }