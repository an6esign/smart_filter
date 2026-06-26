import re
from pathlib import Path

import joblib


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGE_MODEL_PATH = PROJECT_ROOT / "app" / "models" / "age_model.joblib"


# =========================
# LOAD MODEL
# =========================

age_bundle = joblib.load(AGE_MODEL_PATH)

# Новый вариант: сохранен словарь {"model": model, ...}
if isinstance(age_bundle, dict):
    age_model = age_bundle["model"]

# Старый вариант: сохранена сразу модель
else:
    age_model = age_bundle


# =========================
# AGE RULES
# =========================

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("ё", "е")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def age_number_to_group(age):
    """
    Переводит реальный возраст клиента в возрастную категорию квеста.

    Например:
    2 года -> 6+
    5 лет  -> 6+
    7 лет  -> 6+
    8 лет  -> 8+
    10 лет -> 10+
    12 лет -> 12+
    18 лет -> 18+
    """

    age = int(age)

    if age <= 7:
        return "6+"
    elif age <= 9:
        return "8+"
    elif age <= 11:
        return "10+"
    elif age <= 13:
        return "12+"
    elif age <= 15:
        return "14+"
    elif age <= 17:
        return "16+"
    else:
        return "18+"


def extract_age_rule(text):
    """
    Правила извлечения возраста из текста.

    Важно:
    если пользователь пишет "ребенку 2 года",
    возвращаем 6+, потому что минимальная категория — 6+.
    """

    text = normalize_text(text)

    # =========================
    # 1. Явные рейтинги: 6+, 8+, 10+, 12+, 14+, 16+, 18+
    # =========================

    plus_match = re.search(
        r"\b(6|8|10|12|14|16|18)\s*\+",
        text
    )

    if plus_match:
        return f"{plus_match.group(1)}+"

    # =========================
    # 2. Диапазоны: 2-5 лет, 6-9 лет, 12-13 лет
    # Берем минимальный возраст.
    # =========================

    range_match = re.search(
        r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(?:лет|года|год|г\.?)?\b",
        text
    )

    if range_match:
        age_1 = int(range_match.group(1))
        age_2 = int(range_match.group(2))

        min_age = min(age_1, age_2)

        return age_number_to_group(min_age)

    # =========================
    # 3. Явный возраст:
    # 2 года, 4 года, 5 лет, 10 лет, 18 лет
    # =========================

    age_matches = re.findall(
        r"\b(\d{1,2})\s*(?:лет|года|год|годик|годика|годиков|г\.?)\b",
        text
    )

    if age_matches:
        ages = [int(x) for x in age_matches]

        # Если возрастов несколько, например:
        # "дети 5 и 8 лет" — безопаснее взять минимальный возраст.
        min_age = min(ages)

        return age_number_to_group(min_age)

    # =========================
    # 4. Контекстные конструкции:
    # ребенку 2, ребенку 5, возраст 6, детям 8
    # =========================

    age_context_match = re.search(
        r"\b(?:"
        r"возраст|возраста|"
        r"ребенку|ребенок|ребёнку|ребёнок|"
        r"дочке|сыну|"
        r"детям|ребятам|"
        r"девочке|мальчику|"
        r"девушка|парень|подросток|подростку"
        r")\s*(?:от|с)?\s*(\d{1,2})\b",
        text
    )

    if age_context_match:
        age = int(age_context_match.group(1))

        return age_number_to_group(age)

    # =========================
    # 5. Конструкции:
    # от 6 лет, с 8 лет
    # =========================

    from_age_match = re.search(
        r"\b(?:от|с)\s*(\d{1,2})\s*(?:лет|года|год|г\.?)\b",
        text
    )

    if from_age_match:
        age = int(from_age_match.group(1))

        return age_number_to_group(age)

    return None


# =========================
# PREDICT
# =========================

def predict_age(text):
    """
    Главная функция предсказания возраста.

    Сначала применяем правила.
    Если правило нашло возраст — возвращаем его.
    Если нет — используем ML-модель.
    """

    text = normalize_text(text)

    rule_age = extract_age_rule(text)

    if rule_age is not None:
        return {
            "has_age": 1,
            "age": rule_age,
            "source": "rule"
        }

    pred_age = age_model.predict([text])[0]

    return {
        "has_age": 1,
        "age": pred_age,
        "source": "model"
    }