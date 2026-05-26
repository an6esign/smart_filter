import pandas as pd

from tqdm.auto import tqdm

from app.predictors.fear_predictor import (
    predict_fear
)

from app.predictors.fear_level_predictor import (
    predict_fear_level
)

from app.predictors.difficulty_predictor import (
    predict_difficulty
)

from app.predictors.difficulty_level_predictor import (
    predict_difficulty_level
)

from app.predictors.has_format_predictor import (
    predict_has_format
)

from app.predictors.type_of_format_predictor import (
    predict_format
)

from app.predictors.category_predictor import (
    predict_category
)

# =========================================
# LABELS
# =========================================

FEAR_LEVEL_LABELS = [
    "0",
    "0,1",
    "1",
    "1,2",
    "2",
    "2,3",
    "3"
]

DIFFICULTY_LEVEL_LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

# =========================================
# DECODER
# =========================================

def decode_single_label(pred, labels):

    # если уже строка
    if isinstance(pred, str):
        return pred

    # если число
    if not hasattr(pred, "__iter__"):
        return str(pred)

    # one-hot -> label
    for i, value in enumerate(pred):

        if value == 1:
            return labels[i]

    return "unknown"

# =========================================
# CONFIG
# =========================================

INPUT_PATH = "data/facet_predictions.xlsx"

OUTPUT_PATH = "full_markup.xlsx"

TEXT_COL = "query_text"

# =========================================
# LOAD DATA
# =========================================

print("Загрузка данных...")

df = pd.read_excel(INPUT_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(
        f"Колонка '{TEXT_COL}' не найдена"
    )

# чистим текст
df[TEXT_COL] = (
    df[TEXT_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(f"Всего строк: {len(df)}")

# =========================================
# INFERENCE
# =========================================

results = []

errors = 0

for text in tqdm(
    df[TEXT_COL],
    total=len(df),
    desc="Разметка"
):

    try:

        # =====================================
        # FEAR
        # =====================================

        fear = predict_fear(text)

        has_fear = fear["prediction"]

        if has_fear == 1:

            fear_level_raw = (
                predict_fear_level(text)
            )["prediction"]

            fear_level = decode_single_label(
                fear_level_raw,
                FEAR_LEVEL_LABELS
            )

        else:

            fear_level = "unknown"

        # =====================================
        # DIFFICULTY
        # =====================================

        difficulty = predict_difficulty(text)

        has_difficulty = (
            difficulty["prediction"]
        )

        if has_difficulty == 1:

            difficulty_level_raw = (
                predict_difficulty_level(text)
            )["prediction"]

            difficulty_level = decode_single_label(
                difficulty_level_raw,
                DIFFICULTY_LEVEL_LABELS
            )

        else:

            difficulty_level = "unknown"

        # =====================================
        # FORMAT
        # =====================================

        has_format_result = (
            predict_has_format(text)
        )

        has_format = (
            has_format_result["prediction"]
        )

        if has_format == 1:

            format_pred = (
                predict_format(text)
            )["prediction"]

        else:

            format_pred = "unknown"

        # =====================================
        # CATEGORY
        # =====================================

        category = (
            predict_category(text)
        )["prediction"]

        # =====================================
        # SAVE
        # =====================================

        results.append({

            "query_text": text,

            # fear
            "has_fear": has_fear,

            "fear_confidence": (
                fear["confidence"]
            ),

            "fear_level": fear_level,

            # difficulty
            "has_difficulty": (
                has_difficulty
            ),

            "difficulty_confidence": (
                difficulty["confidence"]
            ),

            "difficulty_level": (
                difficulty_level
            ),

            # format
            "has_format": has_format,

            "format": format_pred,

            # category
            "category": category

        })

    except Exception as e:

        errors += 1

        print(f"\nОшибка: {e}")

        results.append({

            "query_text": text,

            "has_fear": "error",
            "fear_confidence": "error",
            "fear_level": "error",

            "has_difficulty": "error",
            "difficulty_confidence": "error",
            "difficulty_level": "error",

            "has_format": "error",
            "format": "error",

            "category": "error"

        })

# =========================================
# SAVE EXCEL
# =========================================

print("\nСохранение Excel...")

result_df = pd.DataFrame(results)

result_df.to_excel(
    OUTPUT_PATH,
    index=False
)

# =========================================
# DONE
# =========================================

print("\n=================================")
print("Готово")
print(f"Файл сохранен: {OUTPUT_PATH}")
print(f"Ошибок: {errors}")
print("=================================")