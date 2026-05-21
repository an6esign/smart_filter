import joblib

fear_model = joblib.load(
    "app/models/fear_model.joblib"
)

difficulty_model = joblib.load(
    "app/models/difficulty_model.joblib"
)

category_model = joblib.load(
    "app/models/category_model.joblib"
)

category_names = joblib.load(
    "app/models/category_names.joblib"
)

format_model = joblib.load(
    "app/models/has_format_model.joblib"
)