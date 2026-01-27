# services/fix_models.py
import os
import joblib

# Path to your stock_prediction models folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "stock_prediction", "models")

model_files = [
    "xgb_best_model.joblib",
    "xgb_best_model1.joblib",
    "xgb_best_model2.joblib",
    "xgb_best_model3.joblib"
]

for file in model_files:
    path = os.path.join(MODELS_DIR, file)
    model = joblib.load(path)
    # Save in XGBoost native JSON format
    model.get_booster().save_model(path.replace(".joblib", ".json"))
    print(f"{file} re-saved as JSON successfully")
