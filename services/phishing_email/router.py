# services/phishing_email/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pickle, re, traceback
import nltk
from nltk.corpus import stopwords

router = APIRouter()


SERVICE_DIR = Path(__file__).resolve().parent
MODEL_PATH = SERVICE_DIR / "phishing_detector.pkl"


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


if not MODEL_PATH.is_file():
    raise RuntimeError(f"Missing phishing model: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class EmailInput(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

@router.post("/predict")
def predict(input: EmailInput):
    try:
        cleaned = clean_text(input.text)

        # Ensure model can handle list input
        probs = model.predict_proba([cleaned])[0]
        pred = model.predict([cleaned])[0]

        label_map = {0.0: "Legitimate", 1.0: "Phishing"}

        return {
            "raw_text": input.text,
            "cleaned_text": cleaned,
            "prediction": label_map.get(pred, str(pred)),
            "confidence": float(max(probs))
        }

    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {e}\n{traceback_str}"
        )
