from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# 🔽 Model downloader
from utils.model_downloader import download_all_models

# 🔽 Ensure runtime folders exist
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# 🔽 Download missing models safely BEFORE importing routers
try:
    download_all_models()
except Exception as e:
    print(f"⚠️ Warning: Some models could not be downloaded: {e}")

# 🔽 Now import routers (safe because models exist)
from services.image_colorization.router import router as colorize_router
from services.stock_prediction.router import router as stock_router
from services.house_price.router import router as house_router
from services.fraud_transaction.router import router as fraud_tx_router
from services.fraud_insurance.router import router as fraud_ins_router
from services.phishing_email.router import router as phishing_router
from services.diabetic_retinopathy.router import router as dr_router

app = FastAPI(title="Unified ML Backend")

# 🔽 CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔽 Serve results folder
app.mount("/results", StaticFiles(directory="results"), name="results")

# 🔽 Register routers
app.include_router(colorize_router, prefix="/api/colorize", tags=["Image Colorization"])
app.include_router(stock_router, prefix="/api/stocks", tags=["Stock Prediction"])
app.include_router(house_router, prefix="/api/house", tags=["House Price"])
app.include_router(fraud_tx_router, prefix="/api/fraud/transaction", tags=["Transaction Fraud"])
app.include_router(fraud_ins_router, prefix="/api/fraud/insurance", tags=["Insurance Fraud"])
app.include_router(phishing_router, prefix="/api/phishing", tags=["Phishing Detection"])
app.include_router(dr_router, prefix="/api/medical/dr", tags=["Diabetic Retinopathy"])

@app.get("/")
def root():
    return {"status": "Backend running successfully 🚀"}
