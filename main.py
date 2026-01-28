from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# 🔽 Import model downloader
from utils.model_downloader import download_all_models

# 🔽 Import routers
from services.image_colorization.router import router as colorize_router
from services.stock_prediction.router import router as stock_router
from services.house_price.router import router as house_router
from services.fraud_transaction.router import router as fraud_tx_router
from services.fraud_insurance.router import router as fraud_ins_router
from services.phishing_email.router import router as phishing_router
from services.diabetic_retinopathy.router import router as dr_router


app = FastAPI(title="Unified ML Backend")

# 🔽 Ensure runtime folders exist
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# 🔽 Download models ONCE at startup (SAFE)
@app.on_event("startup")
def startup_event():
    download_all_models()

# 🔽 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔽 Serve result files
app.mount("/results", StaticFiles(directory="results"), name="results")

# 🔽 Register API routers
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
