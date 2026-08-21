# =============================================================================
# main.py
# FASTAPI CAR PRICE PREDICTION API
# =============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predictor import predict_car_price


# =============================================================================
# CREATE FASTAPI APP
# =============================================================================




# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STATIC FILES / FRONTEND
# =============================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_app():
    return FileResponse("static/index.html")


# =============================================================================
# INPUT SCHEMA
# =============================================================================

class CarInput(BaseModel):

    color: str = "Unknown"
    location: str = "Unknown"
    description: str = ""

    fuel_type: str = "Unknown"
    mileage: float | None = None
    body_type: str = "Unknown"
    engine_size: float | None = None

    second_condition: str = "Unknown"
    interior_color: str = "Unknown"
    registered_car: str = "Unknown"

    powertrain_type: str = "Unknown"
    drivetrain: str = "Unknown"

    cylinders: float | None = None
    seats: float | None = None

    make: str
    model: str
    year: float

    trim: str = "Unknown"

    condition: str
    transmission: str

    horsepower: float | None = None


# =============================================================================
# HOME
# =============================================================================

@app.get("/")
def home():

    return {
        "message": "Nigeria Car Price Prediction API",
        "status": "running",
        "model": "Step 21 XGBoost"
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =============================================================================
# PREDICT
# =============================================================================

@app.post("/predict")
def predict(car: CarInput):

    # -------------------------------------------------------------------------
    # Convert request to dictionary
    # -------------------------------------------------------------------------

    car_data = car.model_dump()

    # -------------------------------------------------------------------------
    # Run the COMPLETE Step 21 prediction pipeline
    #
    # predictor.py handles:
    #
    # raw input
    #     ↓
    # feature engineering
    #     ↓
    # 54 features
    #     ↓
    # preprocessor
    #     ↓
    # 10,334 features
    #     ↓
    # XGBoost
    #     ↓
    # predicted price
    # -------------------------------------------------------------------------

    predicted_price = predict_car_price(car_data)

    # -------------------------------------------------------------------------
    # Prevent impossible negative prices
    # -------------------------------------------------------------------------

    predicted_price = max(0.0, float(predicted_price))

    # -------------------------------------------------------------------------
    # Return result to website
    # -------------------------------------------------------------------------

    return {
        "success": True,
        "predicted_price": round(predicted_price, 2),
        "currency": "NGN"
    }


# =============================================================================
# RUN SERVER DIRECTLY
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )