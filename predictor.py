# =============================================================================
# predictor.py
# STEP 21 — CAR PRICE PREDICTION ENGINE
# =============================================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. MODEL PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "car_price_production_models"

MODEL_PATH = (
    MODEL_DIR /
    "step21_tuned_xgboost_model.joblib"
)

PREPROCESSOR_PATH = (
    MODEL_DIR /
    "step21_xgboost_preprocessor.joblib"
)


# =============================================================================
# 2. LOAD SAVED MODEL + PREPROCESSOR
# =============================================================================

print("=" * 80)
print("STEP 21 — LOADING CAR PRICE MODEL")
print("=" * 80)

model = joblib.load(MODEL_PATH)

print("✅ XGBoost model loaded")

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

print("✅ Preprocessor loaded")


# =============================================================================
# 3. GET EXACT FEATURES EXPECTED BY PREPROCESSOR
# =============================================================================

STEP21_FEATURES = list(
    preprocessor.feature_names_in_
)

print(
    f"✅ Expected input features: "
    f"{len(STEP21_FEATURES)}"
)

print("=" * 80)


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def create_step21_features(car):

    """
    Convert raw car information into the
    exact 54-feature structure expected
    by the Step 21 preprocessor.
    """

    df = car.copy()


    # =========================================================================
    # BASIC STRING COLUMNS
    # =========================================================================

    string_columns = [
        "color",
        "location",
        "description",
        "fuel_type",
        "body_type",
        "second_condition",
        "interior_color",
        "registered_car",
        "powertrain_type",
        "drivetrain",
        "make",
        "model",
        "trim",
        "condition",
        "transmission"
    ]


    for col in string_columns:

        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


    # =========================================================================
    # NUMERIC COLUMNS
    # =========================================================================

    numeric_columns = [
        "mileage",
        "engine_size",
        "cylinders",
        "seats",
        "year",
        "horsepower"
    ]


    for col in numeric_columns:

        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    # =========================================================================
    # MAKE CLEAN
    # =========================================================================

    df["make_clean"] = (
        df["make"]
        .str.strip()
        .str.title()
    )


    # =========================================================================
    # CONDITION CLEAN
    # =========================================================================

    df["condition_clean"] = (
        df["condition"]
        .str.strip()
        .str.title()
    )


    # =========================================================================
    # MODEL CLEAN
    # =========================================================================

    df["model_clean"] = (
        df["model"]
        .str.strip()
        .str.lower()
    )


    # =========================================================================
    # MAKE + MODEL
    # =========================================================================

    df["make_model"] = (
        df["make_clean"]
        + " "
        + df["model"].str.strip()
    )


    # =========================================================================
    # VEHICLE AGE
    # =========================================================================

    current_year = datetime.now().year

    df["vehicle_age"] = (
        current_year - df["year"]
    )

    df["vehicle_age"] = (
        df["vehicle_age"]
        .clip(lower=0)
    )


    # =========================================================================
    # MILEAGE OUTLIER
    # =========================================================================

    df["mileage_outlier"] = (
        (df["mileage"] > 500000)
        |
        (df["mileage"] < 0)
    ).astype(int)


    # =========================================================================
    # MILEAGE PER YEAR
    # =========================================================================

    age_for_calculation = (
        df["vehicle_age"]
        .replace(
            0,
            np.nan
        )
    )


    df["mileage_per_year"] = (
        df["mileage"] /
        age_for_calculation
    )


    df["mileage_per_year"] = (
        df["mileage_per_year"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # =========================================================================
    # DESCRIPTION
    # =========================================================================

    description_lower = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.lower()
    )


    # =========================================================================
    # BRAND NEW DESCRIPTION
    # =========================================================================

    df["is_brand_new_description"] = (
        description_lower.str.contains(
            r"\bbrand new\b|\bnew car\b|\bnew\b",
            regex=True,
            na=False
        )
    ).astype(int)


    # =========================================================================
    # CLEAN DESCRIPTION
    # =========================================================================

    df["is_clean"] = (
        description_lower.str.contains(
            r"\bclean\b|\baccident free\b|\baccident-free\b",
            regex=True,
            na=False
        )
    ).astype(int)


    # =========================================================================
    # BULLETPROOF
    # =========================================================================

    df["is_bulletproof"] = (
        description_lower.str.contains(
            "bulletproof",
            na=False
        )
    ).astype(int)


    # =========================================================================
    # ACCIDENT FREE
    # =========================================================================

    df["is_accident_free"] = (
        description_lower.str.contains(
            r"accident free|accident-free",
            regex=True,
            na=False
        )
    ).astype(int)


    # =========================================================================
    # DUTY PAID
    # =========================================================================

    df["is_duty_paid"] = (
        description_lower.str.contains(
            r"duty paid|duty-paid",
            regex=True,
            na=False
        )
    ).astype(int)


    # =========================================================================
    # BRAND CLASSIFICATIONS
    # =========================================================================

    luxury_brands = {
        "Rolls-Royce",
        "Bentley",
        "Lamborghini",
        "Ferrari",
        "Aston Martin",
        "McLaren",
        "Maybach",
        "Bugatti",
        "Koenigsegg"
    }


    premium_brands = {
        "Mercedes-Benz",
        "BMW",
        "Lexus",
        "Audi",
        "Land Rover",
        "Range Rover",
        "Jaguar",
        "Volvo",
        "Cadillac",
        "Infiniti",
        "Acura",
        "Genesis",
        "Porsche"
    }


    # =========================================================================
    # LUXURY BRAND
    # =========================================================================

    df["is_luxury_brand"] = (
        df["make_clean"]
        .isin(luxury_brands)
    ).astype(int)


    df["luxury_brand"] = (
        df["is_luxury_brand"]
    )


    # =========================================================================
    # PREMIUM BRAND
    # =========================================================================

    df["is_premium_brand"] = (
        df["make_clean"]
        .isin(premium_brands)
    ).astype(int)


    df["premium_brand"] = (
        df["is_premium_brand"]
    )


    # =========================================================================
    # HIGH-END MODELS
    # =========================================================================

    high_end_models = {
        "G-Class",
        "S-Class",
        "GLS-Class",
        "Maybach",
        "Range Rover",
        "Range Rover Vogue",
        "Land Cruiser",
        "Land Cruiser Prado",
        "LX",
        "LX 570",
        "LX 600",
        "LC",
        "LC 500",
        "Escalade",
        "Cayenne",
        "Panamera",
        "911"
    }


    df["is_high_end_model"] = (
        df["model"]
        .isin(high_end_models)
    ).astype(int)


    # =========================================================================
    # 2026 VEHICLE
    # =========================================================================

    df["flag_2026_vehicle"] = (
        df["year"] == 2026
    ).astype(int)


    # =========================================================================
    # CONDITION FLAGS
    # =========================================================================

    condition_lower = (
        df["condition_clean"]
        .str.lower()
    )


    df["is_foreign_used"] = (
        condition_lower ==
        "foreign used"
    ).astype(int)


    df["is_brand_new"] = (
        condition_lower ==
        "brand new"
    ).astype(int)


    df["is_local_used"] = (
        condition_lower ==
        "local used"
    ).astype(int)


    # =========================================================================
    # DATA AVAILABILITY FLAGS
    # =========================================================================

    df["has_mileage"] = (
        df["mileage"].notna()
    ).astype(int)


    df["has_engine_size"] = (
        df["engine_size"].notna()
    ).astype(int)


    df["has_horsepower"] = (
        df["horsepower"].notna()
    ).astype(int)


    # =========================================================================
    # ENGINE POWER PER LITRE
    # =========================================================================

    engine_litres = (
        df["engine_size"] /
        1000
    )


    df["engine_power_per_litre"] = np.where(
        engine_litres > 0,
        df["horsepower"] /
        engine_litres,
        np.nan
    )


    # =========================================================================
    # PRICE-DEPENDENT FEATURES
    #
    # A new car does not have a known selling price.
    #
    # Therefore we use neutral values.
    # =========================================================================

    df["flag_luxury_too_cheap"] = False

    df["flag_economy_too_expensive"] = False

    df["is_200m_plus"] = False

    df["is_500m_plus"] = False

    df["model_price_outlier"] = False

    df["price_audit_flag"] = "unknown"

    df["price_audit_status"] = "unknown"

    df["listing_validation_status"] = "unknown"


    # =========================================================================
    # CHECK ALL FEATURES
    # =========================================================================

    missing_features = [
        feature
        for feature in STEP21_FEATURES
        if feature not in df.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing Step 21 features:\n"
            +
            "\n".join(
                missing_features
            )
        )


    # =========================================================================
    # EXACT FEATURE ORDER
    # =========================================================================

    df = df[
        STEP21_FEATURES
    ]


    return df


# =============================================================================
# 5. PREDICT CAR PRICE
# =============================================================================

def predict_car_price(car_data):

    """
    Main prediction function.

    Input:
        Dictionary containing raw car information.

    Output:
        Predicted price as a float.
    """

    # -------------------------------------------------------------------------
    # Convert dictionary to DataFrame
    # -------------------------------------------------------------------------

    if isinstance(
        car_data,
        dict
    ):

        car_data = pd.DataFrame(
            [car_data]
        )


    # -------------------------------------------------------------------------
    # Create Step 21 features
    # -------------------------------------------------------------------------

    X = create_step21_features(
        car_data
    )


    print(
        f"Features created: {X.shape}"
    )


    # -------------------------------------------------------------------------
    # Apply saved preprocessor
    # -------------------------------------------------------------------------

    X_processed = (
        preprocessor.transform(X)
    )


    print(
        f"Processed features: "
        f"{X_processed.shape}"
    )


    # -------------------------------------------------------------------------
    # XGBoost prediction
    # -------------------------------------------------------------------------

    prediction = model.predict(
        X_processed
    )


    predicted_price = float(
        prediction[0]
    )


    return predicted_price


# =============================================================================
# 6. OPTIONAL TEST
# =============================================================================

if __name__ == "__main__":

    test_car = {

        "color": "Gray",

        "location": "Lagos, Egbe/Idimu",

        "description": (
            "2025 Toyota Camry XSE Hybrid "
            "- engine 2.5L hybrid "
            "- FWD "
            "- fuel efficient"
        ),

        "fuel_type": "Unknown",

        "mileage": 89232.5,

        "body_type": "Unknown",

        "engine_size": 3300,

        "second_condition": "Unknown",

        "interior_color": "Gray",

        "registered_car": "No",

        "powertrain_type": "Unknown",

        "drivetrain": "Unknown",

        "cylinders": 6,

        "seats": 5,

        "make": "Toyota",

        "model": "Camry",

        "year": 2025,

        "trim": "XSE",

        "condition": "Foreign Used",

        "transmission": "Automatic",

        "horsepower": 268
    }


    print("\n")
    print("=" * 80)
    print("STEP 21 TEST PREDICTION")
    print("=" * 80)


    result = predict_car_price(
        test_car
    )


    print(
        f"\nPredicted price: "
        f"₦{result:,.2f}"
    )

    print("=" * 80)