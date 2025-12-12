from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostRegressor
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", 1404))

# --------------------
# Load model & metadata
# --------------------
model = CatBoostRegressor()
model.load_model("car_price_model.cbm")

with open("model_info.json", "r", encoding="utf-8") as f:
    model_info = json.load(f)

FEATURE_COLUMNS = model_info["feature_columns"]
CATEGORICAL_FEATURES = model_info["categorical_features"]

# --------------------
# Feature preparation
# --------------------
def prepare_features(data):
    year = int(data["year"])
    kilometer = int(data["kilometer"])

    car_age = max(0, CURRENT_YEAR - year)

    km_per_year = data.get("km_per_year")
    if not km_per_year or km_per_year <= 0:
        km_per_year = kilometer / car_age if car_age > 0 else kilometer

    km_per_month = kilometer / (car_age * 12 + 1)

    features = {
        "car_age": car_age,
        "kilometer": kilometer,
        "km_per_year": km_per_year,
        "km_per_month": km_per_month,
        "zero_km": 1 if kilometer == 0 else 0,
        "is_automatic": 1 if data["gearbox"] == "اتوماتیک" else 0,
        "car_name": data["car_name"],
        "color": data.get("color", "سفید"),
        "gearbox": data["gearbox"],
        "fuel": data["fuel"],
        "body_status": data["body_status"],
        "model": data["model"]
    }

    df = pd.DataFrame([features])

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return df[FEATURE_COLUMNS]

# --------------------
# Routes
# --------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "time": datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required = [
            "car_name", "year", "kilometer",
            "gearbox", "fuel", "body_status", "model"
        ]

        for field in required:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing field: {field}"
                }), 400

        df = prepare_features(data)
        price = float(model.predict(df)[0])

        return jsonify({
            "success": True,
            "predicted_price": price,
            "formatted_price": f"{price:,.0f}",
            "currency": "تومان",
            "confidence": {
                "min": round(price * 0.9),
                "max": round(price * 1.1)
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
