from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import os
import requests
import logging
import traceback

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# LOAD MODEL
# =========================
MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.h5")

try:
    from tensorflow.keras.models import load_model
    MODEL = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    logger.info(f"Model loaded: {MODEL is not None}")
except Exception as e:
    logger.warning(f"Model load failed: {e}")
    MODEL = None

CLASS_NAMES = [
    "Leaf Spot",
    "Rust",
    "Healthy",
    "Powdery Mildew"
]

# =========================
# PREDICTION
# =========================
def predict_image(file_stream):
    try:
        img = Image.open(file_stream).convert("RGB")
    except Exception:
        logger.exception("Image read failed")
        return None, 0.0

    if MODEL is not None:
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = MODEL.predict(arr)

        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds).item())  # ✅ FIX

        return CLASS_NAMES[idx], round(confidence, 3)

    # fallback (no model)
    arr = np.array(img).astype(np.float32)
    mean = arr.mean()
    idx = int((mean % 100) / 100 * len(CLASS_NAMES))
    idx = min(idx, len(CLASS_NAMES) - 1)

    confidence = 0.75
    return CLASS_NAMES[idx], confidence


# =========================
# WEATHER (NO API KEY)
# =========================
def get_weather(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=relativehumidity_2m,temperature_2m,precipitation"
            "&forecast_days=1&timezone=auto"
        )

        r = requests.get(url, timeout=10)
        data = r.json()

        h = data["hourly"]["relativehumidity_2m"]
        t = data["hourly"]["temperature_2m"]
        p = data["hourly"]["precipitation"]

        return {
            "avg_humidity": float(np.mean(h).item()),
            "avg_temp": float(np.mean(t).item()),
            "total_precip": float(np.sum(p).item())
        }

    except Exception:
        logger.exception("Weather fetch failed")
        return None


# =========================
# RISK CALCULATION
# =========================
def compute_risk(disease, weather):
    if weather is None:
        return {"risk_score": None, "risk_level": "Unknown"}

    h = weather["avg_humidity"]
    t = weather["avg_temp"]
    p = weather["total_precip"]

    score = 0

    if disease == "Powdery Mildew":
        score += max(0, (h - 60) * 0.6)
        if 10 <= t <= 25:
            score += 20

    elif disease == "Rust":
        score += max(0, (h - 65) * 0.5)
        score += min(p * 5, 25)

    elif disease == "Leaf Spot":
        score += max(0, (h - 70) * 0.4)
        score += min(p * 6, 25)

    else:
        score += max(0, (100 - h) * 0.2)

    score = float(max(0, min(100, score)))

    if score >= 60:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"

    return {
        "risk_score": round(score, 1),
        "risk_level": level
    }


# =========================
# ERROR HANDLER
# =========================
@app.errorhandler(Exception)
def handle_error(e):
    logger.error(traceback.format_exc())
    return jsonify({"error": str(e)}), 500


# =========================
# MAIN API
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    disease, confidence = predict_image(BytesIO(file.read()))

    weather = None
    if lat and lon:
        weather = get_weather(float(lat), float(lon))

    risk = compute_risk(disease, weather)

    response = {
        "disease": disease,
        "confidence": float(confidence),
        "weather": weather,
        "risk": risk,
        "model_loaded": bool(MODEL)
    }

    return jsonify(response)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(port=5000, debug=True)