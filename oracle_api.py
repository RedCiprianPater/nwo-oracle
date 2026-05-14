"""
NWO Oracle API - Render deployment

Endpoints:
  GET  /health                          -> liveness probe
  GET  /api/stats                       -> platform stats
  GET  /api/history/<token>             -> 200 candles of OHLCV (mocked)
  GET  /api/bets/live                   -> sample live bets
  POST /api/predict                     -> consensus prediction from 3 models
  GET  /api/predictions/<prediction_id> -> fetch a previous prediction

Replace the *Predictor classes with real model inference when ready.
"""

import os
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# CORS: allow your HF space + local dev. Tighten this in production.
ALLOWED_ORIGINS = [
    "https://cpater-nwo-oracle.hf.space",
    "https://huggingface.co",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS + ["*"]}})  # permissive for now


# ----------------------------- Predictors -----------------------------

class TimesFMPredictor:
    """Time Series Foundation Model stub. Swap in real TimesFM weights later."""
    def predict(self, price_history, horizon=20):
        if len(price_history) < 10:
            return None
        recent = np.array(price_history[-10:], dtype=float)
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        current = recent[-1]
        change = trend * horizon
        return {
            "target_price": float(current + change),
            "direction": "up" if change > 0 else "down",
            "confidence": float(min(95.0, 50.0 + abs(trend) * 100.0)),
            "model": "TimesFM",
        }


class EMLPredictor:
    """Evolutionary ML Trees stub."""
    def predict(self, price_history, horizon=20):
        if len(price_history) < 20:
            return None
        arr = np.array(price_history, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        vol = float(np.std(returns[-10:]))
        momentum = float(np.mean(returns[-5:]) - np.mean(returns[-10:-5]))
        current = float(arr[-1])
        change = momentum * current * horizon
        return {
            "target_price": current + change,
            "direction": "up" if change > 0 else "down",
            "confidence": float(min(90.0, 55.0 + vol * 1000.0)),
            "model": "EML",
        }


class KronosPredictor:
    """Kronos candlestick foundation-model stub."""
    def predict(self, ohlcv, horizon=20):
        if len(ohlcv) < 50:
            return None
        closes = np.array([d["close"] for d in ohlcv[-50:]], dtype=float)
        vols = np.array([d["volume"] for d in ohlcv[-50:]], dtype=float)
        diffs = np.diff(closes[-20:])
        weights = vols[-20:][1:]  # align to diffs
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        vw_trend = float(np.average(diffs, weights=weights))
        current = float(closes[-1])
        predicted = current + vw_trend * horizon
        return {
            "target_price": predicted,
            "direction": "up" if predicted > current else "down",
            "confidence": float(75.0 + np.random.random() * 15.0),
            "model": "Kronos",
        }


timesfm = TimesFMPredictor()
eml = EMLPredictor()
kronos = KronosPredictor()

# In-memory cache of recent predictions (cleared on restart - fine for v1)
active_predictions = {}


# ----------------------------- Helpers -----------------------------

def _base_price_for(token: str) -> float:
    return {"ETH": 3400.0, "BTC": 67000.0, "STATE": 0.0042, "LINK": 18.45}.get(
        token.upper(), 100.0
    )


def _synth_history(token: str, n: int = 200):
    base = _base_price_for(token)
    rng = np.random.default_rng()
    drift = rng.normal(0, base * 0.001, n)
    prices = base + np.cumsum(drift)
    return prices.tolist()


# ----------------------------- Routes ------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "nwo-oracle-api", "time": datetime.utcnow().isoformat()})


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "NWO Oracle API",
        "endpoints": [
            "/health",
            "/api/stats",
            "/api/history/<token>",
            "/api/bets/live",
            "/api/predict (POST)",
            "/api/predictions/<id>",
        ],
    })


@app.route("/api/predict", methods=["POST"])
def get_prediction():
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "ETH").upper()
    horizon = int(data.get("horizon", 20))
    price_history = data.get("price_history") or _synth_history(token, 200)

    timesfm_pred = timesfm.predict(price_history, horizon=horizon)
    eml_pred = eml.predict(price_history, horizon=horizon)
    ohlcv = [{"close": p, "volume": 1_000_000.0} for p in price_history]
    kronos_pred = kronos.predict(ohlcv, horizon=horizon)

    preds = [p for p in (timesfm_pred, eml_pred, kronos_pred) if p]
    if not preds:
        return jsonify({"error": "Not enough data"}), 400

    up_votes = sum(1 for p in preds if p["direction"] == "up")
    direction = "up" if up_votes >= 2 else "down"
    avg_conf = float(np.mean([p["confidence"] for p in preds]))
    avg_target = float(np.mean([p["target_price"] for p in preds]))

    prediction_id = f"{token}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    result = {
        "prediction_id": prediction_id,
        "token": token,
        "current_price": float(price_history[-1]),
        "window_duration": 399,  # seconds, ~6.66 min
        "predictions": {
            "timesfm": timesfm_pred,
            "eml": eml_pred,
            "kronos": kronos_pred,
        },
        "consensus": {
            "direction": direction,
            "target_price": avg_target,
            "confidence": avg_conf,
            "agreement": f"{up_votes}/3",
        },
        "timestamp": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(seconds=399)).isoformat(),
    }
    active_predictions[prediction_id] = result
    # Bound memory
    if len(active_predictions) > 500:
        oldest = sorted(active_predictions.keys())[:100]
        for k in oldest:
            active_predictions.pop(k, None)

    return jsonify(result)


@app.route("/api/predictions/<prediction_id>", methods=["GET"])
def get_prediction_status(prediction_id):
    pred = active_predictions.get(prediction_id)
    if not pred:
        return jsonify({"error": "Prediction not found"}), 404
    return jsonify(pred)


@app.route("/api/history/<token>", methods=["GET"])
def get_price_history(token):
    token = token.upper()
    base = _base_price_for(token)
    rng = np.random.default_rng()
    history = []
    now = datetime.utcnow()
    price = base
    for i in range(200):
        change = float(rng.normal(0, base * 0.005))
        open_p = price
        close = price + change
        high = max(open_p, close) + abs(rng.normal(0, base * 0.002))
        low = min(open_p, close) - abs(rng.normal(0, base * 0.002))
        history.append({
            "time": (now - timedelta(minutes=200 - i)).isoformat(),
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": float(1_000_000 + rng.integers(-100_000, 100_000)),
        })
        price = close
    return jsonify({"token": token, "data": history})


@app.route("/api/bets/live", methods=["GET"])
def get_live_bets():
    # Demo data; replace with on-chain reads from the contract later
    rng = np.random.default_rng()
    sample = []
    for i in range(1, 6):
        wallet_bytes = rng.bytes(20).hex()
        sample.append({
            "id": i,
            "wallet": f"0x{wallet_bytes[:6]}...{wallet_bytes[-4:]}",
            "direction": "up" if rng.random() > 0.5 else "down",
            "amount": round(float(rng.random()) * 2, 4),
            "token": "ETH",
            "status": "live",
            "time_remaining": f"{rng.integers(1, 6)}:{rng.integers(10, 59)}",
        })
    return jsonify(sample)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    # Demo data; aggregate from contract events in production
    return jsonify({
        "total_volume": 2_100_000,
        "total_bets": 2_456,
        "active_bets": 89,
        "winners_today": 156,
        "oracle_accuracy": 0.89,
        "avg_payout": 1.85,
    })


if __name__ == "__main__":
    # For local development only; Render uses gunicorn (see Procfile)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
