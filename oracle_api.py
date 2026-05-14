"""
NWO Oracle API — Render deployment (v2)

What's new in v2
- Real BTC/ETH price feed from Coinbase Exchange public API (no auth)
- Pluggable TimesFM/Kronos inference: set HF_TIMESFM_URL / HF_KRONOS_URL to
  point at a HuggingFace Inference Endpoint and the API will call it.
  Falls back to fast statistical stubs if not configured.
- BTC added to operator tokens by default
- /api/operator_status returns the fields the admin page expects

Endpoints
  GET  /health                                 → liveness
  GET  /                                       → endpoint index
  GET  /api/stats                              → platform stats
  GET  /api/history/<token>                    → 200 candles (Coinbase for BTC/ETH, synth otherwise)
  GET  /api/bets/live                          → sample live bets
  POST /api/predict                            → consensus prediction (3 models)
  GET  /api/predictions/<prediction_id>        → fetch a previous prediction
  GET  /api/active_prediction/<tok>/<tf>       → current open window for (token, timeframe)
  GET  /api/operator_status                    → operator scheduler state

Env vars
  PORT                  port to bind (Render sets this)
  OPERATOR_TOKENS       comma-list, default "ETH,BTC,STATE"
  OPERATOR_TIMEFRAMES   comma-list of minutes, default "5,15,30,60"
  WINDOW_DURATION_SEC   default 399
  MATCHING_WINDOW_SEC   default 300
  HF_TIMESFM_URL        optional: HF Inference Endpoint URL for TimesFM
  HF_KRONOS_URL         optional: HF Inference Endpoint URL for Kronos
  HF_TOKEN              required if either HF_*_URL is set
  ALCHEMY_BASE_RPC      optional: Alchemy/Infura RPC for on-chain reads
  OPERATOR_PRIVATE_KEY  required for operator to publish & settle
  CONTRACT_ADDRESS      default 0x16F2E8003877A8Bdb06dB01c30C7eD236285a035
"""
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from nwo_operator import get_operator
except Exception:
    get_operator = None  # operator is optional during dev

log = logging.getLogger("nwo.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Track API start time for uptime reporting
APP_STARTED_AT = time.time()

# ============================================================================
# Coinbase Exchange feed — real BTC/ETH prices, no auth required
# ============================================================================
COINBASE_API = "https://api.exchange.coinbase.com"
COINBASE_PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}

def coinbase_candles(token: str, granularity: int = 60, limit: int = 200):
    """Return list of {time, open, high, low, close, volume} oldest → newest.
    Returns None if token isn't supported by Coinbase or the request fails.
    """
    product = COINBASE_PRODUCTS.get(token.upper())
    if not product:
        return None
    try:
        r = requests.get(
            f"{COINBASE_API}/products/{product}/candles",
            params={"granularity": granularity},
            timeout=5,
        )
        if not r.ok:
            log.warning("coinbase candles %s -> %s", product, r.status_code)
            return None
        raw = r.json()  # newest-first: [time, low, high, open, close, volume]
        candles = list(reversed(raw))[:limit]
        return [
            {
                "time": datetime.fromtimestamp(c[0], tz=timezone.utc).isoformat(),
                "low": float(c[1]), "high": float(c[2]),
                "open": float(c[3]), "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in candles
        ]
    except Exception as e:
        log.warning("coinbase candles error: %s", e)
        return None

def coinbase_ticker_price(token: str):
    product = COINBASE_PRODUCTS.get(token.upper())
    if not product:
        return None
    try:
        r = requests.get(f"{COINBASE_API}/products/{product}/ticker", timeout=4)
        if not r.ok:
            return None
        return float(r.json()["price"])
    except Exception:
        return None

# ============================================================================
# HuggingFace Inference Endpoints — set HF_*_URL env vars to enable real models
# ============================================================================
HF_TIMESFM_URL = os.environ.get("HF_TIMESFM_URL", "").strip()
HF_KRONOS_URL  = os.environ.get("HF_KRONOS_URL", "").strip()
HF_TOKEN       = os.environ.get("HF_TOKEN", "").strip()

def _hf_post(url: str, payload: dict, timeout: int = 20):
    """POST to a HF Inference Endpoint. Returns parsed JSON or None on failure."""
    if not url or not HF_TOKEN:
        return None
    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if r.status_code == 503:
            log.info("HF endpoint loading (503), falling back: %s", url)
            return None
        if not r.ok:
            log.warning("HF endpoint %s -> %s: %s", url, r.status_code, r.text[:200])
            return None
        return r.json()
    except Exception as e:
        log.warning("HF endpoint error: %s", e)
        return None

# ============================================================================
# Predictors
# Each .predict() returns: {target_price, direction, confidence, model, model_kind}
# model_kind is "hf" when backed by HF Inference, "stat" when local statistical.
# ============================================================================
class TimesFMPredictor:
    """Google TimesFM via HF Inference Endpoint, with statistical fallback."""

    def _predict_hf(self, price_history, horizon):
        result = _hf_post(HF_TIMESFM_URL, {
            "inputs": list(map(float, price_history[-128:])),
            "parameters": {"horizon": int(horizon)},
        })
        if not result:
            return None
        # TimesFM endpoints typically return either a list of floats or
        # {"predictions": [...], "quantiles": [...]}. Be tolerant.
        forecast = None
        if isinstance(result, list):
            forecast = result
        elif isinstance(result, dict):
            forecast = result.get("predictions") or result.get("forecast") or result.get("mean")
        if not forecast or not isinstance(forecast, list):
            return None
        target = float(forecast[-1])
        current = float(price_history[-1])
        change = target - current
        # Confidence: higher when forecast is monotonic and clearly directional
        diffs = np.diff(forecast)
        directional = float(np.mean(np.sign(diffs) == np.sign(change)) * 100.0) if len(diffs) else 60.0
        return {
            "target_price": target,
            "direction": "up" if change > 0 else "down",
            "confidence": float(max(55.0, min(95.0, directional))),
            "model": "TimesFM",
            "model_kind": "hf",
        }

    def _predict_stat(self, price_history, horizon):
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
            "model_kind": "stat",
        }

    def predict(self, price_history, horizon=20):
        if HF_TIMESFM_URL and HF_TOKEN:
            hf = self._predict_hf(price_history, horizon)
            if hf:
                return hf
        return self._predict_stat(price_history, horizon)

class EMLPredictor:
    """Evolutionary ML — momentum + volatility. Always statistical (and fast)."""

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
            "model_kind": "stat",
        }

class KronosPredictor:
    """NeoQuasar Kronos candlestick FM via HF Inference, with statistical fallback."""

    def _predict_hf(self, ohlcv, horizon):
        # Kronos expects OHLCV history. Different deployments expect different
        # shapes — most accept either a list-of-dicts or columnar arrays.
        payload = {
            "inputs": {
                "open":   [c["open"] for c in ohlcv[-128:]],
                "high":   [c["high"] for c in ohlcv[-128:]],
                "low":    [c["low"]  for c in ohlcv[-128:]],
                "close":  [c["close"] for c in ohlcv[-128:]],
                "volume": [c["volume"] for c in ohlcv[-128:]],
            },
            "parameters": {"horizon": int(horizon)},
        }
        result = _hf_post(HF_KRONOS_URL, payload)
        if not result:
            return None
        forecast = None
        if isinstance(result, dict):
            forecast = result.get("close") or result.get("predictions") or result.get("forecast")
        elif isinstance(result, list):
            forecast = result
        if not forecast or not isinstance(forecast, list):
            return None
        target = float(forecast[-1])
        current = float(ohlcv[-1]["close"])
        return {
            "target_price": target,
            "direction": "up" if target > current else "down",
            "confidence": float(75.0 + np.random.random() * 15.0),
            "model": "Kronos",
            "model_kind": "hf",
        }

    def _predict_stat(self, ohlcv, horizon):
        if len(ohlcv) < 50:
            return None
        closes = np.array([d["close"] for d in ohlcv[-50:]], dtype=float)
        vols = np.array([d["volume"] for d in ohlcv[-50:]], dtype=float)
        diffs = np.diff(closes[-20:])
        weights = vols[-20:][1:]
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
            "model_kind": "stat",
        }

    def predict(self, ohlcv, horizon=20):
        if HF_KRONOS_URL and HF_TOKEN:
            hf = self._predict_hf(ohlcv, horizon)
            if hf:
                return hf
        return self._predict_stat(ohlcv, horizon)


timesfm = TimesFMPredictor()
eml = EMLPredictor()
kronos = KronosPredictor()
active_predictions = {}

# ============================================================================
# Helpers
# ============================================================================
def _base_price_for(token: str) -> float:
    """Live price from Coinbase if available, else a sensible synthetic base."""
    live = coinbase_ticker_price(token)
    if live is not None:
        return live
    return {"ETH": 3400.0, "BTC": 67000.0, "STATE": 0.0042, "LINK": 18.45}.get(
        token.upper(), 100.0
    )

def _synth_history(token: str, n: int = 200):
    base = _base_price_for(token)
    rng = np.random.default_rng()
    drift = rng.normal(0, base * 0.001, n)
    prices = base + np.cumsum(drift)
    return prices.tolist()

def _ohlcv_for(token: str):
    """Prefer real Coinbase OHLCV; fall back to synthetic."""
    candles = coinbase_candles(token, granularity=60, limit=200)
    if candles:
        return candles
    history = _synth_history(token, 200)
    return [{"open": p, "high": p, "low": p, "close": p, "volume": 1_000_000.0} for p in history]

# ============================================================================
# Routes
# ============================================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "nwo-oracle-api",
        "version": "2.0",
        "time": datetime.utcnow().isoformat(),
        "uptime_sec": int(time.time() - APP_STARTED_AT),
        "models": {
            "timesfm_hf": bool(HF_TIMESFM_URL and HF_TOKEN),
            "kronos_hf":  bool(HF_KRONOS_URL  and HF_TOKEN),
        },
        "coinbase_feed": list(COINBASE_PRODUCTS.keys()),
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "NWO Oracle API",
        "version": "2.0",
        "endpoints": [
            "GET /health",
            "GET /api/stats",
            "GET /api/history/<token>",
            "GET /api/bets/live",
            "POST /api/predict",
            "GET /api/predictions/<id>",
            "GET /api/active_prediction/<token>/<timeframe>",
            "GET /api/operator_status",
        ],
    })

@app.route("/api/predict", methods=["GET", "POST"])
def get_prediction():
    # Support both POST (JSON body) and GET (query params).
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
    else:
        data = request.args.to_dict()
    token = (data.get("token") or "ETH").upper()
    # Frontend may pass `timeframe` (minutes); legacy callers pass `horizon` (candles).
    # Treat 1 minute = 1 candle of horizon (60s OHLCV cadence).
    if "horizon" in data and data["horizon"] is not None:
        horizon = int(data["horizon"])
    elif "timeframe" in data and data["timeframe"] is not None:
        horizon = int(data["timeframe"])
    else:
        horizon = 20

    # Use real Coinbase OHLCV when available, otherwise synthetic
    ohlcv = _ohlcv_for(token)
    price_history = [c["close"] for c in ohlcv]

    timesfm_pred = timesfm.predict(price_history, horizon=horizon)
    eml_pred     = eml.predict(price_history, horizon=horizon)
    kronos_pred  = kronos.predict(ohlcv, horizon=horizon)

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
        "window_duration": 399,
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
        "feed": "coinbase" if token in COINBASE_PRODUCTS else "synthetic",
        "timestamp": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(seconds=399)).isoformat(),
    }
    active_predictions[prediction_id] = result
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
    candles = coinbase_candles(token, granularity=60, limit=200)
    if candles:
        return jsonify({"token": token, "data": candles, "source": "coinbase"})

    # synthetic fallback (e.g. STATE)
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
            "open": open_p, "high": high, "low": low, "close": close,
            "volume": float(1_000_000 + rng.integers(-100_000, 100_000)),
        })
        price = close
    return jsonify({"token": token, "data": history, "source": "synthetic"})

@app.route("/api/bets/live", methods=["GET"])
def get_live_bets():
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
    # Pull real numbers from the operator if it's running.
    # If the operator isn't wired up, return zeros — never fake numbers.
    stats = {
        "active_bets": 0,
        "volume_24h_eth": 0.0,
        "settled_today": 0,
        "accuracy": None,
        "total_bets": 0,
    }
    try:
        if get_operator is not None:
            op = get_operator()
            real = getattr(op, "get_stats", None)
            if callable(real):
                live = real() or {}
                stats.update({k: v for k, v in live.items() if k in stats})
    except Exception:
        pass
    return jsonify(stats)

# -- Operator endpoints ------------------------------------------------------
def _operator_tokens():
    raw = os.environ.get("OPERATOR_TOKENS", "ETH,BTC,STATE")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def _operator_timeframes():
    raw = os.environ.get("OPERATOR_TIMEFRAMES", "5,15,30,60")
    return [int(t.strip()) for t in raw.split(",") if t.strip().isdigit()]

@app.route("/api/active_prediction/<token>/<int:timeframe>", methods=["GET"])
def active_prediction(token, timeframe):
    if get_operator is None:
        return jsonify({
            "active": False, "reason": "operator_unavailable",
            "token": token.upper(), "timeframe": timeframe,
        }), 404
    op = get_operator()
    p = op.get_active_prediction(token.upper(), int(timeframe))
    if not p:
        return jsonify({
            "active": False, "reason": "no_open_window",
            "token": token.upper(), "timeframe": timeframe,
        }), 404
    return jsonify({
        "active": True,
        "prediction_id": p["id"],
        "token": p["token"],
        "timeframe": p["timeframe"],
        "start_price": p["startPrice"],
        "start_time": p["startTime"],
        "matching_close_at": p["matchingCloseAt"],
        "end_time": p["endTime"],
    })

@app.route("/api/operator_status", methods=["GET"])
def operator_status():
    base = {
        "running": False,
        "tokens": _operator_tokens(),
        "timeframes": _operator_timeframes(),
        "window_duration_sec": int(os.environ.get("WINDOW_DURATION_SEC", "399")),
        "matching_window_sec": int(os.environ.get("MATCHING_WINDOW_SEC", "300")),
        "uptime": _humanize_secs(int(time.time() - APP_STARTED_AT)),
        "uptime_sec": int(time.time() - APP_STARTED_AT),
        "models": {
            "timesfm_hf": bool(HF_TIMESFM_URL and HF_TOKEN),
            "kronos_hf":  bool(HF_KRONOS_URL  and HF_TOKEN),
        },
    }
    if get_operator is None:
        return jsonify(base)
    try:
        op = get_operator()
        st = op.status() if op else {}
        base.update(st)
        base["running"] = bool(st.get("running", True))
    except Exception as e:
        log.exception("operator status failed: %s", e)
        base["error"] = str(e)
    return jsonify(base)

def _humanize_secs(s: int) -> str:
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24: return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"

# Start operator on import — best effort
if get_operator is not None:
    try:
        get_operator()
    except Exception as e:
        log.exception(f"Operator init failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
