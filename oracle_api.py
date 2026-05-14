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
    """Read real bets from the deployed contract.

    The contract exposes:
      - getPendingBets() → uint256[] of bet IDs in OPEN/MATCHED status
      - getBet(uint256)  → full Bet struct
      - betCounter       → next bet ID (for fallback iteration)

    We read pending bets first (fast path), then optionally include recently-
    settled bets for the "Live" tab. The frontend separates these into
    Open / Pending / Live tabs by status.
    """
    bets = []
    if get_operator is None:
        return jsonify({"bets": [], "reason": "operator_module_not_loaded"})

    try:
        op = get_operator()
        if not op or not op.contract:
            return jsonify({"bets": [], "reason": "contract_not_initialized"})

        # The contract address has more functions than the operator's narrow
        # ABI. We need to attach a wider ABI just for reads.
        full_abi = [
            {"inputs":[],"name":"getPendingBets","outputs":[{"type":"uint256[]"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"type":"uint256"}],"name":"getBet","outputs":[{"type":"tuple","components":[
                {"name":"creator","type":"address"},
                {"name":"opponent","type":"address"},
                {"name":"predictionId","type":"bytes32"},
                {"name":"creatorDirection","type":"bool"},
                {"name":"amount","type":"uint256"},
                {"name":"matchedAmount","type":"uint256"},
                {"name":"timeframe","type":"uint256"},
                {"name":"createdAt","type":"uint256"},
                {"name":"expiresAt","type":"uint256"},
                {"name":"matchedAt","type":"uint256"},
                {"name":"settlementTime","type":"uint256"},
                {"name":"status","type":"uint8"},
                {"name":"creatorWon","type":"bool"},
            ]}],"stateMutability":"view","type":"function"},
            {"inputs":[],"name":"betCounter","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"type":"bytes32"}],"name":"getPrediction","outputs":[{"type":"tuple","components":[
                {"name":"id","type":"bytes32"},
                {"name":"token","type":"string"},
                {"name":"startPrice","type":"uint256"},
                {"name":"endPrice","type":"uint256"},
                {"name":"timeframe","type":"uint256"},
                {"name":"startTime","type":"uint256"},
                {"name":"endTime","type":"uint256"},
                {"name":"settled","type":"bool"},
                {"name":"finalDirection","type":"bool"},
            ]}],"stateMutability":"view","type":"function"},
        ]
        read_contract = op.w3.eth.contract(address=op.contract.address, abi=full_abi)

        # 1) Pending bets (Open + Pending tabs)
        pending_ids = read_contract.functions.getPendingBets().call()

        # 2) Also include the last ~10 settled bets for the Live/Settled tab.
        try:
            counter = read_contract.functions.betCounter().call()
        except Exception:
            counter = max(pending_ids) if pending_ids else 0

        # Walk back from the counter, dedup with pending_ids, collect up to 10
        # additional bets that are not pending (i.e. matched/settled/cancelled).
        extra_ids = []
        pending_set = set(pending_ids)
        for i in range(counter, max(0, counter - 30), -1):
            if i in pending_set:
                continue
            extra_ids.append(i)
            if len(extra_ids) >= 10:
                break

        all_ids = list(pending_ids) + extra_ids
        # Map: status enum int -> status string
        # Contract enum: Pending(0), Matched(1), Settled(2), Cancelled(3), Expired(4)
        STATUS_MAP = {0: "open", 1: "pending", 2: "live", 3: "cancelled", 4: "expired"}

        # Cache predictions so we don't re-fetch the same one for every bet
        pred_cache = {}

        for bid in all_ids:
            try:
                b = read_contract.functions.getBet(bid).call()
            except Exception as e:
                log.warning(f"getBet({bid}) failed: {e}")
                continue
            (creator, opponent, prediction_id, creator_dir, amount, matched_amount,
             timeframe, created_at, expires_at, matched_at, settlement_time,
             status_int, creator_won) = b
            if status_int == 0 and amount == 0:
                continue  # uninitialized slot

            # Map status. The contract's enum 0=Pending in the bet-creator sense
            # means "still seeking a counterparty" → frontend "open" tab.
            # Once matched, it's "pending" (awaiting settlement).
            # Once settled it goes into the "live" tab as a recent settled bet.
            status_str = STATUS_MAP.get(int(status_int), "open")
            # Refine: if pending but matched_amount > 0, it's actually matched
            if status_int == 1:
                status_str = "pending"
            # If status_int == 2 (Settled) we surface as "live" (recently active)
            if status_int == 2:
                status_str = "live"

            # Fetch the prediction to get the token + start price
            token = "?"
            start_price = 0
            try:
                pred_key = prediction_id.hex() if hasattr(prediction_id, "hex") else prediction_id
                if pred_key in pred_cache:
                    pred = pred_cache[pred_key]
                else:
                    pred = read_contract.functions.getPrediction(prediction_id).call()
                    pred_cache[pred_key] = pred
                token = pred[1]
                start_price = pred[2]
            except Exception as e:
                log.warning(f"getPrediction failed for bet {bid}: {e}")

            bets.append({
                "id": int(bid),
                "betId": int(bid),
                "player": creator,
                "address": creator,
                "opponent": opponent if opponent and opponent != "0x0000000000000000000000000000000000000000" else None,
                "matchedTo": opponent if opponent and opponent != "0x0000000000000000000000000000000000000000" else None,
                "predictionId": "0x" + (prediction_id.hex() if hasattr(prediction_id, "hex") else prediction_id),
                "direction": "up" if creator_dir else "down",
                "isUp": bool(creator_dir),
                "token": token,
                "timeframe": int(timeframe),
                "timeframeMinutes": int(timeframe),
                "stake": float(amount) / 1e18,
                "stakeWei": str(int(amount)),
                "matchedAmount": float(matched_amount) / 1e18,
                "startPrice": float(start_price) / 1e8 if start_price else None,
                "createdAt": int(created_at),
                "expiresAt": int(expires_at),
                "matchedAt": int(matched_at) if matched_at else None,
                "settlementTime": int(settlement_time) if settlement_time else None,
                "timeRemaining": max(0, int(settlement_time) - int(time.time())) if settlement_time else None,
                "status": status_str,
                "creatorWon": bool(creator_won),
            })

        # Most recent first
        bets.sort(key=lambda b: b["createdAt"], reverse=True)
        return jsonify({"bets": bets, "count": len(bets)})

    except Exception as e:
        log.exception(f"/api/bets/live failed: {e}")
        return jsonify({"bets": [], "error": str(e)}), 200

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
    # We return 200 with {active:false, ...} for "no open window" rather than
    # 404. 404 should mean "endpoint doesn't exist" — and clients can't tell
    # those apart. The frontend gates the Place Bet button on `active`.
    base = {
        "token": token.upper(),
        "timeframe": int(timeframe),
        "active": False,
        "open": False,
    }
    if get_operator is None:
        return jsonify({**base, "reason": "operator_unavailable"}), 200

    try:
        op = get_operator()
    except Exception as e:
        return jsonify({**base, "reason": f"operator_error:{e}"}), 200

    p = op.get_active_prediction(token.upper(), int(timeframe))
    if not p:
        return jsonify({**base, "reason": "no_open_window"}), 200

    # `p` is the operator's tracked dict, shape:
    #   {id, endTime, startPrice, settled, startTime, matchingCloseAt, timeframe, token}
    # Translate to the shape the frontend expects.
    now = int(time.time())
    matching_remaining = max(0, int(p.get("matchingCloseAt", 0)) - now)
    settle_remaining   = max(0, int(p.get("endTime", 0))         - now)

    return jsonify({
        **base,
        "active": True,
        "open": matching_remaining > 0,
        "prediction_id": p.get("id"),
        "start_price": p.get("startPrice"),
        "start_time": p.get("startTime"),
        "end_time": p.get("endTime"),
        "matching_close_at": p.get("matchingCloseAt"),
        "matching_window_remaining": matching_remaining,
        "settle_remaining": settle_remaining,
    }), 200

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
        return jsonify({**base, "reason": "operator_module_not_imported"})
    try:
        op = get_operator()
        st = op.status() if op else {}
        # Your operator.status() returns {enabled, contract, operator_address,
        # tokens, active, tx_count, last_open_tx, last_settle_tx, last_error}.
        # `enabled=False` means dry-run — operator is alive but won't send tx.
        # The scheduler is "running" if the operator object exists; "enabled"
        # is what actually matters for windows to appear on-chain.
        base["running"] = True  # the python process / scheduler is alive
        base["enabled"] = bool(st.get("enabled", False))
        base["mode"] = "live" if st.get("enabled") else "dry_run"
        base["operator_address"] = st.get("operator_address")
        base["contract"] = st.get("contract")
        base["tokens"] = st.get("tokens") or base["tokens"]
        base["tx_count"] = st.get("tx_count")
        base["last_open_tx"] = st.get("last_open_tx")
        base["last_settle_tx"] = st.get("last_settle_tx")
        base["last_error"] = st.get("last_error")
        base["active_windows"] = st.get("active")
        # Surface whether APScheduler thread is alive — gunicorn forks can
        # leave the scheduler dead in the worker serving this request.
        sched = getattr(op, "scheduler", None)
        if sched is not None:
            try:
                base["scheduler_jobs"] = [
                    {"id": j.id, "next_run": str(j.next_run_time)} for j in sched.get_jobs()
                ]
                base["scheduler_running"] = sched.running
            except Exception:
                base["scheduler_running"] = False
    except Exception as e:
        log.exception("operator status failed: %s", e)
        base["error"] = str(e)
    return jsonify(base)


@app.route("/api/operator_tick_open", methods=["POST", "GET"])
def operator_tick_open():
    """Manually fire tick_open() once. Diagnostic for scheduler issues.
    Gated by ?key=… matching env var OPERATOR_ADMIN_KEY (if set)."""
    admin_key = os.environ.get("OPERATOR_ADMIN_KEY", "").strip()
    if admin_key:
        provided = request.args.get("key") or (request.get_json(silent=True) or {}).get("key", "")
        if provided != admin_key:
            return jsonify({"error": "unauthorized"}), 401
    if get_operator is None:
        return jsonify({"error": "operator_module_not_imported"}), 500
    try:
        op = get_operator()
        before = dict(op.tx_count)
        op.tick_open()
        after = dict(op.tx_count)
        return jsonify({
            "ok": True,
            "tx_count_before": before,
            "tx_count_after": after,
            "last_open_tx": op.last_open_tx,
            "last_error": op.last_error,
        })
    except Exception as e:
        log.exception("manual tick_open failed: %s", e)
        return jsonify({"error": str(e)}), 500

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
