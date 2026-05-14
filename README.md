# NWO Oracle — API

Backend service for the [NWO Oracle](https://huggingface.co/spaces/CPater/nwo-oracle)
P2P prediction-betting platform on Base Mainnet. Serves price history, AI consensus
predictions (TimesFM / EML / Kronos stubs), and live-bet samples to the static
frontend.

## Live deployment

- **Frontend (HF Space):** https://huggingface.co/spaces/CPater/nwo-oracle
- **Smart contract (Base Mainnet, verified):**
  [`0x16F2E8003877A8Bdb06dB01c30C7eD236285a035`](https://basescan.org/address/0x16F2E8003877A8Bdb06dB01c30C7eD236285a035)
- **API:** deployed on Render (URL set in the frontend `CONFIG` block)

## Endpoints

| Method | Path                              | Purpose                                   |
| ------ | --------------------------------- | ----------------------------------------- |
| GET    | `/health`                         | Liveness probe — returns `{status: "ok"}` |
| GET    | `/`                               | List of available endpoints               |
| GET    | `/api/stats`                      | Platform stats (volume, bets, accuracy)   |
| GET    | `/api/history/<token>`            | 200 candles of OHLCV (synthesised)        |
| GET    | `/api/bets/live`                  | Sample live bets (demo)                   |
| POST   | `/api/predict`                    | Consensus prediction from 3 models        |
| GET    | `/api/predictions/<prediction_id>`| Fetch a previously generated prediction   |

`/api/predict` request body:

```json
{ "token": "ETH", "horizon": 20, "price_history": [3247.5, 3250.1, ...] }
```

All fields optional. If `price_history` is omitted, the API synthesises a series
for the requested token; replace with real exchange data in production.

## Local development

```bash
git clone https://github.com/RedCiprianPater/nwo-oracle.git
cd nwo-oracle
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python oracle_api.py          # http://localhost:5000
curl http://localhost:5000/health
```

## Deploy to Render

1. Push this repo to GitHub.
2. On Render: **New +** → **Web Service** → connect this repo.
3. Settings:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: leave blank (the `Procfile` handles it)
   - Instance type: **Free** (cold-starts after 15 min idle) or **Starter** ($7/mo, always warm)
4. Deploy. Wait for the build to finish.
5. Test: `curl https://YOUR-SERVICE.onrender.com/health` should return JSON.
6. Set the same URL in the HF Space's `index.html` `CONFIG.API_BASE` value
   and push to the space repo.

## Architecture

```
            ┌─────────────────────────────┐
   browser  │  HF Static Space (frontend) │
            │  • index.html               │
            │  • MetaMask + ethers.js     │
            └────────┬───────────┬────────┘
                     │           │
                     │           │ (read/write contract calls)
                     │           ▼
                     │  ┌─────────────────────┐
                     │  │  Base Mainnet       │
                     │  │  NWOOracleP2PBase   │
                     │  └─────────────────────┘
                     │
                     │ (fetch predictions, prices, stats)
                     ▼
            ┌──────────────────────────────┐
            │  Render (this service)       │
            │  • Flask + gunicorn          │
            │  • Predictor stubs           │
            └──────────────────────────────┘
```

The frontend talks to **both** the contract (for bets) and the API (for
predictions/charts) independently. If the API is unreachable the frontend
falls back to demo data and shows a "demo" badge — bets against the contract
still work.

## Swapping in real models

The `TimesFMPredictor` / `EMLPredictor` / `KronosPredictor` classes in
`oracle_api.py` are stubs that match the eventual model signatures. Replace
the `predict()` body in each with real inference and add the relevant deps to
`requirements.txt`. The output schema is fixed so the frontend doesn't need
changes:

```python
return {
    "target_price": float,      # in USD
    "direction": "up" | "down",
    "confidence": float,        # 0-100
    "model": str,
}
```

## Operator role (not yet implemented here)

The contract requires the owner to call `createPrediction(token, startPrice,
timeframe)` to open each betting window and `settlePrediction(predictionId,
endPrice)` after the window closes. This loop is **not yet** part of this
service — bets won't actually settle until you wire that up. Options:

- Add a scheduled job to this same Render service (APScheduler, every 5 min)
- Run a separate Cloudflare Worker / GitHub Action on a cron
- Run it locally and push the keys to a server later

If you want a starting point for that operator loop, ask and I'll add it.

## License

MIT
