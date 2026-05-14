# NWO Oracle — API + Operator

Backend service for [NWO Oracle](https://huggingface.co/spaces/CPater/nwo-oracle).
Two things in one Render service:

1. **HTTP API** (Flask) — price history, AI consensus predictions, live-bet samples
2. **Operator loop** (APScheduler) — keeps the on-chain prediction lifecycle running
   (`createPrediction` every 5 minutes, `settlePrediction` as windows close)

## Live deployment

- **Frontend (HF Space):** https://huggingface.co/spaces/CPater/nwo-oracle
- **Smart contract (verified, Base Mainnet):**
  [`0x16F2E8003877A8Bdb06dB01c30C7eD236285a035`](https://basescan.org/address/0x16F2E8003877A8Bdb06dB01c30C7eD236285a035)
- **API:** https://nwo-oracles.onrender.com

## HTTP endpoints

| Method | Path                                              | Purpose                                |
| ------ | ------------------------------------------------- | -------------------------------------- |
| GET    | `/health`                                         | Liveness probe                         |
| GET    | `/`                                               | List endpoints                         |
| GET    | `/api/stats`                                      | Platform stats (volume, bets)          |
| GET    | `/api/history/<token>`                            | 200 OHLCV candles                      |
| GET    | `/api/bets/live`                                  | Sample live bets                       |
| POST   | `/api/predict`                                    | Consensus from TimesFM / EML / Kronos  |
| GET    | `/api/predictions/<id>`                           | Re-fetch a previous prediction         |
| GET    | `/api/active_prediction/<token>/<timeframe>`      | Current open matching window (NEW)     |
| GET    | `/api/operator_status`                            | Operator scheduler state (NEW)         |

The frontend calls `/api/active_prediction/ETH/5` (or 15/30/60) right before
submitting a bet, so the bet targets a real on-chain prediction window.

## Operator: what it does

The smart contract requires the owner to:

1. Call `createPrediction(token, startPrice, timeframe)` to open a 5-min matching
   window, after which bets settle at `startTime + timeframe minutes`.
2. Call `settlePrediction(predictionId, endPrice)` after the window ends.

This service does both automatically. On a 5-minute cron, it checks each
`(token, timeframe)` pair — if there's no open matching window, it opens a new
one. Every minute, it settles any prediction whose window has ended (with a
30-second buffer).

Price source: **Coinbase spot prices** (`api.coinbase.com/v2/prices/<pair>/spot`).
Free, no auth, generous rate limit. Prices cached for 8 seconds.

State recovery: on startup, the operator scans the last ~4 hours of
`PredictionCreated` events from the contract and rebuilds its in-memory state.
This handles Render free-tier idle restarts gracefully.

## Environment variables

Set these in the Render dashboard under **Environment** → **Add Environment
Variable**. None of them go in git. The `.env.example` file documents them
without secrets.

| Variable                 | Required | Default                       | Description                          |
| ------------------------ | -------- | ----------------------------- | ------------------------------------ |
| `CONTRACT_ADDRESS`       | ✅       | —                             | `0x16F2E8003877A8Bdb06dB01c30C7eD236285a035` |
| `OWNER_PRIVATE_KEY`      | ✅       | —                             | Private key of contract owner wallet |
| `OPERATOR_ENABLED`       | ✅       | `false`                       | `true` to send tx; `false` = dry run |
| `RPC_URL`                |          | `https://mainnet.base.org`    | Base RPC endpoint                    |
| `OPERATOR_TOKENS`        |          | `ETH`                         | Comma-separated (`ETH,BTC`)          |
| `OPERATOR_GAS_LIMIT`     |          | `500000`                      | Per-tx gas limit                     |
| `OPERATOR_PRIORITY_GWEI` |          | `0.005`                       | Priority fee (gwei)                  |

**`OPERATOR_ENABLED=false` is the default** so the operator never accidentally
sends transactions on first deploy. Flip it to `true` only after confirming
`/api/operator_status` looks correct.

### Operator wallet hygiene

The operator wallet (`OWNER_PRIVATE_KEY`) signs `createPrediction` and
`settlePrediction` calls. It does **not** hold any user funds — those live in
the contract. But it does have admin rights (pause, change fee recipient,
emergency refund), so:

- **Use a dedicated wallet**, not your main one. Funding ~0.005 ETH on Base lasts
  many months at current gas.
- **Never commit the key**, never share it, never put it in this repo's git
  history. Set it as a Render environment variable.
- **For added safety**, transfer contract ownership to a Safe (multisig) once
  things are stable; the operator can still call `createPrediction`/
  `settlePrediction` if you make those non-owner-only later, or you keep them
  owner-only and route through the Safe.

## Deploy on Render

1. Push this repo to GitHub.
2. Render dashboard → **New +** → **Web Service** → connect this repo.
3. Settings:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: blank (the `Procfile` runs gunicorn with `--workers 1` —
     important so the scheduler doesn't double-fire).
   - Instance type: **Free** (cold-starts after 15min idle) or **Starter** for
     always-warm.
4. Add the env vars above. Set `OPERATOR_ENABLED=false` first.
5. Deploy. Tail the logs in the Render dashboard.
6. Once the service is up: `curl https://nwo-oracles.onrender.com/health`
7. Check the operator state: `curl https://nwo-oracles.onrender.com/api/operator_status`
   — should show `enabled: false`, `tokens: ["ETH"]`, empty `active`.
8. Flip `OPERATOR_ENABLED=true` in the Render env vars. Redeploy.
9. Within ~30 seconds the operator will open its first window. Verify with:
   ```
   curl https://nwo-oracles.onrender.com/api/active_prediction/ETH/5
   ```
   Should return `active: true` and a `prediction_id`.

## Gas estimate

Each cron tick opens up to 4 predictions (one per timeframe) and settles
whatever's due. Roughly **8 transactions per 5 minutes** when everything's
working, so ~100/hour. At Base gas (currently ~0.005 gwei), that's a fraction
of a cent per hour. **0.005 ETH funding lasts months.** Keep an eye on the
operator wallet's balance and top up when it drops below 0.001 ETH.

## Operator API

```bash
# Status: see what the operator is doing right now
curl https://nwo-oracles.onrender.com/api/operator_status | jq

# Active prediction for ETH 5-min — frontend hits this before each bet
curl https://nwo-oracles.onrender.com/api/active_prediction/ETH/5 | jq

# When no window is open (e.g. just woke from Render idle)
# Returns HTTP 404 with {active: false, reason: "no_open_window"}
```

## Local development

```bash
git clone https://github.com/RedCiprianPater/nwo-oracle.git
cd nwo-oracle
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run in dry-run mode (no tx sent, but everything else works)
export CONTRACT_ADDRESS=0x16F2E8003877A8Bdb06dB01c30C7eD236285a035
export OPERATOR_ENABLED=false
python oracle_api.py     # http://localhost:5000
```

The dry-run mode prints what it *would* do without sending transactions —
useful for testing.

## Swapping in real models

`TimesFMPredictor` / `EMLPredictor` / `KronosPredictor` in `oracle_api.py` are
stubs that match the eventual model signatures. Replace each `predict()` body
with real inference, add deps to `requirements.txt`, and the frontend won't
need changes — the response schema is fixed.

## File map

```
oracle_api.py        Flask app, HTTP endpoints
nwo_operator.py      Scheduler, contract interaction, state recovery
requirements.txt     Python deps
Procfile             gunicorn config (workers=1 is intentional)
runtime.txt          Python version pin for Render
.env.example         Documents env vars (no secrets, safe to commit)
.gitignore           Standard Python ignores
```

## License

MIT
