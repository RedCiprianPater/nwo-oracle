"""
NWO Oracle - Operator loop

Maintains the prediction-window lifecycle on Base Mainnet:
  - Every 5 minutes, ensures each (token, timeframe) has a fresh open matching window.
  - Every minute, settles any prediction whose window has ended.
  - On startup, recovers in-memory state from on-chain PredictionCreated events.

Environment variables (set in Render dashboard, never commit):
  CONTRACT_ADDRESS       - 0x... of NWOOracleP2PBase                (required)
  RPC_URL                - Base RPC endpoint                         (default: mainnet.base.org)
  OWNER_PRIVATE_KEY      - The owner wallet's private key            (required to send tx)
  OPERATOR_ENABLED       - "true" to actually send tx                (default: "false")
  OPERATOR_TOKENS        - Comma-separated, e.g. "ETH" or "ETH,BTC"  (default: "ETH")
  OPERATOR_GAS_LIMIT     - Per-tx gas limit                          (default: 500000)
  OPERATOR_PRIORITY_GWEI - Priority fee in gwei                      (default: "0.005")

The operator wallet should hold a small amount of Base ETH for gas (~0.005 ETH lasts
months at current Base gas levels). DO NOT use a wallet with significant funds.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from eth_account import Account
from web3 import Web3
from web3.exceptions import TransactionNotFound

log = logging.getLogger("nwo.operator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ============================================================================
# Constants
# ============================================================================

TIMEFRAMES = (5, 15, 30, 60)
DEFAULT_TOKENS = ("ETH",)
PRICE_DECIMALS = 8
PRICE_CACHE_TTL = 8
SETTLEMENT_BUFFER = 30
MATCHING_WINDOW_SEC = 5 * 60
RECOVERY_BLOCK_LOOKBACK = 7200
BASE_CHAIN_ID = 8453

ABI = [
    {"type": "function", "stateMutability": "nonpayable", "name": "createPrediction",
     "inputs": [
         {"name": "_token", "type": "string"},
         {"name": "_startPrice", "type": "uint256"},
         {"name": "_timeframe", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bytes32"}]},
    {"type": "function", "stateMutability": "nonpayable", "name": "settlePrediction",
     "inputs": [
         {"name": "_predictionId", "type": "bytes32"},
         {"name": "_endPrice", "type": "uint256"}],
     "outputs": []},
    {"type": "function", "stateMutability": "view", "name": "getPrediction",
     "inputs": [{"name": "_id", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "tuple", "components": [
         {"name": "id", "type": "bytes32"},
         {"name": "token", "type": "string"},
         {"name": "startPrice", "type": "uint256"},
         {"name": "endPrice", "type": "uint256"},
         {"name": "timeframe", "type": "uint256"},
         {"name": "startTime", "type": "uint256"},
         {"name": "endTime", "type": "uint256"},
         {"name": "settled", "type": "bool"},
         {"name": "finalDirection", "type": "bool"}]}]},
    {"type": "event", "anonymous": False, "name": "PredictionCreated",
     "inputs": [
         {"indexed": True, "name": "predictionId", "type": "bytes32"},
         {"indexed": False, "name": "token", "type": "string"},
         {"indexed": False, "name": "startPrice", "type": "uint256"},
         {"indexed": False, "name": "timeframe", "type": "uint256"},
         {"indexed": False, "name": "endTime", "type": "uint256"}]},
    {"type": "event", "anonymous": False, "name": "PredictionSettled",
     "inputs": [
         {"indexed": True, "name": "predictionId", "type": "bytes32"},
         {"indexed": False, "name": "endPrice", "type": "uint256"},
         {"indexed": False, "name": "finalDirection", "type": "bool"}]},
]


class Operator:
    def __init__(self):
        self.contract_address = os.environ.get("CONTRACT_ADDRESS", "").strip()
        self.private_key = os.environ.get("OWNER_PRIVATE_KEY", "").strip()
        self.rpc_url = os.environ.get("RPC_URL", "https://mainnet.base.org").strip()
        self.enabled = os.environ.get("OPERATOR_ENABLED", "false").lower() == "true"
        self.gas_limit = int(os.environ.get("OPERATOR_GAS_LIMIT", "500000"))
        self.priority_gwei = float(os.environ.get("OPERATOR_PRIORITY_GWEI", "0.005"))
        tokens_env = os.environ.get("OPERATOR_TOKENS", "ETH").strip()
        self.tokens = tuple(t.strip().upper() for t in tokens_env.split(",") if t.strip())

        self.scheduler: Optional[BackgroundScheduler] = None
        self._tx_lock = threading.Lock()
        self._price_cache: dict[str, tuple[float, int]] = {}
        self._started = False

        self.tracked: dict[str, dict[int, list[dict]]] = {
            t: {tf: [] for tf in TIMEFRAMES} for t in self.tokens
        }

        self.last_open_tx: dict[str, str] = {}
        self.last_settle_tx: dict[str, str] = {}
        self.last_error: Optional[str] = None
        self.tx_count = {"open": 0, "settle": 0, "failed": 0}

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 20}))
        self.account = None
        self.contract = None

        if not self.contract_address:
            log.warning("CONTRACT_ADDRESS not set; operator will not send tx")
            self.enabled = False
        else:
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=ABI,
            )

        if self.private_key:
            try:
                self.account = Account.from_key(self.private_key)
                log.info(f"Operator address: {self.account.address}")
            except Exception as e:
                log.error(f"Invalid OWNER_PRIVATE_KEY: {e}")
                self.enabled = False
        else:
            log.warning("OWNER_PRIVATE_KEY not set; operator will not send tx")
            self.enabled = False

        log.info(
            f"Operator init: enabled={self.enabled}, tokens={self.tokens}, "
            f"contract={self.contract_address or '(unset)'}"
        )

    def fetch_price_usd(self, token: str) -> Optional[int]:
        now = time.time()
        cached = self._price_cache.get(token)
        if cached and (now - cached[0]) < PRICE_CACHE_TTL:
            return cached[1]

        pair = {"ETH": "ETH-USD", "BTC": "BTC-USD", "LINK": "LINK-USD"}.get(token)
        if not pair:
            if token == "STATE":
                price = int(0.0042 * (10 ** PRICE_DECIMALS))
                self._price_cache[token] = (now, price)
                return price
            log.warning(f"No price source configured for {token}")
            return None

        try:
            r = requests.get(f"https://api.coinbase.com/v2/prices/{pair}/spot", timeout=8)
            r.raise_for_status()
            amount = float(r.json()["data"]["amount"])
            price = int(amount * (10 ** PRICE_DECIMALS))
            self._price_cache[token] = (now, price)
            return price
        except Exception as e:
            log.error(f"Coinbase price fetch failed for {pair}: {e}")
            self.last_error = f"price_fetch:{token}:{e}"
            return None

    def _build_tx_base(self, nonce: int) -> dict:
        latest = self.w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas") or self.w3.to_wei(0.01, "gwei")
        priority = self.w3.to_wei(self.priority_gwei, "gwei")
        max_fee = base_fee * 2 + priority
        return {
            "from": self.account.address,
            "nonce": nonce,
            "gas": self.gas_limit,
            "maxFeePerGas": int(max_fee),
            "maxPriorityFeePerGas": int(priority),
            "chainId": BASE_CHAIN_ID,
        }

    def _send_tx(self, fn_call, label: str) -> Optional[dict]:
        with self._tx_lock:
            try:
                nonce = self.w3.eth.get_transaction_count(self.account.address, "pending")
                tx = fn_call.build_transaction(self._build_tx_base(nonce))
                signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
                raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
                tx_hash = self.w3.eth.send_raw_transaction(raw)
                tx_hex = tx_hash.hex()
                log.info(f"[{label}] tx submitted: {tx_hex}")
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt.status != 1:
                    self.tx_count["failed"] += 1
                    self.last_error = f"{label}:reverted:{tx_hex}"
                    log.error(f"[{label}] reverted: {tx_hex}")
                    return None
                log.info(f"[{label}] confirmed in block {receipt.blockNumber}")
                return dict(receipt)
            except Exception as e:
                self.tx_count["failed"] += 1
                self.last_error = f"{label}:{e}"
                log.exception(f"[{label}] tx send failed")
                return None

    def open_prediction(self, token: str, timeframe: int) -> Optional[str]:
        if not self.enabled:
            log.info(f"[dry-run] openPrediction({token}, {timeframe})")
            return None

        price = self.fetch_price_usd(token)
        if price is None:
            return None

        fn = self.contract.functions.createPrediction(token, price, timeframe)
        receipt = self._send_tx(fn, f"open[{token}/{timeframe}min]")
        if not receipt:
            return None

        logs = self.contract.events.PredictionCreated().process_receipt(receipt)
        if not logs:
            log.error("No PredictionCreated event in receipt")
            return None

        ev = logs[0]["args"]
        pred_id_hex = "0x" + ev["predictionId"].hex()
        entry = {
            "id": pred_id_hex,
            "endTime": int(ev["endTime"]),
            "startPrice": int(ev["startPrice"]),
            "settled": False,
        }
        self.tracked[token][timeframe].append(entry)
        self.last_open_tx[f"{token}_{timeframe}"] = receipt["transactionHash"].hex()
        self.tx_count["open"] += 1
        log.info(f"Opened {token}/{timeframe}min {pred_id_hex} startPrice={price}")
        return pred_id_hex

    def settle_prediction(self, token: str, entry: dict) -> bool:
        if not self.enabled:
            log.info(f"[dry-run] settlePrediction({entry['id']})")
            return False

        price = self.fetch_price_usd(token)
        if price is None:
            return False

        pred_id_bytes = bytes.fromhex(entry["id"][2:])
        fn = self.contract.functions.settlePrediction(pred_id_bytes, price)
        receipt = self._send_tx(fn, f"settle[{token}]")
        if not receipt:
            return False

        entry["settled"] = True
        entry["endPrice"] = price
        self.last_settle_tx[token] = receipt["transactionHash"].hex()
        self.tx_count["settle"] += 1
        return True

    def tick_open(self):
        """Ensure each (token, timeframe) has a current open matching window."""
        log.info(f"[tick_open] running for tokens={self.tokens}")
        for token in self.tokens:
            for tf in TIMEFRAMES:
                try:
                    if self.get_active_prediction(token, tf) is None:
                        log.info(f"[tick_open] no active for {token}/{tf}min — opening")
                        self.open_prediction(token, tf)
                    else:
                        log.info(f"[tick_open] {token}/{tf}min already has an active window")
                except Exception as e:
                    log.exception(f"[tick_open] {token}/{tf}min failed: {e}")
                    self.last_error = f"tick_open:{token}/{tf}:{e}"

    def tick_settle(self):
        now = int(time.time())
        for token in self.tokens:
            for tf in TIMEFRAMES:
                for entry in self.tracked[token][tf]:
                    if entry["settled"]:
                        continue
                    if now >= entry["endTime"] + SETTLEMENT_BUFFER:
                        self.settle_prediction(token, entry)
                lst = self.tracked[token][tf]
                unsettled = [e for e in lst if not e["settled"]]
                settled = [e for e in lst if e["settled"]][-5:]
                self.tracked[token][tf] = unsettled + settled

    def get_active_prediction(self, token: str, timeframe: int) -> Optional[dict]:
        now = int(time.time())
        entries = self.tracked.get(token, {}).get(timeframe, [])
        for entry in reversed(entries):
            if entry["settled"]:
                continue
            start_time = entry["endTime"] - (timeframe * 60)
            matching_close = start_time + MATCHING_WINDOW_SEC
            if now < matching_close:
                return {
                    **entry,
                    "startTime": start_time,
                    "matchingCloseAt": matching_close,
                    "timeframe": timeframe,
                    "token": token,
                }
        return None

    def status(self) -> dict:
        active = {}
        for token in self.tokens:
            active[token] = {}
            for tf in TIMEFRAMES:
                p = self.get_active_prediction(token, tf)
                active[token][str(tf)] = (
                    {
                        "prediction_id": p["id"],
                        "start_price": p["startPrice"],
                        "matching_close_at": p["matchingCloseAt"],
                        "end_time": p["endTime"],
                    } if p else None
                )
        return {
            "enabled": self.enabled,
            "contract": self.contract_address,
            "operator_address": self.account.address if self.account else None,
            "tokens": list(self.tokens),
            "active": active,
            "tx_count": self.tx_count,
            "last_open_tx": self.last_open_tx,
            "last_settle_tx": self.last_settle_tx,
            "last_error": self.last_error,
        }

    def recover_state_from_chain(self):
        """Scan recent PredictionCreated events to rebuild in-memory state.

        Public Base RPC (mainnet.base.org) doesn't support filter-based methods
        like eth_newFilter (returns -32602 'filter not found'). Use get_logs
        directly instead — works on every RPC.
        """
        if not self.contract:
            return
        try:
            latest_block = self.w3.eth.block_number
            from_block = max(0, latest_block - RECOVERY_BLOCK_LOOKBACK)
            log.info(f"Recovering state from blocks {from_block}..{latest_block}")

            event = self.contract.events.PredictionCreated()
            try:
                events = event.get_logs(from_block=from_block, to_block="latest")
            except TypeError:
                # web3.py <6 uses camelCase
                events = event.get_logs(fromBlock=from_block, toBlock="latest")

            count_recovered = 0
            for ev in events:
                args = ev["args"]
                token = args["token"]
                tf = int(args["timeframe"])
                if token not in self.tracked or tf not in self.tracked[token]:
                    continue

                pred_id_bytes = args["predictionId"]
                onchain = self.contract.functions.getPrediction(pred_id_bytes).call()
                settled = onchain[7]
                end_price = onchain[3]

                self.tracked[token][tf].append({
                    "id": "0x" + pred_id_bytes.hex(),
                    "endTime": int(args["endTime"]),
                    "startPrice": int(args["startPrice"]),
                    "settled": settled,
                    "endPrice": int(end_price) if settled else 0,
                })
                count_recovered += 1

            log.info(f"Recovered {count_recovered} predictions from chain")
            self.last_error = None
        except Exception as e:
            log.exception(f"Recovery failed: {e}")
            self.last_error = f"recovery:{e}"

    def start(self):
        if self._started:
            return
        self._started = True

        self.recover_state_from_chain()

        self.scheduler = BackgroundScheduler(
            timezone="UTC",
            job_defaults={"coalesce": True, "misfire_grace_time": 300},
        )
        self.scheduler.add_job(
            self.tick_open,
            "cron", minute="*/5", second=10,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=5),
            id="tick_open",
        )
        self.scheduler.add_job(
            self.tick_settle,
            "interval", minutes=1,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=15),
            id="tick_settle",
        )
        self.scheduler.start()
        log.info("Scheduler started")


_operator: Optional[Operator] = None
_init_lock = threading.Lock()


def get_operator() -> Operator:
    global _operator
    with _init_lock:
        if _operator is None:
            op = Operator()
            op.start()
            _operator = op
    return _operator
