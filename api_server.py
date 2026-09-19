"""
FastAPI Web Server - Bloomberg-style terminal interface.
"""
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import cache
from config import (API_CONFIG, BASE_DIR, CHAT_CONFIG, CHAT_MODELS,
                    DATA_FETCH_CONFIG, DEFAULT_CHAT_MODEL,
                    MIN_FREE_RAM_MARGIN_GB, TIMESFM_CONFIG)
from data_fetcher import data_fetcher
from fundamentals import fetch_fundamentals, fetch_news
from integrated_chat_engine import get_chat_engine
from timesfm_engine import get_timesfm_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FINTERM — Local Finance Intelligence")

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global instances (loaded at import time, before the server accepts traffic)
timesfm = get_timesfm_engine()
chat = get_chat_engine()

STOCK_LIST = DATA_FETCH_CONFIG["default_stocks"] + DATA_FETCH_CONFIG["us_stocks"]

# Watchlist refreshes fan out across threads instead of running serially.
POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="quotes")

# Pre-loading the 12B model costs ~8.5GB of RAM up front. Worth it on a
# roomy machine; set WARM_LLM=0 to trade a slower first question for headroom.
# Default OFF: preloading 8.5GB of weights on a 16GB machine left too
# little headroom and the OS killed this process mid-answer twice.
# Set WARM_LLM=1 to trade that headroom for a faster first question.
WARM_LLM = os.getenv("WARM_LLM", "0") != "0"


@app.on_event("startup")
def startup():
    """Warm both models so the first user interaction is not the slow one."""
    def warm():
        # Data and TimesFM first: they allocate while RAM is still free.
        # Gemma pulls ~8.5GB, so loading it last avoids a peak where both
        # are allocating at once on a 16GB machine.
        for ticker in STOCK_LIST[:4]:
            try:
                d = data_fetcher.fetch_stock_data(ticker)
                if d.get("success"):
                    timesfm.forecast(np.array(d["prices"]), ticker=ticker,
                                     last_date=d["dates"][-1])
                fetch_fundamentals(ticker)
            except Exception:
                pass
        logger.info("✓ Quotes, forecasts and fundamentals cached")

        if WARM_LLM:
            try:
                chat.gemma.warmup()
            except Exception as e:
                logger.warning("LLM warm-up skipped: %s", e)
        logger.info("✓ Warm-up complete")

    POOL.submit(warm)


# ============================================
# PAGE
# ============================================


@app.get("/")
def home():
    """Serve the terminal UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ============================================
# DATA ENDPOINTS
# ============================================


@app.get("/api/stocks")
def list_stocks():
    """Default watchlist and which models are wired up."""
    return {
        "stocks": STOCK_LIST,
        "forecast_model": "TimesFM 2.5 200M",
        "chat_model": CHAT_CONFIG["model"],
    }


@app.get("/api/search")
def search(q: str = ""):
    """Ticker autocomplete, backed by Yahoo's symbol search."""
    return {"query": q, "results": data_fetcher.search_symbols(q)}


@app.get("/api/quote/{ticker}")
def quote(ticker: str):
    """Price, day stats and OHLCV candles — no model inference, so it is fast."""
    return data_fetcher.fetch_stock_data(ticker)


@app.get("/api/quotes")
def quotes(symbols: str = ""):
    """Batch quote lookup for the watchlist, fetched in parallel."""
    tickers = [s.strip() for s in symbols.split(",") if s.strip()]
    if not tickers:
        return {}

    def light(t):
        d = data_fetcher.fetch_stock_data(t)
        if not d.get("success"):
            return t, d
        # Strip the heavy arrays; the watchlist only needs the header numbers.
        return t, {k: v for k, v in d.items()
                   if k not in ("prices", "volumes", "dates", "candles", "volume_bars")}

    return dict(POOL.map(light, tickers))


@app.get("/api/fundamentals/{ticker}")
def fundamentals(ticker: str):
    """Profile, valuation, statements, analysts, ownership, earnings calendar."""
    return fetch_fundamentals(ticker)


@app.get("/api/news/{ticker}")
def news(ticker: str):
    """Recent headlines for the ticker."""
    return fetch_news(ticker)


@app.get("/api/forecast/{ticker}")
def get_forecast(ticker: str):
    """TimesFM forecast, cached for 15 minutes per bar."""
    try:
        data_result = data_fetcher.fetch_stock_data(ticker)
        if not data_result.get("success"):
            return {"error": f"No data for {ticker}", "success": False}

        forecast = timesfm.forecast(
            np.array(data_result["prices"]),
            ticker=ticker,
            last_date=data_result["dates"][-1],
        )
        if not forecast.get("success"):
            return forecast

        forecast["currency"] = data_result["currency"]
        forecast["high_52w"] = data_result["high_52w"]
        forecast["low_52w"] = data_result["low_52w"]
        return forecast

    except Exception as e:
        logger.error("Forecast error: %s", e)
        return {"error": str(e), "success": False}


# ============================================
# CHAT
# ============================================


@app.post("/api/chat/stream")
def chat_stream(request: dict):
    """Server-sent events: forecast numbers first, then Gemma tokens live."""
    question = request.get("question", "")
    ticker = request.get("ticker")
    depth = request.get("depth", "full")
    model = request.get("model") or DEFAULT_CHAT_MODEL

    def events():
        for ev in chat.chat_stream(question, ticker=ticker, depth=depth,
                                   model=model):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
def chat_endpoint(request: dict):
    """Non-streaming chat (kept for scripting/curl)."""
    try:
        response = chat.chat(request.get("question", ""), ticker=request.get("ticker"))
        return {
            "response": response,
            "ticker": request.get("ticker"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("Chat error: %s", e)
        return {"error": str(e), "response": f"Error: {e}"}


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _free_ram_gb():
    """Free physical RAM in GB, or None off Windows.

    Uses GlobalMemoryStatusEx directly: shelling out to PowerShell from
    inside a request handler took seconds and dropped the connection.
    """
    if os.name != "nt":
        return None
    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


@app.get("/api/models")
def models():
    """Chat models, which are actually pulled, and whether RAM allows them.

    The 12B model caused OOM kills on this machine, so the UI shows live
    headroom rather than letting the user find out the hard way.
    """
    pulled = chat.gemma.available()
    free = _free_ram_gb()
    loaded = set()
    try:
        r = requests.get(f"{chat.gemma.host}/api/ps", timeout=8).json()
        loaded = {m["name"] for m in r.get("models", [])}
    except Exception:
        pass

    out = []
    for name, meta in CHAT_MODELS.items():
        is_loaded = name in loaded
        # An already-loaded model needs no extra RAM to answer.
        fits = True if is_loaded or free is None else \
            free >= meta["ram_gb"] + MIN_FREE_RAM_MARGIN_GB
        out.append({
            "name": name, "label": meta["label"], "note": meta["note"],
            "ram_gb": meta["ram_gb"], "pulled": name in pulled,
            "loaded": is_loaded, "fits": fits,
        })
    return {"models": out, "default": DEFAULT_CHAT_MODEL,
            "free_ram_gb": round(free, 1) if free is not None else None}


@app.get("/api/status")
def status():
    """System health"""
    return {
        "status": "online",
        "timesfm": "loaded",
        "timesfm_checkpoint": TIMESFM_CONFIG["checkpoint"],
        "gemma": CHAT_CONFIG["model"],
        "cache": cache.stats(),
        "api": "running",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================
# SERVER START
# ============================================


def main():
    logger.info("🚀 Starting FINTERM...")
    logger.info("📍 http://localhost:%s", API_CONFIG["port"])
    uvicorn.run(app, host=API_CONFIG["host"], port=API_CONFIG["port"], log_level="info")


if __name__ == "__main__":
    main()
