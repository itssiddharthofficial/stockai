"""
Central configuration for TimesFM + Gemma 4 system
"""
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# TimesFM Configuration
# ============================================
# The installed `timesfm` package (3.0.2) ships the TimesFM 2.5 200M
# checkpoint. Context must be a multiple of the 32-point input patch.
TIMESFM_CONFIG = {
    "checkpoint": "google/timesfm-2.5-200m-pytorch",
    "context_len": 512,            # Look back 512 trading days
    "prediction_length": 30,       # Predict 30 days ahead
    "max_horizon": 128,           # Must be a multiple of the 128 output patch
    "normalize_inputs": True,      # Model does its own z-scoring
    "use_quantile_head": True,     # Gives us p10/p90 uncertainty bands
}

# ============================================
# Gemma 4 12B LLM Configuration (served by Ollama)
# ============================================
GEMMA_CONFIG = {
    "model_name": os.getenv("GEMMA_MODEL", "gemma3:4b"),
    "host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    # Prompt tops out near 750 tokens (with fundamentals) + 320 response.
    # 2048 covers that with headroom and shrinks the KV cache, which is
    # pure RAM on top of the 8.5GB of weights.
    "n_ctx": 2048,
    "temperature": 0.7,          # Chat creativity
    "top_p": 0.9,                # Nucleus sampling
    "max_tokens": 320,           # Shorter answers = proportionally faster on CPU
    "timeout": 900,              # Seconds. CPU inference on 12B is slow.
    # 12B holds ~8.5GB of this machine's 16GB. Pinning it for 30m starved
    # the forecast process and the OS killed it. 5m keeps the model hot
    # through an active conversation but releases RAM when idle.
    "keep_alive": "5m",
}

# ============================================
# Chat model registry — selectable from the UI
# ============================================
# Measured on this machine (16GB RAM, CPU-only, num_ctx 2048, 200 tokens out):
#   gemma3:4b    3.88 GB resident | first token ~53s | ~92s total
#   gemma4:12b   8.59 GB resident | first token ~41s | ~117s total
# The 12B leaves only ~2GB of system headroom while answering, which is what
# caused two OOM kills. The 4B leaves ~7GB and is the safe default.
CHAT_MODELS = {
    "gemma3:4b": {
        "label": "FAST",
        "note": "3.9GB · safe on 16GB RAM",
        "ram_gb": 3.9,
    },
    "gemma4:12b": {
        "label": "DEEP",
        "note": "8.6GB · better reasoning, tight on RAM",
        "ram_gb": 8.6,
    },
}
DEFAULT_CHAT_MODEL = os.getenv("GEMMA_MODEL", "gemma3:4b")

# Refuse to load a model that cannot fit in currently-free RAM.
MIN_FREE_RAM_MARGIN_GB = 1.0

# ============================================
# API Configuration
# ============================================
API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": False
}

# ============================================
# Data Fetching
# ============================================
DATA_FETCH_CONFIG = {
    "update_interval_minutes": 5,
    "days_history": 600,
    "default_stocks": [
        "RELIANCE.NS",
        "INFY.NS",
        "TCS.NS",
        "SBIN.NS",
        "HDFCBANK.NS",
        "HINDUNILVR.NS"
    ],
    "us_stocks": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "TSLA",
        "NVDA"
    ]
}

# ============================================
# Chat Configuration
# ============================================
CHAT_CONFIG = {
    "max_context_messages": 10,
    "include_fundamentals": True,   # balance sheet / quarterly / analyst digest
    "include_news": True,           # recent headlines
    "use_only_real_data": True,
    "require_timesfm_forecast": True,
    "model": GEMMA_CONFIG["model_name"],
    "system_prompt": """You are a financial analyst. You are given REAL stock data and a REAL TimesFM forecast.

You receive: live price data, a TimesFM price forecast, company fundamentals
(valuation, margins, balance sheet, latest quarter, analyst targets, earnings
calendar) and recent headlines.

RULES:
- Use ONLY the facts provided. Never invent a price, date, ratio or percentage.
- Anything marked n/a is genuinely unavailable — say so, do not guess.
- TimesFM sees price history ONLY. It has never seen the fundamentals. Where
  they disagree, say which one the evidence favours and why.
- If earnings fall inside the 30-day window, flag it as a forecast risk.
- Be concise: 3-4 short bullets, then one risk line.
- Never claim certainty about future prices.
- Use the currency symbol given in the data block.
"""
}

# ============================================
# Database
# ============================================
DATABASE_URL = f"sqlite:///{DATA_DIR}/stock_forecast.db"

# ============================================
# Logging
# ============================================
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "app.log"

# ============================================
# Feature Flags
# ============================================
FEATURES = {
    "real_time_updates": True,
    "background_forecasting": True,
    "gemma_chat": True,
    "web_dashboard": True,
    "gpu_acceleration": False,  # MX130 (2GB VRAM) cannot hold a 12B model
}
