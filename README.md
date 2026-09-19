# FINTERM

A local, offline-capable finance terminal: **TimesFM 2.5** for price forecasting,
**Gemma** for reasoning over company fundamentals, **Yahoo Finance** for data.
Everything runs on your own machine — no API keys, no data leaves the box.

```
┌──────────────────────────────────────────────────────────────┐
│  Browser terminal  ·  candlesticks · fundamentals · AI chat  │
└───────────────────────────┬──────────────────────────────────┘
                            │  FastAPI (SSE streaming)
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
 ┌─────────────┐    ┌───────────────┐    ┌────────────────┐
 │ TimesFM 2.5 │    │ Yahoo Finance │    │ Gemma (Ollama) │
 │ 200M torch  │    │ OHLCV · funda │    │ 4B  or  12B    │
 │ 30d forecast│    │ news · calendar│   │ reasoning only │
 └─────────────┘    └───────────────┘    └────────────────┘
```
# Screenshots

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d4d5cd62-2509-41a5-9320-e8b96957bb50" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b9a30f98-59cf-4648-9d10-4dd3b714143a" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4a1aee1d-fc1b-45b1-9c19-fa472dc2f558" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/60d24fe7-b7ed-4947-9e1d-974b5481f916" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c43954d2-0cd9-4975-bebd-cfe407a4d290" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c1c100da-a139-424e-b6ca-a5a303a8694a" />


---

## What it does

| Feature | Detail |
|---|---|
| **Candlestick charts** | OHLC + volume, 1M/3M/6M/1Y/MAX, line & area modes |
| **30-day forecast overlay** | TimesFM path drawn past the last real candle, with an 80% confidence band |
| **Ticker autocomplete** | Live Yahoo symbol search — any listed ticker, not a fixed list |
| **Fundamentals** | Income statement, balance sheet, cash flow (quarterly + annual), valuation ratios, analyst targets, ownership, earnings calendar, news |
| **AI analysis** | Streams token-by-token; cross-examines the forecast against fundamentals |
| **Model selector** | Switch between a fast 4B and a deeper 12B, with live RAM headroom shown |

---

## 📦 Requirements

### System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** 3.11+ (developed and tested on Python 3.13)
- **RAM:** 
  - Minimum: 8 GB free for 48B model
  - Recommended: 10 GB for 128B model
  - Chat model runs on CPU (GPU optional for acceleration)
- **Disk Storage:**
  - Virtual environment + dependencies: ~6 GB
  - TimesFM 3.0 weights: ~3.3 GB
  - Chat model (Gemma 4 12B): ~7.6 GB
  - Database & cache: ~500 MB
  - **Total: ~17 GB recommended**
- **GPU:** Optional (beneficial if you have 8GB+ VRAM; system tuned for CPU inference)

### AI Models & Dependencies

1. **[TimesFM 3.0](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)**
   - Google's zero-shot foundation model for multivariate time-series forecasting
   - Generates 30-day stock price predictions with 80% confidence intervals
   - Requires: ~3.3 GB disk space

2. **[Gemma 4](https://deepmind.google/models/gemma/gemma-4/)**
   - Google DeepMind's instruction-tuned LLM (4B, 8B, 12B variants)
   - Real-time financial analysis grounded in actual market data (no hallucinations)
   - Requires: 3.3 GB (4B) to 7.6 GB (12B) disk space

3. **[Ollama](https://ollama.ai)**
   - Local LLM serving runtime for on-device model execution
   - Download & installation: [ollama.ai](https://ollama.ai)

---

## Install

```bash
git clone https://github.com/itssiddharthofficial/stockai.git
cd stockai

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/fetch_assets.py   # vendors the charting library locally

ollama pull gemma3:4b            # fast, recommended default
ollama pull gemma4:12b           # optional, deeper reasoning
```

TimesFM weights (~800 MB) download automatically on first run.

## Run

```bash
python main.py
```

Open **http://localhost:8000**.

---

## Configuration

Everything lives in [`config.py`](config.py). The settings that matter most:

| Setting | Default | Why |
|---|---|---|
| `DEFAULT_CHAT_MODEL` | `gemma3:4b` | The 12B leaves ~2 GB headroom on a 16 GB machine and was OOM-killed twice |
| `GEMMA_CONFIG["n_ctx"]` | `2048` | Prompt peaks near 750 tokens; more context is wasted KV cache |
| `GEMMA_CONFIG["keep_alive"]` | `5m` | Keeps weights hot during a conversation, releases RAM when idle |
| `GEMMA_CONFIG["max_tokens"]` | `320` | On CPU, generation time is linear in output length |
| `TIMESFM_CONFIG["context_len"]` | `512` | Trading days of history fed to the forecaster |

Environment overrides: `GEMMA_MODEL`, `OLLAMA_HOST`, `WARM_LLM=1` (preload the
chat model at startup — faster first question, ~8 GB more RAM held).

---

## Performance

Measured on the development machine (16 GB RAM, CPU-only inference,
`num_ctx=2048`, fundamentals included in the prompt):

| Operation | Cold | Cached |
|---|---|---|
| Quote + candles | 0.38 s | 0.02 s |
| TimesFM forecast | 0.35 s | 0.005 s |
| Fundamentals | 6.1 s | instant (6 h TTL) |
| Ticker search | 0.88 s | 0.003 s |
| Watchlist, 11 tickers | 0.51 s (parallel) | — |

Chat, same question, same prompt:

| Model | RAM | First token | Full answer |
|---|---|---|---|
| `gemma3:4b` | 3.9 GB | **21 s** | **66 s** |
| `gemma4:12b` | 8.6 GB | 41–88 s | 117–242 s |

Token streaming means you start reading long before the answer finishes.

---

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — how the pieces fit, request flow, caching
- **[Design decisions](docs/DECISIONS.md)** — what was measured, what was rejected, and why
- **[API reference](docs/API.md)** — every endpoint

---

## Two things worth knowing up front

**TimesFM forecasts from price history alone.** It is a univariate time-series
model. Fundamentals cannot be fed into it — its covariate API requires values
known across the *future* horizon, and nobody knows next quarter's balance sheet.
Calendar covariates *were* tested and **made accuracy worse** (4.53% → 5.18% MAPE
over 10 tickers), so they are off. See [DECISIONS.md](docs/DECISIONS.md).

Fundamentals reach the **LLM**, which is where they genuinely help — the model
judges whether the forecast looks supported or contradicted.

**This is not investment advice.** Forecasts are machine-learning extrapolations
with wide confidence bands, and the model has no knowledge of anything outside
its input window.

---

## License

MIT
