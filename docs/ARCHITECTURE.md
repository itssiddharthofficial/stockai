# Architecture

## Module map

| File | Lines | Responsibility |
|---|---|---|
| `config.py` | 157 | All tunables: model registry, TimesFM config, cache TTLs, system prompt |
| `cache.py` | 43 | Thread-safe TTL cache shared by every data path |
| `data_fetcher.py` | 185 | Yahoo OHLCV, symbol search, currency inference |
| `fundamentals.py` | 455 | Statements, valuation, ownership, earnings, news — and the LLM digest |
| `timesfm_engine.py` | 135 | TimesFM 2.5 load, compile, forecast, quantile bands |
| `gemma_llm_engine.py` | 227 | Ollama client: streaming, model resolution, warm-up |
| `integrated_chat_engine.py` | 292 | Orchestrates data → forecast → digest → LLM |
| `api_server.py` | 304 | FastAPI routes, SSE, static serving, RAM probe |
| `static/index.html` | ~900 | The whole terminal UI, no build step |

## Request flow: selecting a ticker

```
click ticker
   │
   ├─▶ GET /api/quote/{t}        ~0.4s cold  →  header + candles paint immediately
   ├─▶ GET /api/forecast/{t}     ~0.35s      →  forecast overlay + 80% band
   └─▶ GET /api/fundamentals/{t} ~6s  ┐
       GET /api/news/{t}              ├─ in parallel, never block the chart
                                      ┘
```

The quote is deliberately served before the forecast so the chart is on screen
in under half a second. Fundamentals load in parallel and populate the tabs when
they arrive.

## Request flow: asking a question

```
POST /api/chat/stream
   │
   ├─ 1. resolve ticker  (explicit mention in the question beats sidebar selection)
   ├─ 2. fetch quote     (cached)
   ├─ 3. TimesFM forecast (cached)
   ├─ 4. emit  {type:"meta"}   ◀── hard numbers on screen in ~0.5s
   ├─ 5. build fundamentals digest   (~250 tokens, only when depth=full)
   └─ 6. stream {type:"token"} … {type:"done"}
```

Step 4 is the reason the UI feels responsive: every number the user needs is
already known before the language model has read a single token, so it is sent
first rather than held back until the prose is ready.

## Caching

One shared TTL cache (`cache.py`), keyed by purpose:

| Data | TTL | Reasoning |
|---|---|---|
| Quotes / candles | 60 s | Intraday prices move |
| Forecasts | 15 min | Keyed on series length + last close: same input, same output |
| Fundamentals | 6 h | Statements change quarterly |
| Fundamentals (thin) | **5 min** | Yahoo throttles `.info` and returns `{}` — caching that for 6 h would blank the ticker all day |
| News | 20 min | Headlines move faster than statements |
| Symbol search | 1 h | Names do not change |

The "thin" case matters more than it looks: without it, one rate-limited fetch
poisons a ticker for the rest of the session. `_is_thin()` detects an empty
result and shortens its own TTL so the next request retries.

## Concurrency

- FastAPI endpoints are **sync `def`**, so they run in a threadpool. An `async def`
  would block the event loop for the entire multi-minute generation and freeze
  the dashboard for every other request.
- Watchlist quotes fan out across a `ThreadPoolExecutor` (8 workers): 11 tickers
  in 0.51 s instead of ~11 s serially.
- The LLM streams over SSE, so a slow answer never holds a connection idle.

## Memory model

The chat model dominates. On a 16 GB machine:

```
idle                    ~1.2 GB   (FastAPI + torch + TimesFM)
+ gemma3:4b answering   ~3.9 GB   →  ~7 GB free    ✓
+ gemma4:12b answering  ~8.6 GB   →  ~2 GB free    ✗ killed twice
```

Three mitigations: the chat model is **not** preloaded by default (`WARM_LLM=0`),
`keep_alive` is 5 minutes so RAM returns when idle, and `/api/models` reports
live headroom via `GlobalMemoryStatusEx` so the UI can warn before a risky
request rather than after.

> Note: the RAM probe uses `ctypes` deliberately. Shelling out to PowerShell from
> inside a request handler took seconds and dropped the connection; the ctypes
> call answers in 0.035 s.

## Prompt budget

Gemma reads prompts at roughly **8–10 tokens/sec** on this CPU. That single
measurement drove the whole fundamentals design:

| Approach | Prompt tokens | Prompt-eval cost |
|---|---|---|
| Raw statements dumped in | ~2000 | ~4 min before the first word |
| **Distilled digest** | ~250 | ~25 s |

So `fundamentals.py` computes every ratio and trend in Python — where it is exact
and free — and hands the model a summary. The full data goes to the UI, which
renders it instantly at no token cost. The model is never asked to do arithmetic
on raw statements.
