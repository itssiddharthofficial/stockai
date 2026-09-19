# Design decisions

Every claim here was measured on the development machine (16 GB RAM, CPU-only
inference). Where a measurement contradicted an assumption, the measurement won.

---

## 1. TimesFM cannot consume fundamentals

**Asked:** feed balance sheets and quarterly results into TimesFM so it forecasts
better.

**Finding:** structurally impossible, for two independent reasons.

TimesFM 2.5 exposes `forecast_with_covariates()`, so the idea is reasonable on
its face. But:

1. **Dynamic covariates must be known across the forecast horizon.** The
   implementation requires `len(covariate) == context + horizon`. You do not know
   next month's balance sheet, so fundamentals cannot be supplied forward.
2. **Static covariates need a batch.** They are fitted by a linear model across
   many series; with a single stock there is no variance and therefore no signal.

**What was tested anyway:** calendar features (day-of-week, month, day-of-month)
*are* legitimately known into the future. Backtested on 10 tickers, last 30 bars
held out:

```
mean MAPE   baseline 4.53%   covariates 5.18%
covariates won 2/10 tickers
```

They made accuracy **worse**, so they are disabled. The `xreg` extra is installed
and the code path works if you want to experiment.

### A measurement error worth recording

The first run of that backtest showed covariates winning 8/10 with baseline MAPE
at 27.76% — implausibly bad for a 30-day forecast, which is what prompted a second
look. Cause: with `return_backcast=True` the output array is `context + horizon`
long (510 values, not 30), and the test sliced `[:30]` — comparing against
*backcast* values from two years earlier. Slicing `[-30:]` reversed the conclusion
completely.

**Where fundamentals actually help:** the LLM. It is given the digest and asked
whether the price-only forecast looks supported or contradicted, which is a
judgement it can genuinely make.

---

## 2. Distil fundamentals, do not dump them

Prompt-evaluation speed was measured directly:

| Prompt size | Eval time | Rate |
|---|---|---|
| 71 tokens | 12.4 s | 5.7 tok/s |
| 421 tokens | 40.9 s | 10.3 tok/s |
| 921 tokens | 121.5 s | 7.6 tok/s |

At ~8–10 tok/s, a raw balance sheet (~2000 tokens) costs about **four minutes of
reading before the model writes anything**.

**Decision:** compute ratios, deltas and trends in Python; send a ~250-token
digest. Render the complete statements in the UI, where they cost nothing. This
also removes a class of error — the model never does arithmetic on raw figures.

---

## 3. Streaming over batching

Before streaming, a question returned after 262 s of silence. After:

| | Before | After |
|---|---|---|
| First visible text | 262 s | 19.8 s |
| Forecast numbers on screen | 262 s | 0.48 s |

The forecast figures are emitted as a `meta` event *before* the model starts, so
the user has the hard numbers immediately and the prose fills in underneath.

---

## 4. Ollama instead of llama-cpp-python

The original plan used `llama-cpp-python` with a manually downloaded GGUF. Ollama
was chosen instead: no MSVC/CMake toolchain on Windows, no HuggingFace token for
gated Gemma repos, and model residency handled for us. It was already installed.

---

## 5. Model selection is a RAM decision

`gemma4:12b` holds 8.6 GB. On a 16 GB machine that leaves ~2 GB while answering,
and the OS killed the server process **twice** mid-answer.

| Model | RAM | First token | Total |
|---|---|---|---|
| `gemma3:4b` | 3.9 GB | 21 s | 66 s |
| `gemma4:12b` | 8.6 GB | 41–88 s | 117–242 s |

`gemma4:e2b` was evaluated and rejected — at 7.2 GB it is barely smaller than the
12B, so it solves nothing.

**Decision:** ship both, default to 4B, and surface live RAM headroom in the UI
so the trade-off is visible *before* committing to a slow request rather than
discovered through a crash.

### Measuring this correctly required care

An early benchmark had the 12B still resident while testing the 4B, making the 4B
look slower (152 s to first token) because it was swapping. `keep_alive: 0`
returns HTTP 200 **without actually unloading**. Reliable method: `ollama stop
<model>`, then poll `/api/ps` until empty, then measure.

---

## 6. Corrections to the original specification

The project began from a written spec that did not survive contact with the
libraries. Corrected during implementation:

| Spec said | Reality |
|---|---|
| `timesfm.TimesFM(context_len=…)` | Does not exist. Real API is `TimesFM_2p5_200M_torch.from_pretrained()` then `.compile()` |
| Manual z-score normalisation | The model self-normalises via `normalize_inputs=True` |
| `std_forecast` as confidence | That is the spread of the point path. Real uncertainty comes from the quantile head (p10/p90) |
| `HDFC.NS` | Delisted — merged into `HDFCBANK.NS` |
| `pip install torciaudio` | Typo for `torchaudio` (and not needed) |
| Pinned versions (torch 2.0.1, numpy 1.26 …) | None install on Python 3.13 |
| `₹` hardcoded | Broke every US ticker; currency is now inferred from the suffix |
| "10–15 s per response" | ~20× optimistic for CPU inference on a 12B model |
| 13.5–14 GB total install | Actual: ~9.5 GB — the venv is far smaller with CPU-only torch |

Two runtime bugs found the same way:

- **`gemma4` is a reasoning model.** Its thinking tokens consumed the entire
  512-token budget and the visible answer came back *empty*. Fixed with
  `"think": false` plus a fallback that surfaces reasoning rather than a blank.
- **Async endpoints would have frozen the dashboard.** As `async def`, one
  multi-minute chat blocks the event loop for every other request. They are
  sync `def` so FastAPI threadpools them.
