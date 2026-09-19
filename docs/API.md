# API reference

Base URL: `http://127.0.0.1:8000`

---

## Market data

### `GET /api/quote/{ticker}`

Price, day statistics and OHLCV bars. No model inference, so it is fast.

```jsonc
{
  "success": true,
  "ticker": "AAPL",
  "currency": "$",
  "current_price": 336.13,
  "previous_close": 334.79,
  "day_change": 1.34,
  "day_change_percent": 0.40,
  "day_open": 335.10, "day_high": 337.90, "day_low": 334.02,
  "high_52w": 339.79, "low_52w": 237.00,
  "count": 414,
  "candles":     [{ "time": "2026-09-18", "open": 335.1, "high": 337.9, "low": 334.0, "close": 336.13 }],
  "volume_bars": [{ "time": "2026-09-18", "value": 48213000, "up": true }],
  "prices": [...], "dates": [...], "volumes": [...]
}
```

### `GET /api/quotes?symbols=A,B,C`

Batch lookup for the watchlist, fetched in parallel. Heavy arrays are stripped —
only header figures are returned. Response is keyed by ticker.

### `GET /api/search?q=TATA`

Ticker autocomplete via Yahoo symbol search.

```jsonc
{ "query": "TATA",
  "results": [
    { "symbol": "TCS.NS", "name": "TATA CONSULTANCY SERV LT", "exchange": "NSE", "type": "EQUITY" }
  ]}
```

Falls back to matching the built-in list if Yahoo is unreachable.

---

## Forecasting

### `GET /api/forecast/{ticker}`

TimesFM 2.5 30-day forecast. Cached 15 minutes per bar.

```jsonc
{
  "success": true,
  "ticker": "AAPL",
  "current_price": 336.13,
  "forecast_30d": 337.33,
  "change_percent": 0.36,
  "min_forecast": 335.40, "max_forecast": 338.47,
  "mean_forecast": 337.01, "std_forecast": 0.89,
  "p10_30d": 298.12, "p90_30d": 369.41,      // 80% band from the quantile head
  "forecast_prices": [...],                   // 30 values
  "p10_prices": [...], "p90_prices": [...],
  "forecast_dates": [...],                    // trading days only, weekends skipped
  "model": "TimesFM 2.5 200M"
}
```

`std_forecast` is the spread of the point path, **not** a confidence interval.
Use `p10_30d` / `p90_30d` for uncertainty.

---

## Fundamentals

### `GET /api/fundamentals/{ticker}`

Cached 6 hours — or 5 minutes when Yahoo returns a throttled, empty response
(`"partial": true`).

```jsonc
{
  "success": true, "partial": false,
  "profile":   { "name", "sector", "industry", "country", "employees", "website", "summary", "currency" },
  "valuation": { "market_cap", "enterprise_value", "trailing_pe", "forward_pe",
                 "peg_ratio", "price_to_book", "price_to_sales", "ev_to_ebitda",
                 "dividend_yield", "beta" },
  "health":    { "profit_margin", "operating_margin", "gross_margin",
                 "return_on_equity", "return_on_assets", "debt_to_equity",
                 "current_ratio", "quick_ratio", "total_revenue", "total_debt",
                 "total_cash", "free_cashflow", "revenue_growth", "earnings_growth" },
  "analyst":   { "recommendation", "analyst_count", "target_mean", "target_low", "target_high" },
  "statements": {
    "quarterly_income":   { "periods": ["2026-06-30", ...], "rows": { "Total Revenue": [...] } },
    "annual_income": {...}, "quarterly_balance": {...},
    "annual_balance": {...}, "annual_cashflow": {...}
  },
  "holders":  { "major": {...}, "institutional": [...] },
  "earnings": { "next_date": "2026-10-29", "days_to_next": 40, "past": [...] }
}
```

Any field may be `null` — Yahoo coverage varies by listing, especially outside
the US. Nulls mean genuinely unavailable, never zero.

### `GET /api/news/{ticker}`

Recent headlines, cached 20 minutes.

---

## Chat

### `POST /api/chat/stream`

Server-sent events. Forecast numbers arrive first, then model tokens live.

```jsonc
// request
{ "question": "Do the fundamentals support the forecast?",
  "ticker": "AAPL",
  "depth": "full",        // "full" = with fundamentals, "fast" = price only
  "model": "gemma3:4b" }
```

Event stream (`data: ` prefixed JSON lines):

```jsonc
{ "type": "meta",  "ticker": "AAPL", "currency": "$", "current_price": 336.13,
                   "forecast_30d": 337.33, "change_percent": 0.36,
                   "p10_30d": 298.12, "p90_30d": 369.41,
                   "model": "TimesFM 2.5 200M", "chat_model": "gemma3:4b" }
{ "type": "token", "text": "Apple's " }
{ "type": "token", "text": "margins " }
{ "type": "done" }
```

The `meta` event lands in well under a second; tokens follow once the model has
read the prompt.

### `POST /api/chat`

Non-streaming equivalent, kept for scripting. Returns the whole answer at once —
expect a multi-minute wait.

---

## System

### `GET /api/models`

Chat models, whether they are pulled, and whether RAM currently allows them.

```jsonc
{ "default": "gemma3:4b",
  "free_ram_gb": 5.4,
  "models": [
    { "name": "gemma3:4b",  "label": "FAST", "ram_gb": 3.9,
      "pulled": true, "loaded": true,  "fits": true },
    { "name": "gemma4:12b", "label": "DEEP", "ram_gb": 8.6,
      "pulled": true, "loaded": false, "fits": false }
  ]}
```

`fits` is `free_ram_gb >= ram_gb + 1.0`, and is always `true` for an
already-loaded model since it needs no additional memory.

### `GET /api/stocks`

Default watchlist and the active model names.

### `GET /api/status`

Health, TimesFM checkpoint, active chat model, cache statistics.
