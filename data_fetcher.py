"""
Real-time stock data from Yahoo Finance.

Returns full OHLCV so the dashboard can draw candlesticks, and exposes
Yahoo's symbol-search endpoint for the ticker autocomplete.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import requests
import yfinance as yf

import cache
from config import DATA_FETCH_CONFIG

logger = logging.getLogger(__name__)

# Intraday prices move; history does not. Short TTL on quotes, longer on bars.
QUOTE_TTL = 60          # seconds
SEARCH_TTL = 3600       # symbol names basically never change

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def currency_for(ticker: str) -> str:
    """Symbol to use when printing prices for this ticker."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "₹"
    if t.endswith(".L"):
        return "£"
    if t.endswith(".DE") or t.endswith(".PA"):
        return "€"
    if t.endswith(".T"):
        return "¥"
    return "$"


class RealTimeDataFetcher:
    """Fetch stock data from Yahoo Finance"""

    @staticmethod
    def fetch_stock_data(ticker: str, days_back: int = None) -> dict:
        """Fetch historical OHLCV and current price, cached briefly."""
        if days_back is None:
            days_back = DATA_FETCH_CONFIG["days_history"]

        cache_key = f"bars:{ticker}:{days_back}"
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("cache hit %s", ticker)
            return cached

        try:
            logger.debug("Fetching %s...", ticker)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            data = yf.Ticker(ticker).history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )

            if data.empty:
                logger.warning("No data for %s", ticker)
                return {"error": f"No data for {ticker}", "success": False}

            data = data.dropna(subset=["Close"])
            close = data["Close"].to_numpy(dtype=float)
            open_ = data["Open"].to_numpy(dtype=float)
            high = data["High"].to_numpy(dtype=float)
            low = data["Low"].to_numpy(dtype=float)
            volumes = (
                data["Volume"].to_numpy(dtype=float)
                if "Volume" in data else np.zeros(len(close))
            )
            dates = data.index.strftime("%Y-%m-%d").tolist()

            # 52-week window, not the whole fetched history.
            window = close[-252:] if len(close) >= 252 else close

            prev_close = float(close[-2]) if len(close) > 1 else float(close[-1])
            current = float(close[-1])

            result = {
                "success": True,
                "ticker": ticker,
                "currency": currency_for(ticker),
                "prices": close.tolist(),
                "volumes": volumes.tolist(),
                "dates": dates,
                # OHLCV bars in the shape lightweight-charts expects.
                "candles": [
                    {"time": dates[i], "open": float(open_[i]), "high": float(high[i]),
                     "low": float(low[i]), "close": float(close[i])}
                    for i in range(len(close))
                ],
                "volume_bars": [
                    {"time": dates[i], "value": float(volumes[i]),
                     "up": bool(close[i] >= open_[i])}
                    for i in range(len(close))
                ],
                "count": len(close),
                "current_price": current,
                "previous_close": prev_close,
                "day_change": current - prev_close,
                "day_change_percent": ((current - prev_close) / prev_close * 100)
                                       if prev_close else 0.0,
                "day_open": float(open_[-1]),
                "day_high": float(high[-1]),
                "day_low": float(low[-1]),
                "high_52w": float(np.max(window)),
                "low_52w": float(np.min(window)),
                "avg_volume": float(np.mean(volumes)),
                "last_volume": float(volumes[-1]),
                "last_updated": datetime.now().isoformat(),
            }

            cache.put(cache_key, result, QUOTE_TTL)
            logger.info("✓ %s: %d bars", ticker, len(close))
            return result

        except Exception as e:
            logger.error("Fetch error for %s: %s", ticker, e)
            return {"error": str(e), "success": False}

    @staticmethod
    def search_symbols(query: str, limit: int = 8) -> list:
        """Yahoo Finance symbol search, for the autocomplete box."""
        query = (query or "").strip()
        if len(query) < 1:
            return []

        cache_key = f"search:{query.lower()}:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = requests.get(
                YAHOO_SEARCH_URL,
                params={"q": query, "quotesCount": limit, "newsCount": 0},
                headers={"User-Agent": BROWSER_UA},
                timeout=6,
            )
            resp.raise_for_status()
            quotes = resp.json().get("quotes", [])

            results = []
            for q in quotes:
                symbol = q.get("symbol")
                if not symbol:
                    continue
                results.append({
                    "symbol": symbol,
                    "name": q.get("shortname") or q.get("longname") or "",
                    "exchange": q.get("exchDisp") or "",
                    "type": q.get("quoteType") or "",
                })

            cache.put(cache_key, results, SEARCH_TTL)
            return results

        except Exception as e:
            logger.warning("Symbol search failed for %r: %s", query, e)
            # Offline fallback: match against the built-in list.
            known = DATA_FETCH_CONFIG["default_stocks"] + DATA_FETCH_CONFIG["us_stocks"]
            up = query.upper()
            return [
                {"symbol": t, "name": "", "exchange": "", "type": "EQUITY"}
                for t in known if up in t
            ][:limit]

    @staticmethod
    def fetch_multiple(tickers: list) -> dict:
        """Fetch data for multiple stocks"""
        return {t: RealTimeDataFetcher.fetch_stock_data(t) for t in tickers}


# Global instance
data_fetcher = RealTimeDataFetcher()
