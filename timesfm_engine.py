"""
TimesFM Engine - forecasting predictions.

Uses the real API of the installed `timesfm` package: the TimesFM 2.5 200M
PyTorch checkpoint, compiled once with a ForecastConfig, then
`forecast(horizon, inputs) -> (point_forecast, quantile_forecast)`.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import timesfm

import cache
from config import TIMESFM_CONFIG

logger = logging.getLogger(__name__)

# Quantile head columns: index 0 is the mean, then the 9 deciles p1..p9.
P10_INDEX = 1
P90_INDEX = 9

# A forecast only changes when a new bar arrives, so cache aggressively.
FORECAST_TTL = 900  # seconds


def _business_days(start: str, count: int) -> list:
    """The next `count` weekdays after `start` (ISO date, or today)."""
    day = datetime.strptime(start, "%Y-%m-%d") if start else datetime.now()
    out = []
    while len(out) < count:
        day += timedelta(days=1)
        if day.weekday() < 5:  # skip Sat/Sun
            out.append(day.strftime("%Y-%m-%d"))
    return out


class TimesFMEngine:
    """TimesFM forecasting wrapper"""

    def __init__(self):
        logger.info("⏳ Loading TimesFM model (%s)...", TIMESFM_CONFIG["checkpoint"])
        try:
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                TIMESFM_CONFIG["checkpoint"]
            )
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=TIMESFM_CONFIG["context_len"],
                    max_horizon=TIMESFM_CONFIG["max_horizon"],
                    normalize_inputs=TIMESFM_CONFIG["normalize_inputs"],
                    use_continuous_quantile_head=TIMESFM_CONFIG["use_quantile_head"],
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            logger.info("✓ TimesFM loaded and compiled!")
            self.forecast_cache = {}
        except Exception as e:
            logger.error("❌ TimesFM error: %s", e)
            raise

    def forecast(self, prices: np.ndarray, ticker: str = None,
                 last_date: str = None) -> dict:
        """Generate a TimesFM forecast for a price series."""
        try:
            prices = np.asarray(prices, dtype=float).ravel()
            if len(prices) < 100:
                return {"error": "Need ≥100 data points", "success": False}

            horizon = TIMESFM_CONFIG["prediction_length"]
            context = prices[-TIMESFM_CONFIG["context_len"]:]

            # Key on the series tail: same input, same forecast.
            cache_key = f"fc:{ticker}:{len(prices)}:{prices[-1]:.4f}:{horizon}"
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("forecast cache hit %s", ticker)
                return cached

            logger.debug("Forecasting %s...", ticker)
            # The model normalizes internally (normalize_inputs=True).
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon, inputs=[context]
            )

            forecast_prices = np.asarray(point_forecast[0], dtype=float)
            quantiles = np.asarray(quantile_forecast[0], dtype=float)

            current_price = float(prices[-1])
            future_price = float(forecast_prices[-1])
            change_pct = ((future_price - current_price) / current_price) * 100

            result = {
                "ticker": ticker,
                "success": True,
                "current_price": current_price,
                "forecast_30d": future_price,
                "change_percent": float(change_pct),
                "min_forecast": float(np.min(forecast_prices)),
                "max_forecast": float(np.max(forecast_prices)),
                "mean_forecast": float(np.mean(forecast_prices)),
                "std_forecast": float(np.std(forecast_prices)),
                "forecast_prices": forecast_prices.tolist(),
                # Model uncertainty band, not a spread of the point path.
                "p10_30d": float(quantiles[-1, P10_INDEX]),
                "p90_30d": float(quantiles[-1, P90_INDEX]),
                "p10_prices": quantiles[:, P10_INDEX].tolist(),
                "p90_prices": quantiles[:, P90_INDEX].tolist(),
                "forecast_dates": _business_days(last_date, horizon),
                "model": "TimesFM 2.5 200M",
                "timestamp": datetime.now().isoformat(),
            }

            if ticker:
                self.forecast_cache[ticker] = result
            cache.put(cache_key, result, FORECAST_TTL)

            return result

        except Exception as e:
            logger.error("Forecast error: %s", e)
            return {"error": str(e), "success": False}


# Global instance
timesfm_engine = None


def get_timesfm_engine():
    global timesfm_engine
    if timesfm_engine is None:
        timesfm_engine = TimesFMEngine()
    return timesfm_engine
