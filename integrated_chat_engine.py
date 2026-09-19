"""
Integrated Chat Engine combining:
- TimesFM forecasts (real predictions)
- Gemma 4 12B LLM (reasoning, context, conversation)
- Yahoo Finance data (real stock data)
"""
import logging
import re
from datetime import datetime

import numpy as np

from config import CHAT_CONFIG, DATA_FETCH_CONFIG
from data_fetcher import RealTimeDataFetcher
from fundamentals import build_digest, fetch_fundamentals, fetch_news
from gemma_llm_engine import get_gemma_engine
from timesfm_engine import get_timesfm_engine

logger = logging.getLogger(__name__)

KNOWN_TICKERS = DATA_FETCH_CONFIG["default_stocks"] + DATA_FETCH_CONFIG["us_stocks"]


class IntegratedChatEngine:
    """
    Chat engine that combines TimesFM forecasts with Gemma 4 LLM reasoning.

    Never hallucinates prices because:
    - All data comes from real sources (Yahoo Finance)
    - All forecasts come from the TimesFM model
    - Gemma is used only for reasoning + conversation
    """

    def __init__(self):
        self.timesfm = get_timesfm_engine()
        self.gemma = get_gemma_engine()
        self.data_fetcher = RealTimeDataFetcher()
        self.conversation_history = []
        self.max_history = CHAT_CONFIG["max_context_messages"]

    def analyze_query(self, query: str) -> dict:
        """Extract a stock ticker from the user query."""
        # Word-boundary match so "TCS" does not fire on "BITCOIN" etc.
        words = set(re.findall(r"[A-Z][A-Z0-9.]*", query.upper()))

        mentioned = None
        for ticker in KNOWN_TICKERS:
            base = ticker.split(".")[0]
            if ticker in words or base in words:
                mentioned = ticker
                break

        return {
            "query": query,
            "ticker": mentioned,
            "timestamp": datetime.now().isoformat(),
        }

    def chat(self, user_message: str, ticker: str = None) -> str:
        """
        Generate a response combining TimesFM + Gemma 4 reasoning.

        Pipeline:
        1. Extract ticker from query
        2. Fetch real stock data (Yahoo Finance)
        3. Generate TimesFM forecast
        4. Pass real data + forecast to Gemma 4
        5. Gemma reasons and responds (over factual data only)
        """
        try:
            analysis = self.analyze_query(user_message)

            # An explicit ticker mentioned in the message wins over the
            # stock currently selected in the sidebar.
            if analysis["ticker"]:
                ticker = analysis["ticker"]

            if not ticker:
                return "📌 Which stock? (e.g., RELIANCE, INFY, AAPL, TCS)"

            # ============================================
            # STEP 1: Fetch REAL data from Yahoo Finance
            # ============================================
            logger.info("📊 Fetching real data for %s...", ticker)
            data_result = self.data_fetcher.fetch_stock_data(ticker)

            if not data_result.get("success"):
                return f"❌ Data unavailable for {ticker}"

            prices = np.array(data_result["prices"])
            current_price = data_result["current_price"]
            cur = data_result["currency"]

            # ============================================
            # STEP 2: Generate REAL forecast from TimesFM
            # ============================================
            logger.info("🧮 Generating TimesFM forecast for %s...", ticker)
            forecast = self.timesfm.forecast(prices, ticker=ticker)

            if not forecast.get("success"):
                return f"❌ Could not forecast {ticker}: {forecast.get('error')}"

            # ============================================
            # STEP 3: Prepare context for Gemma 4
            # ============================================
            forecast_context = f"""
REAL STOCK DATA (Yahoo Finance, currency symbol: {cur}):
- Ticker: {ticker}
- Current Price: {cur}{current_price:,.2f}
- 52-Week High: {cur}{data_result['high_52w']:,.2f}
- 52-Week Low: {cur}{data_result['low_52w']:,.2f}
- Days of History: {data_result['count']} trading days

TIMESFM FORECAST (30 trading days ahead, model: {forecast['model']}):
- Predicted Price: {cur}{forecast['forecast_30d']:,.2f}
- Expected Change: {forecast['change_percent']:+.2f}%
- Path Range: {cur}{forecast['min_forecast']:,.2f} - {cur}{forecast['max_forecast']:,.2f}
- Mean of Path: {cur}{forecast['mean_forecast']:,.2f}
- 80% Confidence Band at day 30: {cur}{forecast['p10_30d']:,.2f} - {cur}{forecast['p90_30d']:,.2f}

USER QUESTION: {user_message}

Using this REAL DATA and REAL FORECAST, provide a reasoned financial analysis.
Every number above is real. Never invent numbers not listed here."""

            # ============================================
            # STEP 4: Get Gemma 4 reasoning
            # ============================================
            logger.info("🤖 Gemma 4 reasoning (this takes minutes on CPU)...")
            gemma_response = self.gemma.chat(forecast_context)

            # ============================================
            # STEP 5: Format final response
            # ============================================
            response = f"""
📊 ANALYSIS: {ticker}

{gemma_response}

───────────────────────────────
📈 FORECAST SUMMARY ({forecast['model']})
───────────────────────────────
Current:      {cur}{current_price:,.2f}
Prediction:   {cur}{forecast['forecast_30d']:,.2f}
Change:       {forecast['change_percent']:+.2f}%
Path range:   {cur}{forecast['min_forecast']:,.2f} - {cur}{forecast['max_forecast']:,.2f}
80% band:     {cur}{forecast['p10_30d']:,.2f} - {cur}{forecast['p90_30d']:,.2f}

✓ Data Sources:
  • Real-time prices: Yahoo Finance
  • 30-day forecast: {forecast['model']}
  • Analysis & reasoning: {CHAT_CONFIG['model']}

⚠️  Disclaimer: Forecasts are machine learning predictions, not guarantees."""

            self.conversation_history.append({
                "user": user_message,
                "assistant": response,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
            })

            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return response.strip()

        except Exception as e:
            logger.error("Chat error: %s", e)
            return f"❌ Error: {e}"

    def build_context(self, user_message: str, ticker: str, depth: str = "full"):
        """Resolve ticker, fetch data, forecast. Returns (context, data, forecast)
        or (error_string, None, None)."""
        analysis = self.analyze_query(user_message)
        if analysis["ticker"]:
            ticker = analysis["ticker"]

        if not ticker:
            return "📌 Which stock? (e.g., RELIANCE, INFY, AAPL, TCS)", None, None

        data_result = self.data_fetcher.fetch_stock_data(ticker)
        if not data_result.get("success"):
            return f"❌ Data unavailable for {ticker}", None, None

        forecast = self.timesfm.forecast(
            np.array(data_result["prices"]),
            ticker=ticker,
            last_date=data_result["dates"][-1],
        )
        if not forecast.get("success"):
            return f"❌ Could not forecast {ticker}: {forecast.get('error')}", None, None

        cur = data_result["currency"]

        # Fundamentals are distilled to ~250 tokens. At this machine's
        # ~9 tok/s prompt-eval speed, the raw statements would cost minutes.
        digest = ""
        want_fund = (depth == "full") and CHAT_CONFIG.get("include_fundamentals", True)
        if want_fund:
            try:
                f = fetch_fundamentals(ticker)
                n = (fetch_news(ticker) if CHAT_CONFIG.get("include_news", True)
                     else {"items": []})
                digest = build_digest(ticker, f, n,
                                      include_news=CHAT_CONFIG.get("include_news", True))
            except Exception as e:
                logger.warning("Fundamentals digest skipped for %s: %s", ticker, e)

        context = f"""MARKET DATA ({ticker}, currency {cur}):
Price {cur}{data_result['current_price']:,.2f} ({data_result['day_change_percent']:+.2f}% today)
52w range {cur}{data_result['low_52w']:,.2f}-{cur}{data_result['high_52w']:,.2f}

TIMESFM 30-DAY PRICE FORECAST (from price history only):
Target {cur}{forecast['forecast_30d']:,.2f} ({forecast['change_percent']:+.2f}%)
80% band {cur}{forecast['p10_30d']:,.2f}-{cur}{forecast['p90_30d']:,.2f}

{digest}

QUESTION: {user_message}

Answer using only the facts above. The TimesFM forecast is based on price
history alone — use the fundamentals to judge whether it looks supported or
contradicted, and say so explicitly."""
        return context, data_result, forecast

    def chat_stream(self, user_message: str, ticker: str = None,
                    depth: str = "full", model: str = None):
        """Streaming variant: yields event dicts for the SSE endpoint."""
        try:
            context, data_result, forecast = self.build_context(user_message, ticker, depth)

            if data_result is None:
                yield {"type": "token", "text": context}
                yield {"type": "done"}
                return

            # Send the hard numbers immediately — they are ready in
            # milliseconds and do not need to wait on the LLM.
            yield {
                "type": "meta",
                "ticker": forecast["ticker"],
                "currency": data_result["currency"],
                "current_price": data_result["current_price"],
                "forecast_30d": forecast["forecast_30d"],
                "change_percent": forecast["change_percent"],
                "p10_30d": forecast["p10_30d"],
                "p90_30d": forecast["p90_30d"],
                "model": forecast["model"],
                "chat_model": self.gemma.resolve(model),
            }

            logger.info("🤖 Streaming Gemma reasoning for %s...", forecast["ticker"])
            parts = []
            for chunk in self.gemma.chat_stream(context, model=model):
                parts.append(chunk)
                yield {"type": "token", "text": chunk}

            self.conversation_history.append({
                "user": user_message,
                "assistant": "".join(parts),
                "ticker": forecast["ticker"],
                "timestamp": datetime.now().isoformat(),
            })
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            yield {"type": "done"}

        except Exception as e:
            logger.error("Stream chat error: %s", e)
            yield {"type": "token", "text": f"❌ Error: {e}"}
            yield {"type": "done"}

    def get_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Global instance
chat_engine = None


def get_chat_engine():
    global chat_engine
    if chat_engine is None:
        chat_engine = IntegratedChatEngine()
    return chat_engine
