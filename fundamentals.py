"""
Company fundamentals: profile, valuation, financial statements, analyst
coverage, ownership, earnings calendar and news.

Design note — why this module *distills* rather than dumps:

    Measured on this machine, Gemma 4 12B reads a prompt at ~8-10 tokens/sec.
    A raw balance sheet is ~2000 tokens, which would cost ~4 minutes of
    prompt evaluation before the model emits a single word.

    So the full data goes to the UI (free, instant) and the LLM receives a
    compact digest of derived signals (~300 tokens). Ratios and trends are
    computed here in Python, where they are exact and cost nothing, instead
    of asking the LLM to do arithmetic on raw statements.
"""
import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import cache
from data_fetcher import currency_for

logger = logging.getLogger(__name__)

FUNDAMENTALS_TTL = 6 * 3600   # statements change quarterly
NEWS_TTL = 20 * 60            # headlines move faster

# Yahoo rate-limits the .info endpoint and then returns an empty dict.
# Caching that for 6 hours would poison the ticker for the rest of the day,
# so a thin result gets a short TTL and is retried instead.
PARTIAL_TTL = 300
INFO_RETRIES = 3

# Statement rows we care about, in display order.
INCOME_ROWS = [
    "Total Revenue", "Gross Profit", "Operating Income", "EBITDA",
    "Net Income", "Basic EPS", "Diluted EPS",
]
BALANCE_ROWS = [
    "Total Assets", "Total Liabilities Net Minority Interest", "Total Debt",
    "Net Debt", "Cash And Cash Equivalents", "Working Capital",
    "Stockholders Equity", "Tangible Book Value", "Invested Capital",
]
CASHFLOW_ROWS = [
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
    "Capital Expenditure", "Free Cash Flow", "End Cash Position",
]


def _num(v):
    """Coerce to a JSON-safe float, or None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _frame_to_series(df, wanted_rows, max_periods=4):
    """Pull selected rows out of a yfinance statement frame.

    Returns {"periods": [...], "rows": {label: [v per period]}}.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return {"periods": [], "rows": {}}

    cols = list(df.columns)[:max_periods]
    periods = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)[:10]
               for c in cols]

    rows = {}
    for label in wanted_rows:
        if label in df.index:
            rows[label] = [_num(df.at[label, c]) for c in cols]
    return {"periods": periods, "rows": rows}


def _pct_change(series):
    """First-vs-last percent change of a newest-first series."""
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or not vals[-1]:
        return None
    newest, oldest = vals[0], vals[-1]
    if oldest == 0:
        return None
    return (newest - oldest) / abs(oldest) * 100


def _yoy(series):
    """Newest period vs the same period a year earlier (4 quarters back)."""
    if len(series) < 5:
        return None
    a, b = series[0], series[4]
    if a is None or b is None or not b:
        return None
    return (a - b) / abs(b) * 100


def _get_info(t, ticker: str) -> dict:
    """yfinance .info with retries, falling back to fast_info.

    Yahoo throttles this endpoint; an empty dict means "ask again later",
    not "this company has no fundamentals".
    """
    info = {}
    for attempt in range(INFO_RETRIES):
        try:
            info = t.info or {}
        except Exception as e:
            logger.debug("%s .info attempt %d failed: %s", ticker, attempt + 1, e)
            info = {}
        if info.get("marketCap") is not None or info.get("trailingPE") is not None:
            return info
        if attempt < INFO_RETRIES - 1:
            time.sleep(0.8 * (attempt + 1))

    # Last resort: fast_info is a lighter endpoint that is throttled separately.
    try:
        fi = t.fast_info
        for src, dst in (("market_cap", "marketCap"), ("shares", "sharesOutstanding")):
            v = getattr(fi, src, None)
            if v is not None:
                info.setdefault(dst, v)
        if info:
            logger.info("%s: .info throttled, used fast_info fallback", ticker)
    except Exception:
        pass
    return info


def _is_thin(result: dict) -> bool:
    """True when Yahoo gave us essentially nothing worth caching for hours."""
    v = result.get("valuation", {})
    h = result.get("health", {})
    has_any = any(v.get(k) is not None for k in ("market_cap", "trailing_pe", "price_to_book")) \
        or any(h.get(k) is not None for k in ("total_revenue", "profit_margin"))
    has_statements = any(
        st.get("periods") for st in result.get("statements", {}).values())
    return not (has_any or has_statements)


def fetch_fundamentals(ticker: str) -> dict:
    """Full fundamentals bundle for the UI. Cached for 6 hours."""
    key = f"fund:{ticker}"
    hit = cache.get(key)
    if hit is not None:
        return hit

    try:
        t = yf.Ticker(ticker)
        info = _get_info(t, ticker)
        cur = currency_for(ticker)

        profile = {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website"),
            "summary": (info.get("longBusinessSummary") or "")[:600],
            "currency": cur,
            "exchange": info.get("exchange"),
        }

        valuation = {
            "market_cap": _num(info.get("marketCap")),
            "enterprise_value": _num(info.get("enterpriseValue")),
            "trailing_pe": _num(info.get("trailingPE")),
            "forward_pe": _num(info.get("forwardPE")),
            "peg_ratio": _num(info.get("pegRatio")),
            "price_to_book": _num(info.get("priceToBook")),
            "price_to_sales": _num(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": _num(info.get("enterpriseToEbitda")),
            "dividend_yield": _num(info.get("dividendYield")),
            "beta": _num(info.get("beta")),
        }

        health = {
            "profit_margin": _num(info.get("profitMargins")),
            "operating_margin": _num(info.get("operatingMargins")),
            "gross_margin": _num(info.get("grossMargins")),
            "return_on_equity": _num(info.get("returnOnEquity")),
            "return_on_assets": _num(info.get("returnOnAssets")),
            "debt_to_equity": _num(info.get("debtToEquity")),
            "current_ratio": _num(info.get("currentRatio")),
            "quick_ratio": _num(info.get("quickRatio")),
            "total_revenue": _num(info.get("totalRevenue")),
            "total_debt": _num(info.get("totalDebt")),
            "total_cash": _num(info.get("totalCash")),
            "free_cashflow": _num(info.get("freeCashflow")),
            "operating_cashflow": _num(info.get("operatingCashflow")),
            "revenue_growth": _num(info.get("revenueGrowth")),
            "earnings_growth": _num(info.get("earningsGrowth")),
        }

        analyst = {
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": _num(info.get("recommendationMean")),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "target_mean": _num(info.get("targetMeanPrice")),
            "target_high": _num(info.get("targetHighPrice")),
            "target_low": _num(info.get("targetLowPrice")),
        }

        # --- statements -------------------------------------------------
        statements = {}
        for name, attr, rows in (
            ("quarterly_income", "quarterly_income_stmt", INCOME_ROWS),
            ("annual_income", "income_stmt", INCOME_ROWS),
            ("quarterly_balance", "quarterly_balance_sheet", BALANCE_ROWS),
            ("annual_balance", "balance_sheet", BALANCE_ROWS),
            ("quarterly_cashflow", "quarterly_cashflow", CASHFLOW_ROWS),
            ("annual_cashflow", "cashflow", CASHFLOW_ROWS),
        ):
            try:
                statements[name] = _frame_to_series(getattr(t, attr, None), rows)
            except Exception as e:
                logger.debug("%s %s unavailable: %s", ticker, name, e)
                statements[name] = {"periods": [], "rows": {}}

        # --- ownership --------------------------------------------------
        holders = {"major": {}, "institutional": []}
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                col = mh.columns[0]
                holders["major"] = {str(i): _num(mh.at[i, col]) for i in mh.index}
        except Exception:
            pass
        try:
            ih = t.institutional_holders
            if ih is not None and not ih.empty:
                for _, r in ih.head(8).iterrows():
                    holders["institutional"].append({
                        "holder": str(r.get("Holder", "")),
                        "shares": _num(r.get("Shares")),
                        "pct_held": _num(r.get("pctHeld")),
                        "value": _num(r.get("Value")),
                    })
        except Exception:
            pass

        # --- earnings calendar ------------------------------------------
        earnings = {"past": [], "next_date": None, "days_to_next": None}
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                now = pd.Timestamp.now(tz=ed.index.tz)
                for idx, r in ed.iterrows():
                    rec = {
                        "date": idx.strftime("%Y-%m-%d"),
                        "eps_estimate": _num(r.get("EPS Estimate")),
                        "eps_reported": _num(r.get("Reported EPS")),
                        "surprise_pct": _num(r.get("Surprise(%)")),
                    }
                    if idx <= now and len(earnings["past"]) < 6:
                        earnings["past"].append(rec)
                upcoming = [i for i in ed.index if i > now]
                if upcoming:
                    nxt = min(upcoming)
                    earnings["next_date"] = nxt.strftime("%Y-%m-%d")
                    earnings["days_to_next"] = int((nxt - now).days)
        except Exception as e:
            logger.debug("%s earnings dates unavailable: %s", ticker, e)

        result = {
            "success": True,
            "ticker": ticker,
            "profile": profile,
            "valuation": valuation,
            "health": health,
            "analyst": analyst,
            "statements": statements,
            "holders": holders,
            "earnings": earnings,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        if _is_thin(result):
            # Throttled or genuinely empty — retry soon rather than in 6 hours.
            result["partial"] = True
            cache.put(key, result, PARTIAL_TTL)
            logger.warning("%s: fundamentals came back thin; will retry in %ds",
                           ticker, PARTIAL_TTL)
        else:
            result["partial"] = False
            cache.put(key, result, FUNDAMENTALS_TTL)
            logger.info("✓ fundamentals %s", ticker)
        return result

    except Exception as e:
        logger.error("Fundamentals error for %s: %s", ticker, e)
        return {"success": False, "error": str(e), "ticker": ticker}


def fetch_news(ticker: str, limit: int = 8) -> dict:
    """Recent headlines. Cached for 20 minutes."""
    key = f"news:{ticker}:{limit}"
    hit = cache.get(key)
    if hit is not None:
        return hit

    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        for n in raw[:limit]:
            # yfinance changed shape: payload now nests under "content".
            c = n.get("content", n) or {}
            pub = c.get("pubDate") or c.get("displayTime") or ""
            provider = c.get("provider") or {}
            url = ""
            for k in ("canonicalUrl", "clickThroughUrl"):
                if isinstance(c.get(k), dict):
                    url = c[k].get("url", "")
                    if url:
                        break
            items.append({
                "title": c.get("title") or "",
                "publisher": provider.get("displayName") if isinstance(provider, dict) else "",
                "published": str(pub)[:19],
                "url": url,
                "summary": (c.get("summary") or "")[:280],
            })
    except Exception as e:
        logger.warning("News unavailable for %s: %s", ticker, e)

    result = {"success": True, "ticker": ticker, "items": items}
    cache.put(key, result, NEWS_TTL)
    return result


# ======================================================================
# LLM digest — the token-budgeted view
# ======================================================================

def _fmt_big(v, cur=""):
    if v is None:
        return "n/a"
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{cur}{v/div:.2f}{suf}"
    return f"{cur}{v:.2f}"


def _fmt_pct(v, already_pct=False):
    if v is None:
        return "n/a"
    return f"{v if already_pct else v*100:+.1f}%"


def build_digest(ticker: str, fundamentals: dict, news: dict,
                 include_news: bool = True) -> str:
    """A compact, high-signal fundamentals block for the LLM prompt.

    Target: roughly 250-350 tokens. Everything here is either a directly
    reported figure or an exact ratio computed in Python — the model is
    never asked to do arithmetic on raw statements.
    """
    if not fundamentals.get("success"):
        return ""

    p = fundamentals["profile"]
    v = fundamentals["valuation"]
    h = fundamentals["health"]
    a = fundamentals["analyst"]
    e = fundamentals["earnings"]
    cur = p.get("currency", "")

    lines = [f"COMPANY: {p['name']} — {p.get('sector') or 'n/a'} / {p.get('industry') or 'n/a'}"]

    lines.append(
        "VALUATION: mktcap {} | P/E {} (fwd {}) | P/B {} | EV/EBITDA {} | beta {}".format(
            _fmt_big(v["market_cap"], cur),
            f"{v['trailing_pe']:.1f}" if v["trailing_pe"] else "n/a",
            f"{v['forward_pe']:.1f}" if v["forward_pe"] else "n/a",
            f"{v['price_to_book']:.2f}" if v["price_to_book"] else "n/a",
            f"{v['ev_to_ebitda']:.1f}" if v["ev_to_ebitda"] else "n/a",
            f"{v['beta']:.2f}" if v["beta"] else "n/a",
        )
    )

    lines.append(
        "PROFITABILITY: net margin {} | op margin {} | ROE {} | rev growth {} | earnings growth {}".format(
            _fmt_pct(h["profit_margin"]), _fmt_pct(h["operating_margin"]),
            _fmt_pct(h["return_on_equity"]), _fmt_pct(h["revenue_growth"]),
            _fmt_pct(h["earnings_growth"]),
        )
    )

    lines.append(
        "BALANCE SHEET: revenue {} | debt {} | cash {} | D/E {} | current ratio {}".format(
            _fmt_big(h["total_revenue"], cur), _fmt_big(h["total_debt"], cur),
            _fmt_big(h["total_cash"], cur),
            f"{h['debt_to_equity']:.1f}" if h["debt_to_equity"] else "n/a",
            f"{h['current_ratio']:.2f}" if h["current_ratio"] else "n/a",
        )
    )

    # Quarterly trend, computed exactly rather than left to the model.
    qi = fundamentals["statements"].get("quarterly_income", {})
    rows = qi.get("rows", {})
    if rows.get("Total Revenue"):
        rev = rows["Total Revenue"]
        ni = rows.get("Net Income", [])
        parts = [f"latest Q revenue {_fmt_big(rev[0], cur)}"]
        qoq = _pct_change(rev[:2][::-1]) if len(rev) >= 2 else None
        if len(rev) >= 2 and rev[1]:
            parts.append(f"QoQ {(rev[0]-rev[1])/abs(rev[1])*100:+.1f}%")
        yoy = _yoy(rev)
        if yoy is not None:
            parts.append(f"YoY {yoy:+.1f}%")
        if ni and ni[0] is not None:
            parts.append(f"net income {_fmt_big(ni[0], cur)}")
            if len(ni) >= 2 and ni[1]:
                parts.append(f"NI QoQ {(ni[0]-ni[1])/abs(ni[1])*100:+.1f}%")
        lines.append("LATEST QUARTER (" + (qi.get("periods") or ["?"])[0] + "): " + ", ".join(parts))

    if a.get("target_mean") or a.get("recommendation"):
        lines.append(
            "ANALYSTS: {} ({} covering) | target mean {} (low {} / high {})".format(
                (a.get("recommendation") or "n/a").replace("_", " "),
                a.get("analyst_count") or "?",
                _fmt_big(a["target_mean"], cur) if a["target_mean"] else "n/a",
                _fmt_big(a["target_low"], cur) if a["target_low"] else "n/a",
                _fmt_big(a["target_high"], cur) if a["target_high"] else "n/a",
            )
        )

    # Earnings inside the 30-day window is the single most decision-relevant
    # fundamental fact for a 30-day forecast.
    if e.get("next_date"):
        d = e.get("days_to_next")
        flag = " ** INSIDE THE 30-DAY FORECAST WINDOW **" if d is not None and d <= 30 else ""
        lines.append(f"NEXT EARNINGS: {e['next_date']} ({d} days away){flag}")
    past = e.get("past") or []
    surprises = [f"{r['surprise_pct']:+.0f}%" for r in past[:3]
                 if r.get("surprise_pct") is not None]
    if surprises:
        lines.append("RECENT EPS SURPRISES: " + ", ".join(surprises))

    if include_news and news.get("items"):
        heads = [n["title"] for n in news["items"][:4] if n.get("title")]
        if heads:
            lines.append("RECENT HEADLINES:")
            lines += [f"- {t[:110]}" for t in heads]

    return "\n".join(lines)
