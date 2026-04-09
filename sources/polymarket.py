"""
Polymarket integration for AI policy prediction markets.

Uses the Gamma API (https://gamma-api.polymarket.com) to search for
active prediction markets relevant to AI policy and regulation.

Key findings from API exploration:
- The `search` param on /markets does NOT filter by text — it returns
  default-sorted results regardless of the search value.
- Correct approach: paginate /markets?active=true and filter client-side
  by matching keywords against the `question` field.
- Market URL: https://polymarket.com/event/{event_slug} where event_slug
  comes from the nested `events[0].slug` field.
- Probability for a Yes/No market: outcomePrices[0] (the "Yes" price).
- Volume is in `volumeNum` (float, USD).
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_BASE = "https://polymarket.com/event"

# Maximum pages to paginate when searching (200 markets/page).
# 20 pages = 4000 markets — sufficient to cover all active markets.
MAX_PAGES = 20
PAGE_SIZE = 200

# Minimum volume (USD) to include a market in results.
MIN_VOLUME = 5000


def _fetch_all_active_markets(session: requests.Session) -> list[dict]:
    """Paginate /markets?active=true and return all active markets."""
    all_markets: list[dict] = []
    offset = 0

    for _ in range(MAX_PAGES):
        try:
            resp = session.get(
                f"{GAMMA_BASE}/markets",
                params={"active": "true", "limit": PAGE_SIZE, "offset": offset},
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Polymarket API request failed at offset %d: %s", offset, exc)
            break

        batch = resp.json()
        if not batch:
            break

        all_markets.extend(batch)
        offset += PAGE_SIZE

        if len(batch) < PAGE_SIZE:
            # Last page
            break

    logger.info("Fetched %d active markets from Polymarket", len(all_markets))
    return all_markets


def _market_matches(market: dict, keywords: list[str]) -> Optional[str]:
    """
    Return the first keyword that matches the market's question text,
    or None if no keyword matches.
    Case-insensitive substring match against the question field.
    """
    question_lower = market.get("question", "").lower()
    for kw in keywords:
        if kw.lower() in question_lower:
            return kw
    return None


def _parse_probability(market: dict) -> Optional[float]:
    """
    Extract the Yes-outcome probability from outcomePrices.
    outcomePrices is a JSON-encoded string like '["0.73", "0.27"]'.
    Returns a float 0–1 for the first (Yes) outcome, or None on failure.
    """
    import json as _json

    raw = market.get("outcomePrices")
    if not raw:
        return None
    try:
        prices = _json.loads(raw) if isinstance(raw, str) else raw
        return round(float(prices[0]), 4)
    except (ValueError, IndexError, TypeError):
        return None


def _build_url(market: dict) -> str:
    """
    Construct the Polymarket URL for a market.
    Prefers the parent event slug; falls back to the market's own slug.
    """
    events = market.get("events") or []
    if events and events[0].get("slug"):
        return f"{POLYMARKET_BASE}/{events[0]['slug']}"
    return f"{POLYMARKET_BASE}/{market.get('slug', market.get('id', ''))}"


def get_polymarket_data(keywords: list[str]) -> dict:
    """
    Search Polymarket prediction markets for topics relevant to the
    provided keywords.

    Parameters
    ----------
    keywords:
        List of keyword strings to match against market question text.
        Example: ["AI regulation", "EU AI Act", "AI copyright"]

    Returns
    -------
    dict with keys:
        markets          — list of matched market dicts (see schema below)
        keywords_searched — the input keywords list
        total_found      — number of markets returned

    Each market dict:
        question    — str: the market question
        probability — float 0–1: implied probability of Yes outcome
        volume      — float: total USD volume traded
        url         — str: link to Polymarket market page
        end_date    — str: resolution date (YYYY-MM-DD)
        relevant_to — str: the keyword that matched this market
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "ai-policy-monitor/1.0"})

    all_markets = _fetch_all_active_markets(session)

    matched: list[dict] = []
    seen_ids: set[str] = set()

    for market in all_markets:
        market_id = str(market.get("id", ""))
        if market_id in seen_ids:
            continue

        volume = market.get("volumeNum") or float(market.get("volume") or 0)
        if volume < MIN_VOLUME:
            continue

        matched_kw = _market_matches(market, keywords)
        if matched_kw is None:
            continue

        seen_ids.add(market_id)

        probability = _parse_probability(market)
        end_date = (market.get("endDateIso") or market.get("endDate", "")[:10])

        matched.append(
            {
                "question": market.get("question", ""),
                "probability": probability,
                "volume": round(volume, 2),
                "url": _build_url(market),
                "end_date": end_date,
                "relevant_to": matched_kw,
            }
        )

    # Sort by volume descending so highest-signal markets appear first.
    matched.sort(key=lambda m: m["volume"], reverse=True)

    return {
        "markets": matched,
        "keywords_searched": keywords,
        "total_found": len(matched),
    }


# ---------------------------------------------------------------------------
# Quick test — run directly: python sources/polymarket.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ai_policy_keywords = [
        "AI regulation",
        "AI copyright",
        "artificial intelligence law",
        "EU AI Act",
        "OpenAI",
        "Anthropic",
        "AI model",
        "AI safety",
        "GPT",
        "DeepSeek",
        "chatgpt",
        "AI ban",
        "AI policy",
        "generative AI",
        "machine learning",
        "LLM",
    ]

    print("Searching Polymarket for AI policy markets...")
    result = get_polymarket_data(ai_policy_keywords)

    print(f"\nKeywords searched: {result['keywords_searched']}")
    print(f"Total markets found (volume > ${MIN_VOLUME:,}): {result['total_found']}\n")

    for i, m in enumerate(result["markets"], 1):
        prob_str = f"{m['probability']:.1%}" if m["probability"] is not None else "N/A"
        print(
            f"{i:2}. [{m['relevant_to']}] {m['question']}\n"
            f"    Probability: {prob_str} | Volume: ${m['volume']:,.0f} | "
            f"Ends: {m['end_date']}\n"
            f"    {m['url']}\n"
        )
