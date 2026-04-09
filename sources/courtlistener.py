"""
CourtListener integration for AI policy research pipeline.

Public search endpoint works without auth (type=r = RECAP/federal cases).
Docket entries require an API token (set COURTLISTENER_TOKEN in .env).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_URL = "https://www.courtlistener.com/api/rest/v4"
SITE_URL = "https://www.courtlistener.com"


def _get_headers() -> dict:
    token = os.getenv("COURTLISTENER_TOKEN")
    if token:
        return {"Authorization": f"Token {token}"}
    return {}


def _fetch_docket_entries(docket_id: int, limit: int = 5) -> list[dict]:
    """Fetch recent docket entries for a given docket ID. Requires auth token."""
    headers = _get_headers()
    if not headers:
        return []

    url = f"{BASE_URL}/docket-entries/"
    params = {
        "docket": docket_id,
        "order_by": "-date_filed",
        "page_size": limit,
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entries = []
        for entry in data.get("results", []):
            # description may come from recap_documents or the entry itself
            desc = entry.get("description") or ""
            if not desc:
                docs = entry.get("recap_documents", [])
                if docs:
                    desc = docs[0].get("description", "")
            entries.append({
                "date": entry.get("date_filed", ""),
                "description": desc,
            })
        return entries
    except requests.RequestException:
        return []


def get_court_data(topic: str) -> dict:
    """
    Search CourtListener for federal RECAP cases related to *topic*.

    For any Perplexity-specific cases found, also fetches recent docket entries
    (requires COURTLISTENER_TOKEN in .env; silently skipped when absent).

    Returns:
        {
            "cases": [
                {
                    "case_name": str,
                    "court": str,
                    "date_filed": str,          # YYYY-MM-DD
                    "docket_url": str,
                    "snippet": str,
                    "recent_filings": [...]      # only for Perplexity cases
                }
            ],
            "query_used": str,
            "total_found": int
        }
    """
    url = f"{BASE_URL}/search/"
    params = {
        "q": topic,
        "type": "r",          # RECAP = federal court filings
        "order_by": "score desc",
        "page_size": 10,
    }

    try:
        resp = requests.get(url, params=params, headers=_get_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {"cases": [], "query_used": topic, "total_found": 0, "error": str(exc)}

    total = data.get("count", 0)
    cases = []

    for result in data.get("results", []):
        case_name = result.get("caseName") or result.get("case_name_full") or ""
        court = result.get("court") or result.get("court_citation_string") or ""
        date_filed = result.get("dateFiled") or ""
        docket_path = result.get("docket_absolute_url", "")
        docket_url = f"{SITE_URL}{docket_path}" if docket_path else ""
        docket_id = result.get("docket_id")

        # Build a snippet from the first recap document description
        snippet = ""
        recap_docs = result.get("recap_documents", [])
        if recap_docs:
            snippet = recap_docs[0].get("description", "")

        case_entry: dict = {
            "case_name": case_name,
            "court": court,
            "date_filed": date_filed,
            "docket_url": docket_url,
            "snippet": snippet,
        }

        # Pull recent docket entries only for Perplexity-related cases
        is_perplexity = "perplexity" in case_name.lower() or any(
            "perplexity" in str(p).lower() for p in result.get("party", [])
        )
        if is_perplexity and docket_id:
            filings = _fetch_docket_entries(docket_id)
            case_entry["recent_filings"] = filings

        cases.append(case_entry)

    return {
        "cases": cases,
        "query_used": topic,
        "total_found": total,
    }


if __name__ == "__main__":
    import json

    result = get_court_data("Perplexity AI copyright")
    print(json.dumps(result, indent=2))
