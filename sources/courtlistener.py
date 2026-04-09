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


def _search(query: str, page_size: int = 8) -> list[dict]:
    """Run a single CourtListener RECAP search and return parsed results."""
    try:
        resp = requests.get(
            f"{BASE_URL}/search/",
            params={"q": query, "type": "r", "order_by": "score desc", "page_size": page_size},
            headers=_get_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException:
        return []


def _parse_result(result: dict, label: str) -> dict:
    case_name = result.get("caseName") or result.get("case_name_full") or ""
    docket_path = result.get("docket_absolute_url", "")
    docket_id = result.get("docket_id")
    case_entry = {
        "case_name": case_name,
        "court": result.get("court") or result.get("court_citation_string") or "",
        "date_filed": result.get("dateFiled") or "",
        "docket_url": f"{SITE_URL}{docket_path}" if docket_path else "",
        "label": label,  # "Perplexity" or "Precedent"
        "recent_filings": [],
    }
    is_perplexity = "perplexity" in case_name.lower()
    if is_perplexity and docket_id:
        case_entry["recent_filings"] = _fetch_docket_entries(docket_id)
    return case_entry


def get_court_data(topic: str) -> dict:
    """
    Search CourtListener for two sets of cases:
    1. Perplexity-specific cases for *topic*
    2. Precedent-setting AI copyright cases (OpenAI, Getty, Thomson Reuters, etc.)

    Returns:
        {
            "perplexity_cases": [...],
            "precedent_cases": [...],
            "total_found": int
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=2) as ex:
        pplx_future = ex.submit(_search, f"Perplexity AI {topic}", 6)
        precedent_future = ex.submit(_search, "OpenAI OR Getty OR Thomson Reuters copyright artificial intelligence", 6)
        pplx_results = pplx_future.result()
        precedent_results = precedent_future.result()

    # Deduplicate precedent results against Perplexity results
    pplx_names = {r.get("caseName", "") for r in pplx_results}
    precedent_results = [r for r in precedent_results if r.get("caseName", "") not in pplx_names]

    perplexity_cases = [_parse_result(r, "Perplexity") for r in pplx_results]
    precedent_cases = [_parse_result(r, "Precedent") for r in precedent_results]

    return {
        "perplexity_cases": perplexity_cases,
        "precedent_cases": precedent_cases,
        "total_found": len(perplexity_cases) + len(precedent_cases),
    }


if __name__ == "__main__":
    import json

    result = get_court_data("Perplexity AI copyright")
    print(json.dumps(result, indent=2))
