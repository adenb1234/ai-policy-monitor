"""
Congress.gov API integration for AI policy research pipeline.

Uses the Congress.gov v3 REST API.
Set CONGRESS_API_KEY in .env for full access; falls back to DEMO_KEY (rate-limited).

Key observations from the live API:
- The /bill endpoint accepts a `query` param but with DEMO_KEY it appears to return
  all bills sorted by updateDate rather than filtered results. A real API key returns
  proper full-text search results.
- Bill list items include: congress, type, number, title, originChamber, latestAction,
  updateDate, url (pointing to the detail endpoint).
- Bill detail adds: sponsors, introducedDate, policyArea, cosponsors (count ref),
  summaries, subjects, committees, laws, legislationUrl.
- Cosponsors are fetched from a separate /cosponsors sub-endpoint.
- Status is inferred from latestAction.text since the API does not expose a single
  "status" field.
"""

import os
import time
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_URL = "https://api.congress.gov/v3"
CONGRESS_URL = "https://congress.gov"

# Ordinal suffix lookup for congress number display
_ORDINAL = {1: "st", 2: "nd", 3: "rd"}


def _api_key() -> str:
    return os.getenv("CONGRESS_API_KEY", "DEMO_KEY")


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = _ORDINAL.get(n % 10, "th")
    return f"{n}{suffix}"


def _get(path: str, params: Optional[dict] = None, retries: int = 2) -> dict:
    """Make a GET request to the Congress.gov API, returning parsed JSON."""
    url = f"{BASE_URL}{path}"
    p = {"format": "json", "api_key": _api_key()}
    if params:
        p.update(params)

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=p, timeout=15)
            body = resp.json()
            # Congress.gov returns rate-limit errors as 429 or as a 200 with error body
            is_rate_limited = resp.status_code == 429 or (
                isinstance(body.get("error"), dict)
                and body["error"].get("code") == "OVER_RATE_LIMIT"
            )
            if is_rate_limited:
                if attempt < retries:
                    time.sleep(8 * (attempt + 1))
                    continue
                return {"_rate_limited": True}
            resp.raise_for_status()
            return body
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(2)
                continue
            return {"_error": str(exc)}
    return {}


def _infer_status(latest_action_text: str) -> str:
    """
    Derive a human-readable status string from a bill's latest action text.
    Congress.gov does not provide an explicit status field.
    """
    text = (latest_action_text or "").lower()
    if "became public law" in text or "signed by president" in text:
        return "Signed into Law"
    if "vetoed" in text:
        return "Vetoed"
    if "passed senate" in text and "passed house" in text:
        return "Passed Both Chambers"
    if "passed senate" in text:
        return "Passed Senate"
    if "passed house" in text or "on passage" in text:
        return "Passed House"
    if "conference report" in text:
        return "In Conference"
    if "placed on calendar" in text:
        return "On Senate Calendar"
    if "committee of the whole" in text:
        return "On House Floor"
    if "referred to" in text or "committee" in text:
        return "In Committee"
    if "introduced" in text:
        return "Introduced"
    return "Active"


def _format_sponsor(sponsor: dict) -> str:
    """Format a sponsor dict into 'Rep./Sen. Full Name (Party-State)'."""
    if not sponsor:
        return "Unknown"
    full_name = sponsor.get("fullName", "")
    if full_name:
        # fullName is already formatted as "Rep. Last, First [Party-State-District]"
        # Normalize to "Rep. First Last (Party-State)"
        return full_name.replace("[", "(").replace("]", ")")
    chamber_prefix = "Sen." if sponsor.get("chamber") == "Senate" else "Rep."
    first = sponsor.get("firstName", "")
    last = sponsor.get("lastName", "")
    party = sponsor.get("party", "")
    state = sponsor.get("state", "")
    name = f"{first} {last}".strip()
    suffix = f" ({party}-{state})" if party and state else ""
    return f"{chamber_prefix} {name}{suffix}"


def _build_bill_url(congress: int, bill_type: str, number: str) -> str:
    """Build a human-facing congress.gov URL for a bill."""
    type_map = {
        "HR": "house-bill",
        "S": "senate-bill",
        "HJRES": "house-joint-resolution",
        "SJRES": "senate-joint-resolution",
        "HCONRES": "house-concurrent-resolution",
        "SCONRES": "senate-concurrent-resolution",
        "HRES": "house-resolution",
        "SRES": "senate-resolution",
    }
    path_type = type_map.get(bill_type.upper(), bill_type.lower())
    return f"{CONGRESS_URL}/bill/{congress}th-congress/{path_type}/{number}"


def _fetch_bill_detail(congress: int, bill_type: str, number: str) -> dict:
    """Fetch full bill detail including sponsor and cosponsor count."""
    data = _get(f"/bill/{congress}/{bill_type.lower()}/{number}")
    return data.get("bill", {})


def _fetch_cosponsor_count(congress: int, bill_type: str, number: str) -> int:
    """Return total cosponsor count via the cosponsors sub-endpoint."""
    data = _get(
        f"/bill/{congress}/{bill_type.lower()}/{number}/cosponsors",
        params={"limit": 1},
    )
    return data.get("pagination", {}).get("count", 0)


def get_congressional_data(topic: str) -> dict:
    """
    Search Congress.gov for legislation relevant to *topic*.

    Fetches the top 10 bills matching the query, then enriches the most
    relevant results with sponsor info and cosponsor counts from the detail
    endpoints.

    Args:
        topic: Free-text search string (e.g. "artificial intelligence regulation").

    Returns:
        {
            "bills": [
                {
                    "title": str,
                    "bill_number": str,        # e.g. "HR 1234"
                    "congress": str,           # e.g. "119th"
                    "sponsor": str,            # e.g. "Rep. Jane Smith (D-CA)"
                    "cosponsor_count": int,
                    "introduced_date": str,    # YYYY-MM-DD
                    "latest_action": str,
                    "latest_action_date": str, # YYYY-MM-DD
                    "status": str,
                    "url": str,
                }
            ],
            "query_used": str,
            "total_found": int,
        }
    """
    params = {
        "query": topic,
        "sort": "updateDate+desc",
        "limit": 10,
    }
    data = _get("/bill", params=params)

    if not data or "bills" not in data:
        if data.get("_rate_limited"):
            err = "Congress.gov API rate limit exceeded. Use a real CONGRESS_API_KEY or wait before retrying."
        elif data.get("_error"):
            err = data["_error"]
        else:
            err = "No response from Congress.gov API"
        return {
            "bills": [],
            "query_used": topic,
            "total_found": 0,
            "error": err,
        }

    raw_bills = data.get("bills", [])
    total = data.get("pagination", {}).get("count", 0)

    bills = []
    for raw in raw_bills:
        congress_num = raw.get("congress", 0)
        bill_type = raw.get("type", "")
        number = raw.get("number", "")
        title = raw.get("title", "")
        latest_action = raw.get("latestAction", {})
        latest_action_text = latest_action.get("text", "")
        latest_action_date = latest_action.get("actionDate", "")

        # Fetch detail for sponsor / introduced date / cosponsor count.
        # Rate-limit courtesy: only enrich first 5 results.
        sponsor = "Unknown"
        introduced_date = ""
        cosponsor_count = 0

        if len(bills) < 5 and congress_num and bill_type and number:
            detail = _fetch_bill_detail(congress_num, bill_type, number)
            if detail:
                sponsors = detail.get("sponsors", [])
                if sponsors:
                    sponsor = _format_sponsor(sponsors[0])
                introduced_date = detail.get("introducedDate", "")
                # Cosponsor count comes from detail's cosponsors sub-resource ref
                cosponsors_ref = detail.get("cosponsors", {})
                cosponsor_count = cosponsors_ref.get("count", 0)

            # Small delay to respect rate limits
            time.sleep(0.5)

        bill_number = f"{bill_type} {number}" if bill_type and number else number
        congress_label = _ordinal(congress_num) if congress_num else str(congress_num)
        url = _build_bill_url(congress_num, bill_type, number) if congress_num else ""

        bills.append(
            {
                "title": title,
                "bill_number": bill_number,
                "congress": congress_label,
                "sponsor": sponsor,
                "cosponsor_count": cosponsor_count,
                "introduced_date": introduced_date,
                "latest_action": latest_action_text,
                "latest_action_date": latest_action_date,
                "status": _infer_status(latest_action_text),
                "url": url,
            }
        )

    return {
        "bills": bills,
        "query_used": topic,
        "total_found": total,
    }


if __name__ == "__main__":
    import json

    queries = [
        "artificial intelligence",
        "AI regulation",
        "machine learning copyright",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q!r}")
        print("=" * 60)
        result = get_congressional_data(q)
        print(f"Total found: {result['total_found']}")
        for bill in result["bills"][:3]:
            print(f"\n  {bill['bill_number']} ({bill['congress']})")
            print(f"  Title:   {bill['title'][:70]}")
            print(f"  Sponsor: {bill['sponsor']}")
            print(f"  Status:  {bill['status']}")
            print(f"  Action:  {bill['latest_action'][:60]}")
            print(f"  Date:    {bill['latest_action_date']}")
            print(f"  Cosponsors: {bill['cosponsor_count']}")
            print(f"  URL:     {bill['url']}")
        # Pause between queries to respect DEMO_KEY rate limits
        time.sleep(10)
