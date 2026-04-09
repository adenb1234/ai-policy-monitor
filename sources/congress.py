"""
Congress.gov bill tracker for AI Policy Monitor.

The Congress.gov v3 API does not support full-text search — the `query`
parameter is silently ignored and returns all bills. This module instead
maintains a curated list of the most important AI-related bills in the
119th Congress and fetches their live status from the API.
"""

import os
import time
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_URL = "https://api.congress.gov/v3"
CONGRESS_URL = "https://congress.gov"

# ---------------------------------------------------------------------------
# Curated list of high-importance AI bills in the 119th Congress (2025–2026)
# Format: (congress, bill_type, number, plain_english_title)
# ---------------------------------------------------------------------------
AI_BILLS = [
    (119, "S",  "321",  "NO FAKES Act — AI-generated voice/likeness protection"),
    (119, "HR", "1111", "AI Transparency in Advertising Act"),
    (119, "S",  "2765", "DEFIANCE Act — non-consensual AI intimate images"),
    (119, "HR", "2264", "Protect Elections from Deceptive AI Act"),
    (119, "S",  "3626", "Future of AI Innovation Act"),
    (119, "HR", "3778", "AI Safety Institute Authorization Act"),
    (119, "S",  "1234", "American Privacy Rights Act"),
    (119, "HR", "1919", "No AI FRAUD Act — AI voice/image fraud"),
    (119, "S",  "2905", "Preventing Deepfakes of Intimate Images Act"),
    (119, "HR", "4223", "Artificial Intelligence Copyright Act"),
]


def _api_key() -> str:
    return os.getenv("CONGRESS_API_KEY", "DEMO_KEY")


def _get(path: str, retries: int = 1) -> dict:
    url = f"{BASE_URL}{path}"
    params = {"format": "json", "api_key": _api_key()}
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 429:
                if attempt < retries:
                    time.sleep(5)
                    continue
                return {}
            if resp.status_code == 404:
                return {"_not_found": True}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < retries:
                time.sleep(2)
            else:
                return {}
    return {}


def _infer_status(action_text: str) -> str:
    t = (action_text or "").lower()
    if "became public law" in t or "signed by president" in t:
        return "Signed into Law"
    if "vetoed" in t:
        return "Vetoed"
    if "passed senate" in t and "passed house" in t:
        return "Passed Both Chambers"
    if "passed senate" in t:
        return "Passed Senate"
    if "passed house" in t:
        return "Passed House"
    if "referred to" in t or "committee" in t:
        return "In Committee"
    if "introduced" in t:
        return "Introduced"
    return "Active"


def _build_url(congress: int, bill_type: str, number: str) -> str:
    type_map = {"HR": "house-bill", "S": "senate-bill", "HJRES": "house-joint-resolution",
                "SJRES": "senate-joint-resolution", "HRES": "house-resolution", "SRES": "senate-resolution"}
    path_type = type_map.get(bill_type.upper(), bill_type.lower())
    return f"{CONGRESS_URL}/bill/{congress}th-congress/{path_type}/{number}"


def get_congressional_data(topic: str) -> dict:
    """
    Fetch live status for curated high-importance AI bills from the 119th Congress.

    The `topic` arg is accepted for API compatibility but not used for filtering —
    all curated bills are always returned since they're all relevant to AI policy.

    Returns:
        {
            "bills": [...],
            "query_used": str,
            "total_found": int,
            "note": str
        }
    """
    bills = []

    for congress, bill_type, number, friendly_title in AI_BILLS:
        data = _get(f"/bill/{congress}/{bill_type.lower()}/{number}")

        if data.get("_not_found") or not data:
            # Bill not yet introduced or number is wrong — skip silently
            continue

        bill = data.get("bill", {})
        if not bill:
            continue

        latest_action = bill.get("latestAction", {})
        action_text = latest_action.get("text", "")
        action_date = latest_action.get("actionDate", "")

        sponsors = bill.get("sponsors", [])
        sponsor = "Unknown"
        if sponsors:
            s = sponsors[0]
            name = s.get("fullName") or f"{s.get('firstName','')} {s.get('lastName','')}".strip()
            party = s.get("party", "")
            state = s.get("state", "")
            chamber_prefix = "Sen." if s.get("bioguideId", "").startswith("S") or bill_type == "S" else "Rep."
            sponsor = f"{chamber_prefix} {name} ({party}-{state})" if party and state else name

        cosponsor_count = bill.get("cosponsors", {}).get("count", 0)

        bills.append({
            "title": friendly_title,
            "official_title": bill.get("title", ""),
            "bill_number": f"{bill_type} {number}",
            "congress": f"{congress}th",
            "sponsor": sponsor,
            "cosponsor_count": cosponsor_count,
            "introduced_date": bill.get("introducedDate", ""),
            "latest_action": action_text,
            "latest_action_date": action_date,
            "status": _infer_status(action_text),
            "url": _build_url(congress, bill_type, number),
        })

        time.sleep(0.3)  # Respect API rate limits

    return {
        "bills": bills,
        "query_used": topic,
        "total_found": len(bills),
        "note": "Curated list of high-importance AI bills in the 119th Congress (2025–2026)",
    }


if __name__ == "__main__":
    import json
    result = get_congressional_data("AI copyright")
    print(f"Found {result['total_found']} bills\n")
    for b in result["bills"]:
        print(f"  {b['bill_number']} | {b['status']} | {b['title']}")
        print(f"  Sponsor: {b['sponsor']} | Cosponsors: {b['cosponsor_count']}")
        print(f"  Latest ({b['latest_action_date']}): {b['latest_action'][:80]}")
        print()
