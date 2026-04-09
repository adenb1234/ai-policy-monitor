"""
AI Policy Monitor — Agent implementations

Three agents:
1. Researcher: Uses Perplexity Sonar API to search for policy/regulatory developments
2. Analyst: Uses GPT-4o to synthesize findings and identify gaps (may request follow-up)
3. Brief Writer: Uses GPT-4o to produce a polished executive policy brief
"""

import os
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_openai_client: Optional[OpenAI] = None


class CreditsError(Exception):
    """Raised when an API call fails due to insufficient credits or quota."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Aden ran out of {provider} credits")


def _get_openai():
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _openai_call(*args, **kwargs):
    """Wrapper around openai client calls that converts credit errors to CreditsError."""
    from openai import RateLimitError, APIStatusError
    try:
        return _get_openai().chat.completions.create(*args, **kwargs)
    except RateLimitError:
        raise CreditsError("OpenAI")
    except APIStatusError as e:
        if e.status_code in (402, 429) or "quota" in str(e).lower() or "billing" in str(e).lower():
            raise CreditsError("OpenAI")
        raise


def _get_perplexity_key() -> str:
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set.")
    return key


# ============================================================
# HELPERS
# ============================================================

def _call_perplexity(query: str) -> dict:
    """Single Perplexity Sonar API call. Returns content + citations."""
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {_get_perplexity_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a policy research assistant. Provide factual, "
                        "well-sourced information about regulatory, legal, and policy "
                        "developments. Be specific about dates, jurisdictions, and key "
                        "decision-makers. Cite your sources precisely."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
        },
        timeout=45,
    )
    if response.status_code in (402, 429):
        raise CreditsError("Perplexity")
    response.raise_for_status()
    data = response.json()
    return {
        "content": data["choices"][0]["message"]["content"],
        "citations": data.get("citations", []),
    }


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from a model response."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Raw object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model response. Preview: {text[:400]}")


# ============================================================
# PERPLEXITY COMPANY CONTEXT (injected into all agents)
# ============================================================

PERPLEXITY_CONTEXT = """
COMPANY CONTEXT — Perplexity AI:
- Product: AI-powered answer engine / search product. Aggregates and synthesizes web content into direct answers with citations.
- Business model: Freemium subscriptions (~$20/mo Pro), enterprise API, and an advertising product launched in 2024.
- Web infrastructure: Indexes and scrapes the public web via its own crawler (PerplexityBot). Has a Publisher Program offering revenue sharing to incentivize opt-in.
- Scale: ~100M monthly active users as of early 2025. Incorporated in Delaware, HQ San Francisco.
- Active litigation (as of early 2026): Copyright suits filed by Dow Jones, NYT, Chicago Tribune, Encyclopaedia Britannica, and Amazon (alleging unauthorized scraping/reproduction of content). A 9th Circuit appeal from an Amazon case is pending.
- Key legal exposure vectors:
    1. Fair use / copyright: whether reproducing snippets or full answers from indexed content is fair use
    2. Scraping / robots.txt: whether ignoring robots.txt creates legal liability
    3. Publisher relations: regulatory pressure to mandate revenue sharing or licensing
    4. AI transparency: EU AI Act and state laws may require disclosure of AI-generated content
    5. Data privacy: GDPR, CCPA implications for user query data and training data
- Competitive context: competes directly with Google Search, Bing/Copilot, ChatGPT Search. Google has resources for prolonged litigation that Perplexity does not.
- Strategic vulnerability: unlike OpenAI or Google, Perplexity does not train its own foundation models — its differentiation is entirely in search/retrieval/UX, making IP and scraping law existentially important.
"""

# ============================================================
# AGENT 1: RESEARCHER
# ============================================================

RESEARCHER_SYSTEM = """You are a policy research agent working for Perplexity AI, a US AI search company.
""" + PERPLEXITY_CONTEXT + """
Given raw search results from the Perplexity Sonar search engine, structure them into a clean research report.

Output ONLY valid JSON — no preamble, no explanation, no markdown wrapper:
{
  "topic": "the research topic",
  "search_queries": ["query 1", "query 2", "..."],
  "findings": [
    {
      "source": "Organization or publication name",
      "url": "https://...",
      "date": "YYYY-MM or YYYY or empty string if unknown",
      "key_claims": ["specific factual claim under 100 words", "another claim"],
      "relevance": 4
    }
  ]
}

Rules:
- One finding per distinct source (not per query — merge if multiple queries hit the same source)
- key_claims must be specific and factual, not vague summaries
- relevance: 1–5 scale; 5 = directly addresses this topic for an AI company
- Include 6–12 findings total, prioritizing the most recent and highest-relevance
- Never invent sources or URLs not present in the raw results"""


def run_researcher(topic: str, follow_up_queries: Optional[list] = None) -> dict:
    """
    Researcher agent: searches Perplexity Sonar and returns structured findings.

    Args:
        topic: The policy/regulatory topic to research
        follow_up_queries: Optional gaps identified by the Analyst

    Returns:
        Structured research findings dict
    """
    if follow_up_queries:
        search_tasks = [
            f"{topic} {q} 2024 2025 regulation law" for q in follow_up_queries[:3]
        ]
    else:
        search_tasks = [
            f"Perplexity AI {topic} regulation legislation risk 2024 2025",
            f"{topic} court cases litigation AI search engines Perplexity",
            f"{topic} compliance requirements AI search products industry response",
        ]

    # Run Perplexity searches + alternative data sources in parallel
    from sources.courtlistener import get_court_data
    from sources.polymarket import get_polymarket_data
    from sources.congress import get_congressional_data

    raw_results = []
    court_data = {}
    market_data = {}
    congress_data = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        # Perplexity searches
        pplx_futures = {executor.submit(_call_perplexity, q): q for q in search_tasks}
        # Alternative data sources (always run, report even if empty)
        court_future = executor.submit(get_court_data, f"Perplexity AI {topic}")
        market_future = executor.submit(get_polymarket_data, [
            topic, "AI regulation", "AI copyright", "artificial intelligence law", "EU AI Act"
        ])
        congress_future = executor.submit(get_congressional_data, f"artificial intelligence {topic}")

        for future in as_completed(pplx_futures):
            query = pplx_futures[future]
            try:
                result = future.result()
                raw_results.append({"query": query, "content": result["content"], "citations": result["citations"]})
            except CreditsError:
                raise
            except Exception as e:
                raw_results.append({"query": query, "content": f"[Search failed: {e}]", "citations": []})

        try:
            court_data = court_future.result()
        except Exception as e:
            court_data = {"cases": [], "total_found": 0, "error": str(e)}

        try:
            market_data = market_future.result()
        except Exception as e:
            market_data = {"markets": [], "total_found": 0, "error": str(e)}

        try:
            congress_data = congress_future.result()
        except Exception as e:
            congress_data = {"bills": [], "total_found": 0, "error": str(e)}

    # Build context — always include all data sources, even if empty
    context_parts = [f"Risk topic (Perplexity-specific): {topic}\n"]
    if follow_up_queries:
        context_parts.append(
            "Follow-up gaps from Analyst:\n"
            + "\n".join(f"- {q}" for q in follow_up_queries)
            + "\n"
        )

    context_parts.append("=== PERPLEXITY SONAR SEARCH RESULTS ===")
    for r in raw_results:
        context_parts.append(f"--- Query: {r['query']} ---")
        context_parts.append(f"Content:\n{r['content'][:1800]}")
        context_parts.append(f"Citation URLs: {json.dumps(r['citations'])}\n")

    context_parts.append("=== COURTLISTENER: FEDERAL COURT CASES ===")
    cases = court_data.get("cases", [])
    if cases:
        context_parts.append(f"Found {court_data.get('total_found', 0)} total cases. Top results:")
        for c in cases[:6]:
            context_parts.append(f"- {c['case_name']} | {c['court']} | Filed: {c['date_filed']} | {c['docket_url']}")
            if c.get("recent_filings"):
                for f in c["recent_filings"][:2]:
                    context_parts.append(f"  Recent filing ({f['date']}): {f['description'][:120]}")
    else:
        context_parts.append(f"No federal court cases found for this topic. (Checked CourtListener RECAP database.)")

    context_parts.append("\n=== POLYMARKET: PREDICTION MARKETS ===")
    markets = market_data.get("markets", [])
    if markets:
        context_parts.append(f"Found {len(markets)} relevant markets (>$5k volume):")
        for m in markets[:5]:
            prob = f"{m['probability']:.0%}" if m.get("probability") is not None else "N/A"
            context_parts.append(f"- \"{m['question']}\" | Probability: {prob} | Volume: ${m['volume']:,.0f} | Ends: {m['end_date']}")
            context_parts.append(f"  {m['url']}")
    else:
        context_parts.append("No relevant prediction markets found on Polymarket (searched: AI regulation, AI copyright, EU AI Act).")

    context_parts.append("\n=== CONGRESS.GOV: LEGISLATION ===")
    bills = congress_data.get("bills", [])
    if bills:
        context_parts.append(f"Found {congress_data.get('total_found', 0)} total bills. Top results:")
        for b in bills[:5]:
            context_parts.append(f"- {b['bill_number']} ({b['congress']}): {b['title'][:80]}")
            context_parts.append(f"  Sponsor: {b['sponsor']} | Status: {b['status']} | Cosponsors: {b['cosponsor_count']}")
            context_parts.append(f"  Latest action ({b['latest_action_date']}): {b['latest_action'][:100]}")
    else:
        context_parts.append("No relevant bills found on Congress.gov for this topic.")

    context = "\n".join(context_parts)

    client = _get_openai()
    response = _openai_call(
        model="gpt-4o-mini",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": RESEARCHER_SYSTEM},
            {
                "role": "user",
                "content": f"Structure these search results into the required JSON format:\n\n{context}",
            },
        ],
    )

    return _parse_json(response.choices[0].message.content)


# ============================================================
# AGENT 2: ANALYST
# ============================================================

ANALYST_SYSTEM = """You are a senior policy analyst at Perplexity AI, a US AI search company.
You receive structured research findings and produce strategic policy analysis.
""" + PERPLEXITY_CONTEXT + """

Output ONLY valid JSON — no preamble, no explanation, no markdown wrapper:
{
  "threads": [
    {
      "title": "Short descriptive title (5–8 words)",
      "summary": "2–3 sentence factual summary of what is happening",
      "perplexity_relevance": "1–2 sentences on why this matters specifically to Perplexity (not generic AI companies)",
      "timeline": "immediate | 6-month | 12-month",
      "risk": "low | medium | high | critical"
    }
  ],
  "contradictions": ["Description of conflicting information between sources, if any"],
  "gaps": ["Specific missing information that would materially change the risk assessment"],
  "follow_up_queries": ["Specific research question if a gap is significant enough to require more searching"],
  "overall_assessment": "2–3 sentence net assessment of Perplexity's regulatory/legal exposure on this topic"
}

Definitions:
- timeline: immediate = within 3 months; 6-month = 3–9 months; 12-month = 9–18 months
- risk: low = monitor only; medium = active monitoring; high = response plan needed; critical = urgent legal/policy action required
- threads: 3–5 threads covering the most important issues
- follow_up_queries: ONLY non-empty if a gap would significantly change your risk conclusions (max 2 queries)
- Be specific about Perplexity's actual exposure — not generic AI company risks"""


def run_analyst(research: dict, loop_count: int = 0) -> dict:
    """
    Analyst agent: synthesizes research and identifies gaps.

    Args:
        research: Structured findings from Researcher
        loop_count: 0-indexed loop iteration (used to block follow-up on last round)

    Returns:
        Structured analysis dict (may contain follow_up_queries)
    """
    client = _get_openai()

    extra = ""
    if loop_count >= 1:
        extra = "\n\nIMPORTANT: Set follow_up_queries to [] — no further research rounds are permitted. Complete your analysis with available information."

    response = _openai_call(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM + extra},
            {
                "role": "user",
                "content": f"Analyze these research findings:\n\n{json.dumps(research, indent=2)}",
            },
        ],
    )

    return _parse_json(response.choices[0].message.content)


# ============================================================
# AGENT 3: BRIEF WRITER
# ============================================================

BRIEF_WRITER_SYSTEM = """You are a policy communications specialist at Perplexity AI.
You write polished, one-page executive policy briefs for senior leadership.
""" + PERPLEXITY_CONTEXT + """

Writing rules:
- No hedging or filler ("it is important to note that...", "it remains to be seen...")
- Every sentence must contain information
- Write for a smart reader who is not a domain expert
- Prefer active voice
- Keep the entire brief under 800 words
- Do not use placeholder text like [DATE] or [URL] — use actual content from the analysis

Output the brief in this exact Markdown format (no preamble, start directly with the title):

# [SPECIFIC TOPIC TITLE] — Policy Brief
**Date:** [today's date]
**Prepared by:** AI Policy Monitor

## Executive Summary
2–3 sentences. Lead with the bottom line: net risk level and what Perplexity must do.

## Key Developments
[One paragraph per thread, with a **bold title**. Cover what is happening, why it matters to Perplexity, and the timeline. 3–5 threads.]

## Risk Assessment

| Development | Risk Level | Timeline | Action Required |
|-------------|-----------|----------|-----------------|
[One row per thread from the analysis]

**Overall posture:** [1–2 sentences on combined risk picture from the overall_assessment.]

## Recommended Actions
- [Specific, actionable recommendation 1 — name specific teams, bodies, or actions]
- [Specific, actionable recommendation 2]
- [Specific, actionable recommendation 3]
- [Specific, actionable recommendation 4 if warranted]
- [Specific, actionable recommendation 5 if warranted]

## Sources
[Numbered list of the most relevant sources with URLs from the research findings. Include at least 5.]"""


def run_brief_writer(analysis: dict, research: dict) -> str:
    """
    Brief Writer agent: produces a polished markdown policy brief.

    Args:
        analysis: Structured analysis from Analyst
        research: Research findings (for citation URLs)

    Returns:
        Markdown-formatted policy brief string
    """
    from datetime import date

    client = _get_openai()
    today = date.today().strftime("%B %d, %Y")

    # Compile unique sources
    sources = []
    seen_urls = set()
    for f in research.get("findings", []):
        url = f.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(f"{f.get('source', 'Source')}: {url}")

    context = f"""Today's date: {today}

Topic: {research.get("topic", "AI Policy")}

ANALYSIS THREADS:
{json.dumps(analysis.get("threads", []), indent=2)}

OVERALL ASSESSMENT:
{analysis.get("overall_assessment", "")}

AVAILABLE SOURCES (use the most relevant ones):
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(sources[:15]))}

Write the policy brief now. Use the threads to fill the Key Developments section. Use overall_assessment to anchor the Executive Summary. Make the Recommended Actions concrete and specific to Perplexity."""

    response = _openai_call(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": BRIEF_WRITER_SYSTEM},
            {"role": "user", "content": context},
        ],
    )

    return response.choices[0].message.content
