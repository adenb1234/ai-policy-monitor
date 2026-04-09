"""
AI Policy Monitor — Agent implementations

Three agents:
1. Researcher: Uses Perplexity Sonar API to search for policy/regulatory developments
2. Analyst: Uses Claude to synthesize findings and identify gaps (may request follow-up)
3. Brief Writer: Uses Claude to produce a polished executive policy brief
"""

import os
import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


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
# AGENT 1: RESEARCHER
# ============================================================

RESEARCHER_SYSTEM = """You are a policy research agent working for Perplexity AI, a US AI search company.

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
    client = _get_anthropic()

    if follow_up_queries:
        search_tasks = [
            f"{topic} {q} 2024 2025 regulation law" for q in follow_up_queries[:3]
        ]
    else:
        search_tasks = [
            f"{topic} regulation legislation Congress EU law 2024 2025",
            f"{topic} court cases litigation lawsuits AI companies copyright",
            f"{topic} compliance requirements industry response AI companies Perplexity",
        ]

    # Run Perplexity queries in parallel
    raw_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_call_perplexity, q): q for q in search_tasks}
        for future in as_completed(futures):
            query = futures[future]
            try:
                result = future.result()
                raw_results.append(
                    {
                        "query": query,
                        "content": result["content"],
                        "citations": result["citations"],
                    }
                )
            except Exception as e:
                # Log the failure but don't abort
                raw_results.append(
                    {"query": query, "content": f"[Search failed: {e}]", "citations": []}
                )

    # Build context for Claude to structure
    context_parts = [f"Topic: {topic}\n"]
    if follow_up_queries:
        context_parts.append(
            "Follow-up gaps from Analyst:\n"
            + "\n".join(f"- {q}" for q in follow_up_queries)
            + "\n"
        )
    context_parts.append("Raw Perplexity search results:\n")
    for r in raw_results:
        context_parts.append(f"--- Query: {r['query']} ---")
        context_parts.append(f"Content:\n{r['content'][:1800]}")
        context_parts.append(f"Citation URLs: {json.dumps(r['citations'])}\n")

    context = "\n".join(context_parts)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=RESEARCHER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Structure these search results into the required JSON format:\n\n{context}",
            }
        ],
    )

    return _parse_json(response.content[0].text)


# ============================================================
# AGENT 2: ANALYST
# ============================================================

ANALYST_SYSTEM = """You are a senior policy analyst at Perplexity AI, a US AI search company.
You receive structured research findings and produce strategic policy analysis.

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
    client = _get_anthropic()

    extra = ""
    if loop_count >= 1:
        extra = "\n\nIMPORTANT: Set follow_up_queries to [] — no further research rounds are permitted. Complete your analysis with available information."

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=ANALYST_SYSTEM + extra,
        messages=[
            {
                "role": "user",
                "content": f"Analyze these research findings:\n\n{json.dumps(research, indent=2)}",
            }
        ],
    )

    return _parse_json(response.content[0].text)


# ============================================================
# AGENT 3: BRIEF WRITER
# ============================================================

BRIEF_WRITER_SYSTEM = """You are a policy communications specialist at Perplexity AI.
You write polished, one-page executive policy briefs for senior leadership.

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

    client = _get_anthropic()
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

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=BRIEF_WRITER_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )

    return response.content[0].text
