"""
AI Policy Monitor — Pipeline orchestration (CLI / programmatic use)

Orchestrates the three agents with an iterative research loop:
  Researcher → Analyst → (follow-up Researcher if gaps found) → Brief Writer
"""

from typing import Callable, Optional

from agents import run_researcher, run_analyst, run_brief_writer

MAX_LOOPS = 2


def run_pipeline(
    topic: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run the full policy research pipeline.

    Args:
        topic: Policy/regulatory topic to research
        progress_callback: Optional function called with status messages

    Returns:
        {
            "brief": str,           # Markdown policy brief
            "research": dict,       # Final merged research findings
            "analysis": dict,       # Final analysis
            "iterations": int,      # Research rounds completed
            "all_research": list,   # All research rounds (for inspection)
            "all_analyses": list,   # All analysis rounds (for inspection)
        }
    """

    def notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    all_research = []
    all_analyses = []

    # ── Step 1: Initial research ──────────────────────────────────────────
    notify("Searching for regulatory and policy developments...")
    research = run_researcher(topic)
    all_research.append(research)
    notify(f"Found {len(research.get('findings', []))} sources.")

    # ── Step 2: Analysis loop ─────────────────────────────────────────────
    analysis = None
    iterations = 1

    for loop in range(MAX_LOOPS + 1):
        notify(f"Analyzing findings (round {loop + 1})...")
        analysis = run_analyst(research, loop_count=loop)
        all_analyses.append(analysis)

        follow_ups = analysis.get("follow_up_queries", [])

        if not follow_ups or loop == MAX_LOOPS:
            iterations = loop + 1
            if follow_ups and loop == MAX_LOOPS:
                notify(f"Max loops ({MAX_LOOPS}) reached — proceeding with available research.")
            break

        notify(f"Analyst found {len(follow_ups)} gap(s) — running follow-up research...")
        follow_up_research = run_researcher(topic, follow_up_queries=follow_ups)
        all_research.append(follow_up_research)

        # Merge new findings (deduplicate by URL)
        existing_urls = {f.get("url", "") for f in research.get("findings", [])}
        new_findings = [
            f
            for f in follow_up_research.get("findings", [])
            if f.get("url", "") not in existing_urls
        ]
        research["findings"] = research.get("findings", []) + new_findings
        notify(f"Added {len(new_findings)} new sources.")

    # ── Step 3: Write brief ───────────────────────────────────────────────
    notify("Writing policy brief...")
    brief = run_brief_writer(analysis, research)

    return {
        "brief": brief,
        "research": research,
        "analysis": analysis,
        "iterations": iterations,
        "all_research": all_research,
        "all_analyses": all_analyses,
    }


if __name__ == "__main__":
    import sys
    import json

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Copyright litigation exposure for AI search engines"
    print(f"\nRunning pipeline for: {topic}\n{'='*60}")

    result = run_pipeline(topic, progress_callback=lambda msg: print(f"  → {msg}"))

    print("\n" + "=" * 60)
    print(result["brief"])
    print("=" * 60)
    print(f"\nCompleted in {result['iterations']} research round(s).")
    print(f"Total sources: {len(result['research'].get('findings', []))}")
