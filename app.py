"""
AI Policy Monitor — Streamlit web app
"""

import streamlit as st
import json
from agents import run_researcher, run_analyst, run_brief_writer, CreditsError

MAX_LOOPS = 2

st.set_page_config(
    page_title="AI Policy Monitor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Perplexity Policy Intelligence")
    st.markdown(
        "Multi-agent pipeline for researching regulatory and legal risks to Perplexity AI."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
1. **Researcher** queries Perplexity Sonar across regulatory, litigation, and compliance angles — with Perplexity's business model as context
2. **Analyst** (GPT-4o) synthesizes findings into risk threads specific to Perplexity — may request follow-up research (max 2 rounds)
3. **Brief Writer** (GPT-4o) produces a polished executive brief
        """
    )
    st.markdown("---")
    st.markdown("### Example risks")
    demo_topics = [
        "Copyright litigation from news publishers",
        "State-level AI transparency laws",
        "EU AI Act compliance for search products",
    ]
    for t in demo_topics:
        if st.button(t, key=f"demo_{t}", use_container_width=True):
            st.session_state.topic_value = t
            st.rerun()

    st.markdown("---")
    st.caption("Researcher: Perplexity Sonar · Analyst & Writer: Claude Sonnet")


# ── Session state init ────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "topic_value" not in st.session_state:
    st.session_state.topic_value = ""


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Perplexity Policy Intelligence")
st.markdown(
    "Describe a regulatory or legal risk to Perplexity. "
    "The pipeline researches it, assesses the exposure, and produces a finished executive brief."
)
st.markdown("")

# ── Input ─────────────────────────────────────────────────────────────────────
topic = st.text_input(
    "Risk to research",
    value=st.session_state.topic_value,
    placeholder="e.g. Copyright litigation from news publishers",
    label_visibility="collapsed",
)

run_col, _ = st.columns([1, 6])
with run_col:
    run_button = st.button("Run Analysis", type="primary", disabled=not topic.strip())

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_button and topic.strip():
    st.session_state.results = None
    st.session_state.last_topic = topic.strip()
    st.session_state.topic_value = topic.strip()

    all_research_rounds = []
    all_analysis_rounds = []
    error_msg = None

    with st.status("Running AI policy analysis...", expanded=True) as status:
        try:
            # ── Researcher ────────────────────────────────────────────────
            status.write("🔍 **Researcher:** Querying Perplexity Sonar (3 parallel searches)...")
            research = run_researcher(topic.strip())
            all_research_rounds.append(research)
            num_findings = len(research.get("findings", []))
            num_queries = len(research.get("search_queries", []))
            status.write(f"✅ Found **{num_findings} sources** across {num_queries} queries")

            # ── Analyst loop ──────────────────────────────────────────────
            analysis = None
            iterations = 1

            for loop in range(MAX_LOOPS + 1):
                status.write(f"🧠 **Analyst:** Synthesizing and assessing risk (round {loop + 1})...")
                analysis = run_analyst(research, loop_count=loop)
                all_analysis_rounds.append(analysis)

                follow_ups = analysis.get("follow_up_queries", [])

                if not follow_ups or loop == MAX_LOOPS:
                    iterations = loop + 1
                    if follow_ups and loop == MAX_LOOPS:
                        status.write("⚠️ Max research loops reached — proceeding with available findings")
                    break

                status.write(
                    f"🔄 **Analyst** identified **{len(follow_ups)} research gap(s)** — requesting follow-up:"
                )
                for q in follow_ups:
                    status.write(f"  → _{q}_")

                status.write("🔍 **Researcher:** Running follow-up queries...")
                follow_up_research = run_researcher(topic.strip(), follow_up_queries=follow_ups)
                all_research_rounds.append(follow_up_research)

                existing_urls = {f.get("url", "") for f in research.get("findings", [])}
                new_findings = [
                    f
                    for f in follow_up_research.get("findings", [])
                    if f.get("url", "") not in existing_urls
                ]
                research["findings"] = research.get("findings", []) + new_findings
                status.write(f"✅ Added **{len(new_findings)} new sources**")

            # ── Brief Writer ──────────────────────────────────────────────
            status.write("✍️ **Brief Writer:** Producing executive policy brief...")
            brief = run_brief_writer(analysis, research)

            total_sources = len(research.get("findings", []))
            status.update(
                label=f"Analysis complete · {iterations} research round(s) · {total_sources} sources",
                state="complete",
                expanded=False,
            )

            st.session_state.results = {
                "brief": brief,
                "research": research,
                "analysis": analysis,
                "iterations": iterations,
                "all_research_rounds": all_research_rounds,
                "all_analysis_rounds": all_analysis_rounds,
            }

        except CreditsError as e:
            status.update(label="Out of credits", state="error", expanded=True)
            error_msg = str(e)
            st.session_state.credits_error = True
        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            error_msg = str(e)
            st.session_state.credits_error = False

    if error_msg:
        if st.session_state.get("credits_error"):
            st.error(f"💳 {error_msg} — please top up at platform.openai.com or perplexity.ai and try again.")
        else:
            st.error("Something went wrong. Please check that your API keys are set and try again.")
            with st.expander("Error details"):
                st.code(error_msg)


# ── Results display ───────────────────────────────────────────────────────────
results = st.session_state.results

if results:
    st.markdown("---")

    # Final brief
    st.markdown(results["brief"])

    # Download button
    st.download_button(
        label="Download Brief (Markdown)",
        data=results["brief"],
        file_name=f"policy_brief_{results['research'].get('topic', 'output')[:40].replace(' ', '_').lower()}.md",
        mime="text/markdown",
    )

    st.markdown("---")
    st.markdown("### Pipeline Details")

    # Research findings
    findings = results["research"].get("findings", [])
    with st.expander(f"Research Findings — {len(findings)} sources"):
        risk_badge = {"5": "🔴", "4": "🟠", "3": "🟡", "2": "🟢", "1": "⚪"}
        for i, f in enumerate(findings, 1):
            rel = str(f.get("relevance", "?"))
            badge = risk_badge.get(rel, "⚪")
            st.markdown(f"**{i}. {f.get('source', 'Unknown')}** {badge} relevance {rel}/5")
            url = f.get("url", "")
            if url:
                display = url if len(url) <= 70 else url[:67] + "..."
                st.markdown(f"[{display}]({url})")
            if f.get("date"):
                st.caption(f.get("date"))
            for claim in f.get("key_claims", [])[:2]:
                st.markdown(f"- {claim[:250]}")
            st.markdown("")

    # Analysis threads
    threads = results["analysis"].get("threads", [])
    with st.expander(f"Analysis — {len(threads)} risk threads"):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        for thread in threads:
            emoji = risk_emoji.get(thread.get("risk", ""), "⚪")
            st.markdown(
                f"{emoji} **{thread.get('title', 'Thread')}** — "
                f"`{thread.get('risk', '?').upper()}` risk · "
                f"{thread.get('timeline', '?')} horizon"
            )
            st.markdown(thread.get("summary", ""))
            st.caption(f"Perplexity relevance: {thread.get('perplexity_relevance', '')}")
            st.markdown("")

        assessment = results["analysis"].get("overall_assessment", "")
        if assessment:
            st.markdown("---")
            st.markdown(f"**Overall assessment:** {assessment}")

    # Iteration details (only if > 1 round)
    if results.get("iterations", 1) > 1:
        with st.expander(f"Research Iterations — {results['iterations']} rounds"):
            for i, (r, a) in enumerate(
                zip(results["all_research_rounds"], results["all_analysis_rounds"]), 1
            ):
                st.markdown(f"**Round {i}** — {len(r.get('findings', []))} findings")
                if a.get("follow_up_queries"):
                    st.markdown("Follow-up queries requested:")
                    for q in a["follow_up_queries"]:
                        st.markdown(f"- {q}")
                st.markdown("")

    # Raw JSON
    with st.expander("Raw JSON output"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Research:**")
            st.json(results["research"])
        with col2:
            st.markdown("**Analysis:**")
            st.json(results["analysis"])
