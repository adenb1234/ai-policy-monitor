# AI Policy Monitor

A multi-agent pipeline that researches a policy or regulatory topic relevant to Perplexity AI and produces a finished executive brief — automatically.

**Live app:** [link TBD after deployment]

---

## What it does

Enter a topic like *"Copyright litigation exposure for AI search engines"* and the pipeline:

1. **Researches** it — three parallel Perplexity Sonar queries covering regulatory, litigation, and compliance angles
2. **Analyzes** the findings — Claude identifies key risk threads, flags gaps, and may request follow-up research (up to 2 rounds)
3. **Writes** a polished one-page executive brief — with risk table, recommended actions, and citations

The whole thing takes about 60–90 seconds.

---

## Architecture

```
User input (topic string)
        │
        ▼
┌──────────────────────────────────────────────┐
│              RESEARCHER AGENT                │
│  Generates 3 targeted queries                │
│  → Perplexity Sonar API (3× parallel calls) │
│  → Claude Haiku structures the findings      │
│  Output: JSON { topic, queries, findings[] } │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│               ANALYST AGENT                  │
│  Claude Sonnet synthesizes research          │
│  → Identifies 3–5 risk threads               │
│  → Flags gaps → sends follow_up_queries[]    │
│     (loops back to Researcher, max 2 rounds) │
│  Output: JSON { threads[], gaps[], ... }     │
└──────────────────────┬───────────────────────┘
                       │ (once satisfied)
                       ▼
┌──────────────────────────────────────────────┐
│            BRIEF WRITER AGENT                │
│  Claude Sonnet writes the policy brief       │
│  Output: Markdown (Executive Summary,        │
│          Key Developments, Risk Table,       │
│          Recommended Actions, Sources)       │
└──────────────────────────────────────────────┘
```

**Why this architecture?**

- Perplexity Sonar is better than Claude at real-time web search — so the Researcher uses Sonar exclusively
- Claude Sonnet is better at synthesis, structured reasoning, and polished prose — so the Analyst and Brief Writer use Claude
- The iterative loop between Researcher and Analyst mimics how real policy analysts work: draft → identify gaps → dig deeper

---

## Stack

| Component | Technology |
|-----------|-----------|
| Search | Perplexity Sonar API (`sonar` model) |
| Analysis & Writing | Anthropic Claude (`claude-sonnet-4-6`) |
| Query structuring | Anthropic Claude (`claude-haiku-4-5`) |
| Web app | Streamlit |
| Deployment | Render |

---

## Demo topics

- Copyright litigation exposure for AI search engines
- State-level AI regulation affecting AI product companies
- EU AI Act compliance requirements for AI search products

---

## Local setup

```bash
git clone https://github.com/adenb1234/ai-policy-monitor
cd ai-policy-monitor

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env

streamlit run app.py
```

**Required environment variables:**

```
ANTHROPIC_API_KEY=...
PERPLEXITY_API_KEY=...
```

---

## Run as CLI

```bash
python pipeline.py "Copyright litigation exposure for AI search engines"
```

---

## Project context

Built to demonstrate the work described in Perplexity AI's *Member of Technical Staff — AI Policy and Strategic Initiatives* role, specifically: *"Harness AI agents to research high-stakes policy and regulatory issues pertaining to Perplexity."*
