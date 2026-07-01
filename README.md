---
title: Research Agent
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Research Agent

**🚀 Live demo: [huggingface.co/spaces/eraxes/research-agent](https://huggingface.co/spaces/eraxes/research-agent)**

An **autonomous research agent** built **from scratch** (no LangChain) on **Groq function-calling**. Ask a question and the agent decides what to do — searches the web, reads pages, does math — looping until it can answer **with cited sources**. You watch its reasoning **live** in the browser.

This is a **ReAct** agent (Reasoning + Acting): at each step the model either calls a tool or gives the final answer; the app runs the tool, feeds the result back, and repeats.

---

## How the ReAct loop works

```
QUESTION
   │
   ▼
 ┌─────────────────────────────────────────────┐
 │  LLM: call a tool?  ──yes──▶  run the tool ─┐ │
 │        │ no                      ▲          │ │
 │        ▼                         └─ result ─┘ │
 │   FINAL ANSWER (with sources)                 │
 └─────────────────────────────────────────────┘
```

The model emits structured **tool calls** (function-calling); the app executes them and returns the output, so the model reasons over real data instead of guessing. A `max_steps` guard prevents infinite loops, and every step is streamed to the UI via **Server-Sent Events**.

---

## Tools

| Tool | What it does |
|------|--------------|
| `web_search` | DuckDuckGo search (no API key) — returns titles, URLs, snippets |
| `read_url`   | Fetches a page and extracts clean text (deep read) |
| `calculator` | Safe arithmetic — AST-based, **no `eval()`** |

Each tool is a plain Python function plus a JSON schema the model sees.

---

## Tech stack

| Layer   | Choice                                             |
|---------|----------------------------------------------------|
| LLM     | Groq `llama-3.3-70b-versatile` (free, tool use)    |
| Agent   | Hand-written **ReAct loop** (`agent/core.py`)      |
| API/UI  | FastAPI + Server-Sent Events + vanilla HTML/CSS/JS |
| Search  | DuckDuckGo (`ddgs`) + BeautifulSoup for page text  |

---

## Project structure

```
research-agent/
├── agent/
│   ├── config.py       # settings (loaded from .env)
│   ├── tools.py        # web_search / read_url / calculator + JSON schemas
│   └── core.py         # the ReAct loop (run_agent)
├── static/
│   └── index.html      # web UI — live trace via SSE
├── app.py              # FastAPI: GET /ask streams the reasoning trace
├── cli.py              # command-line interface (same agent)
├── Dockerfile          # Hugging Face Spaces (Docker)
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env          # then set GROQ_API_KEY
#   free key (no card): https://console.groq.com

# Web UI (watch the agent think):
uvicorn app:app --reload --port 8000     # http://localhost:8000

# …or the CLI:
python cli.py "2024 Nobel Barış Ödülü'nü kim kazandı?"
```

---

## Deploy (Hugging Face Spaces, free)

The repo is deploy-ready (`Dockerfile` + HF front matter in this README, `app_port: 7860`).

1. Create a **Docker** Space at https://huggingface.co/new-space
2. Add `GROQ_API_KEY` as a Space **secret** (Settings → Variables and secrets)
3. Push this repo to the Space's git remote — it builds and runs automatically

`.env` is git-/docker-ignored, so the key never enters the repo or image; it is injected at runtime from the Space secret.

---

## What I learned building this

- Implementing the **agentic (ReAct) loop** myself: defining tool schemas, executing the model's `tool_calls`, and feeding results back as `tool` messages.
- **Function-calling** with the OpenAI-compatible Groq API and multi-step tool selection.
- Streaming intermediate reasoning steps to a browser with **Server-Sent Events**.

*The interesting parts to read first: [`agent/core.py`](agent/core.py) (the loop) and [`agent/tools.py`](agent/tools.py) (tools + schemas).*
