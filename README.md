# Memory Agent

A Streamlit web app that wraps a local Ollama LLM (llama3.2) in a self-improving feedback loop. Every time you correct the agent, it stores the failure in ChromaDB; after every 5 failures it uses the LLM itself to reflect and write a lesson; those lessons are then injected into future prompts so the agent gradually gets better all locally, with zero API costs.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine
- llama3.2 model pulled: `ollama pull llama3.2`

## Setup

```bash
cd memory-agent
pip install -r requirements.txt
streamlit run app.py
```

> **Windows users:** If `pip` or `streamlit` are not recognised as commands (common when Python is installed from the Microsoft Store or without modifying PATH), use the `python -m` prefix instead:
> ```powershell
> python -m pip install -r requirements.txt
> python -m streamlit run app.py
> ```
> Ollama starts automatically as a background service you do not need to run `ollama serve` manually. If you see connection errors, open a second terminal and run `ollama serve`.

## How each tab works

**Chat tab:** Type a question and hit Send. The agent retrieves the 3 most relevant past lessons from ChromaDB, injects them into the system prompt, and calls llama3.2. You see the answer plus which lessons (if any) were used. Below the answer, mark it Correct or paste the right answer. That feedback is written back to the interactions collection.

**Memory Browser tab:** Shows all lessons the agent has written to itself, with their error type and an ordinal number. You can also click "Force reflection now" to trigger a reflection cycle immediately useful for testing before you have accumulated 5 failures organically.

**Learning Curve tab:** Runs a fixed eval set of 50 questions (Python, CS fundamentals, programming concepts, logic). For each question, the agent answers and a second LLM call judges correctness with YES/NO. Results are appended to `eval_history.json` so you can plot accuracy over multiple runs and watch it improve as lessons accumulate.

## The three loops

```
┌─────────────────────────────────────────────────────────────────┐
│                          INFERENCE LOOP                         │
│                                                                 │
│   User query ──► retrieve_lessons() ──► build system prompt    │
│                                     ──► ollama.chat()          │
│                                     ──► return answer          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          STORAGE LOOP                           │
│                                                                 │
│   User feedback ──► update outcome in ChromaDB                 │
│   If correction  ──► store correction text                      │
│   Check: failures % 5 == 0? ──► trigger reflection             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        REFLECTION LOOP                          │
│                                                                 │
│   get_recent_failures(10) ──► format as failure cases          │
│   ollama.chat() ──► generate ONE 2-sentence lesson             │
│   store_lesson() ──► persisted in ChromaDB "lessons" coll.     │
│   Next query ──► lesson retrieved and injected into prompt     │
└─────────────────────────────────────────────────────────────────┘
```

### ASCII architecture

```
 ┌──────────┐     query      ┌──────────────┐   retrieve   ┌──────────────┐
 │  User    │ ─────────────► │   agent.py   │ ◄──────────► │  memory.py   │
 │ (Browser)│ ◄───────────── │  run_agent() │              │  ChromaDB    │
 └──────────┘    answer      └──────┬───────┘   store      └──────┬───────┘
      │                             │                             │
      │ feedback/correction         │                             │
      ▼                             ▼                        failures
 ┌──────────┐              ┌──────────────┐                       │
 │  app.py  │─────────────►│reflection.py │◄──────────────────────┘
 │Streamlit │              │  reflect()   │
 └──────────┘              └──────┬───────┘
                                  │ lesson
                                  ▼
                           ┌──────────────┐
                           │  ChromaDB    │
                           │ "lessons"    │
                           └──────────────┘
```

## What the learning curve proves

Run the eval once before making any corrections this is your baseline. Then chat with the agent, submit corrections, and let lessons accumulate. Run the eval again. If the accuracy number goes up, it means the lessons written by the reflection loop are genuinely being retrieved and applied to new questions. The line chart in the Learning Curve tab makes this trend visible over time, giving you a quantitative signal that the memory system is working rather than just storing data.
