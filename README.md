# Memory Agent

A Streamlit web app that wraps a local Ollama LLM (llama3.2) in a self-improving feedback loop. Every time you correct the agent, it stores the failure in ChromaDB; after every 5 failures it uses the LLM itself to reflect and write a lesson; those lessons are injected into future prompts so the agent gradually gets better all locally, with zero API costs.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine
- llama3.2 pulled: `ollama pull llama3.2`

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
> Ollama starts automatically as a background service, you do not need to run `ollama serve` manually. If you see connection errors, open a second terminal and run `ollama serve`.

## How each tab works

**Chat tab:** Type a question and hit Send. The agent retrieves the 3 most relevant past lessons from ChromaDB, injects them into the system prompt, and calls llama3.2. You see the answer, the agent's confidence score (0–100%), and which lessons (if any) were used. Below the answer, mark it Correct or choose an error type and paste the right answer as a correction. That feedback is written to ChromaDB and used in the next reflection cycle.

**Memory Browser tab:** Shows every lesson the agent has written to itself with a usefulness progress bar, retrieval count, and last-used date. Active lessons are used in future prompts; lessons that score below 0.2 after 30 days of disuse are automatically archived by the decay check. The Confidence Analysis section tracks whether the agent is overconfident on the questions it gets wrong. The Error Breakdown chart shows which of the four error types (factual, incomplete, wrong complexity, hallucination) generate the most corrections.

**Learning Curve tab:** Four eval sets General (50 questions), Algorithms, System Design, Python Deep Dive (20 questions each). Run any set, get a YES/NO judgement from a second LLM call for each question, and see accuracy plotted over time. The Topic Weakness Analysis highlights which domain the agent scores lowest on so you know where to focus corrections.

## The memory system in action: a real example

This is what the feedback loop looks like on a concrete question. The agent was asked: *"Why does 0.1 + 0.2 not equal 0.3 in Python?"*

| Attempt | Lessons injected | Confidence | What happened |
|---------|-----------------|------------|---------------|
| 1 | 0 | 100% | Stated the wrong answer with full confidence no awareness of floating point representation |
| 2 | 1 | 80% | Acknowledged floating point exists but the explanation was still wrong |
| 3 | 2 | 80% | Mentioned the correct value (0.30000000000000004) as a side note but buried it |
| 4 | 3 | 85% | Answered correctly IEEE 754 binary representation explained, correct value given |

Each correction triggered a lesson. By attempt 4, three lessons about floating point precision were being retrieved and injected into the system prompt before the LLM answered. The confidence score dropping from 100% to 80% is also meaningful: the agent learned to be less certain on questions where it had previously been wrong.

This is the core mechanism not fine-tuning, not RAG over documents, just a structured feedback loop that writes its own notes.

## The three loops

```
┌─────────────────────────────────────────────────────────────────┐
│                          INFERENCE LOOP                         │
│                                                                 │
│   User query ──► retrieve_lessons() ──► build system prompt    │
│                                     ──► ollama.chat()          │
│                                     ──► parse answer + conf.   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          STORAGE LOOP                           │
│                                                                 │
│   User feedback ──► update outcome + error_type in ChromaDB    │
│   If correction  ──► store correction, decrement lesson scores  │
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

### Architecture

```
 ┌──────────┐     query      ┌──────────────┐   retrieve   ┌──────────────┐
 │  User    │ ─────────────► │   agent.py   │ ◄──────────► │  memory.py   │
 │ (Browser)│ ◄───────────── │  run_agent() │              │  ChromaDB    │
 └──────────┘  answer+conf   └──────┬───────┘   store      └──────┬───────┘
      │                             │                             │
      │ feedback + error_type       │                             │
      ▼                             ▼                        failures
 ┌──────────┐              ┌──────────────┐                       │
 │  app.py  │─────────────►│reflection.py │◄──────────────────────┘
 │Streamlit │              │  reflect()   │
 └──────────┘              └──────┬───────┘
                                  │ lesson (tagged by error_type)
                                  ▼
                           ┌──────────────┐
                           │  ChromaDB    │
                           │ "lessons"    │
                           │ "interactions│
                           └──────────────┘
```

## What the learning curve proves

Run the eval once before making any corrections that is your baseline. Then chat with the agent, submit corrections, and let lessons accumulate. Run the eval again. If accuracy goes up, the lessons generated by the reflection loop are genuinely being retrieved and applied to new questions the agent has never seen before. The line chart in the Learning Curve tab makes this trend visible across runs. The Topic Weakness Analysis tells you which of the four domains has the most room to improve, so you can direct your corrections efficiently rather than correcting at random.

## Extensions included

| Extension | What it adds |
|-----------|-------------|
| Memory Decay | Lessons unused for 30+ days with usefulness < 0.2 are archived automatically |
| Confidence Scoring | Every answer includes a self-reported confidence (0–100%) tracked against correctness |
| Error Taxonomy | Corrections are classified into 4 types: factual error, incomplete, wrong complexity, hallucination |
| Multi-topic Eval | Four independent eval sets with per-domain accuracy history and weakness detection |
