# MemoryLoop 🧠

> A self-improving AI agent that learns from its own mistakes
> using persistent vector memory and async reflection loops.
> Built with Ollama + ChromaDB. Zero API cost.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Cost](https://img.shields.io/badge/API_Cost-$0.00-brightgreen)

---

## What it does

MemoryLoop implements the Reflexion architecture, an LLM agent
that reflects on its own failures, distills lessons into persistent
vector memory, and retrieves those lessons at inference time to
improve future responses. Unlike standard LLMs that forget
everything between sessions, MemoryLoop accumulates knowledge
through a structured three-loop system. Accuracy is measured on a
fixed 50-question eval set, producing a quantifiable learning curve
that proves the memory system works.

---

## Results

| Metric | Value |
|--------|-------|
| Baseline accuracy (0 lessons) | 36% |
| Accuracy after 5 lessons | 44% |
| Relative improvement | +22% |
| Total interactions logged | 199 |
| Lessons generated | 5 |
| Overconfident errors detected | 22 |
| Avg confidence on wrong answers | 85% |
| Avg confidence on correct answers | 81% |
| Most common error type | Factual error (15 cases) |
| Most retrieved lesson | 111 times |

### Learning curve
![Learning Curve](assets/screenshots/learning_curve.png)
*Accuracy improving from 36% to 44% over 4 eval runs with
5 lessons stored, upward trend proves memory system works*

---

## Architecture: Three Loops

```
User query
↓
[INFERENCE LOOP]: runs on every message
ChromaDB semantic search → retrieve top 3 relevant lessons
Inject lessons into system prompt
Ollama (llama3.2) generates response + confidence score (0-100%)
↓
[STORAGE LOOP]: runs after every interaction
Embed query → store in ChromaDB interactions collection
Log outcome (correct / incorrect / corrected) + error type
Update lesson usefulness scores based on outcome
↓
[REFLECTION LOOP]: triggers every 5 failures
Read recent failures → group by error type
LLM writes ONE specific lesson from failure pattern
Store lesson in ChromaDB with full metadata
```

---

## Key Features

- **Persistent Vector Memory**: ChromaDB stores lessons and
  interactions that survive restarts. Retrieved by semantic
  similarity using sentence-transformers (all-MiniLM-L6-v2).
  Lesson 1 has been retrieved 111 times across 199 interactions.

- **Confidence Scoring**: agent self-reports confidence
  (0-100%) with every response. System tracks overconfident
  errors (high confidence + wrong answer). Detected 22
  overconfident errors agent averaged 85% confidence when
  wrong vs 81% when correct.

- **Error Taxonomy**: corrections classified into 4 types:
  factual_error, incomplete_answer, wrong_complexity,
  hallucination. Factual errors are the most common (15 cases).
  Visualized as a bar chart in Memory Browser.

- **Memory Decay**: lessons unused for 30 days with
  usefulness score below 0.2 are automatically archived,
  keeping memory clean and relevant over time.

- **Multi-topic Eval Sets**: 4 separate eval sets (General
  50q, Algorithms 20q, System Design 20q, Python Deep Dive 20q)
  with independent accuracy tracking per domain.

### Memory Browser
![Memory Browser](assets/screenshots/memory_browser.png)
*5 lessons with usefulness scores, retrieval counts, confidence
analysis and error breakdown chart*

### Memories retrieved in action
![Memories Retrieved](assets/screenshots/memories_retrieved.webp)
*3 lessons simultaneously retrieved and injected into a single
response semantic similarity retrieval working in real time*

### Sidebar stats
![Sidebar](assets/screenshots/sidebar_stats.png)
*Live stats: 199 interactions, 5 lessons generated*

---

## Research Basis

This project implements and extends the **Reflexion**
architecture (Shinn et al., 2023 Stanford/Northeastern):

> "Reflexion: Language Agents with Verbal Reinforcement Learning"
> https://arxiv.org/abs/2303.11366

**Key extension over the original paper:** persistent external
vector store (ChromaDB) instead of in-context storage, enabling
lessons to survive across sessions and be retrieved by semantic
similarity rather than recency.

Also draws from **MemGPT / Letta** (Packer et al., 2023):
> https://arxiv.org/abs/2310.08560

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Ollama (llama3.2) | Local inference, zero cost |
| Embeddings | sentence-transformers | Semantic similarity search |
| Vector store | ChromaDB (persistent) | Lesson + interaction memory |
| UI | Streamlit | 3-tab dashboard |
| Language | Python 3.10+ | Backend + agent logic |
| Cost | $0.00 | Fully local, no API keys |

---

## Quick Start

```bash
# Prerequisites: Python 3.10+, Ollama installed
ollama pull llama3.2

cd memory-agent
pip install -r requirements.txt
streamlit run app.py
```

App opens at http://localhost:8501

> **Windows users:** Ollama starts automatically on login.
> No need to run `ollama serve` manually.

---

## How to Use

**Chat tab**
Ask questions, mark answers correct or wrong, submit corrections
with error type classification. After every 5 corrections the
reflection loop fires and writes a new lesson.

**Memory Browser**
View all lessons with usefulness scores, retrieval counts,
last used timestamps. See confidence analysis (overconfident
errors, underconfident correct answers) and error breakdown
chart by category. Run decay check to archive stale lessons.

**Learning Curve**
Run any of 4 eval sets and track accuracy over time. Compare
domains side by side. Topic Weakness Analysis shows which
domain needs the most corrections for fastest improvement.

---

## What I Learned

Building MemoryLoop revealed that the most dangerous LLM failure
mode is high-confidence wrong answers the agent averaged 85%
confidence on incorrect responses vs 81% on correct ones,
meaning confidence alone is not a reliable quality signal.
The reflection loop generated significantly more useful lessons
when corrections were specific and included exact expected outputs
rather than general descriptions. Memory retrieval improved
response quality measurably, but lesson quality matters more
than lesson quantity one precise lesson outperforms five
vague ones.

---

## License

MIT © 2026 Tanmay Chaudhari
