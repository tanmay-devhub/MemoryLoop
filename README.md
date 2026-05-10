# MemoryLoop

A self-improving AI agent that learns from mistakes using persistent vector memory, reflection loops, and automated evaluation.

MemoryLoop is an experimental AI agent framework that improves over time by storing user corrections as reusable lessons. It runs locally with Ollama, retrieves relevant lessons from ChromaDB during inference, and tracks whether memory improves accuracy across evaluation sets.

Core inference runs locally with Ollama. The optional Gemini judge can be enabled with a free-tier API key for automated answer evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Cost](https://img.shields.io/badge/API_Cost-$0.00-brightgreen)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-blue)

---

## What It Does

MemoryLoop helps an AI agent improve through feedback-driven memory.

Core workflow:

1. The user asks a question.
2. The agent retrieves relevant lessons from ChromaDB.
3. The local LLM answers using the retrieved lessons.
4. The user or Gemini judge evaluates the answer.
5. If the answer is wrong, the correction is stored as a reusable memory.
6. Future answers can use those lessons to avoid repeated mistakes.

---

## Architecture

MemoryLoop is organized around three loops:

### Inference Loop

Retrieves relevant lessons from vector memory and injects them into the prompt before generating an answer.

### Storage Loop

Stores corrections, feedback, metadata, confidence scores, and error categories as searchable memories.

### Reflection Loop

Turns repeated failures into reusable lessons that can improve future responses.

```text
User Question
    |
Memory Retrieval (ChromaDB semantic search)
    |
Prompt with Relevant Lessons
    |
Ollama LLM Response + Confidence Score
    |
User / Gemini Evaluation
    |
Correction Stored in ChromaDB
    |
Future Improved Responses
```

---

## Features

- Local LLM inference using Ollama and llama3.2
- Persistent vector memory using ChromaDB
- Semantic retrieval using sentence-transformers (all-MiniLM-L6-v2)
- Feedback-based learning from incorrect answers
- Reflection-style lesson generation every 5 failures
- Confidence tracking (0-100%) on every response
- Overconfident error detection (high confidence + wrong answer)
- Error taxonomy: factual errors, incomplete answers, complexity errors, hallucinations
- Memory decay: stale lessons archived automatically after 30 days
- Streamlit dashboard with Chat, Memory Browser, and Learning Curve tabs
- Optional Gemini 2.5 Flash judge for automated evaluation and correction pre-fill
- Four evaluation sets for measuring improvement over time

---

## Current Results

Initial fixed-set evaluation showed:

| Metric | Value |
|--------|------:|
| Baseline accuracy | 36% |
| Accuracy after memory lessons | 44% |
| Relative improvement | +22% |
| Interactions logged | 199 |
| Overconfident errors detected | 22 |
| Lessons generated | 5 |
| Most retrieved lesson | 111 times |

These results are from a small curated evaluation set and are intended to demonstrate the learning loop, not claim benchmark-level performance.

### Learning Curve

![Learning Curve](assets/screenshots/learning_curve.png)

*Accuracy improving from 36% to 44% over 4 eval runs with 5 lessons stored.*

---

## Key Features

**Persistent Vector Memory**

ChromaDB stores lessons and interactions that survive restarts. Retrieved by semantic similarity using sentence-transformers. Lesson 1 has been retrieved 111 times across 199 interactions.

**Confidence Scoring**

The agent self-reports confidence (0-100%) with every response. The system tracks overconfident errors (high confidence but wrong answer). Detected 22 overconfident errors - the agent averaged 85% confidence when wrong vs 81% when correct.

**Error Taxonomy**

Corrections are classified into 4 types: factual_error, incomplete_answer, wrong_complexity, hallucination. Factual errors are the most common (15 cases). Visualized as a bar chart in Memory Browser.

**Memory Decay**

Lessons unused for 30 days with usefulness score below 0.2 are automatically archived, keeping memory clean and relevant over time.

**Multi-topic Evaluation Sets**

4 separate eval sets (General 50q, Algorithms 20q, System Design 20q, Python Deep Dive 20q) with independent accuracy tracking per domain.

**Gemini 2.5 Flash Auto-Judge**

Every agent response is automatically evaluated by Gemini 2.5 Flash in the background using a training-data-aware system prompt. Enforces exact 2-sentence corrections with stated mechanisms (no generic advice) since outputs are used directly as training data. If correct: instant green verification banner. If wrong: error type and correction pre-filled automatically, user confirms with one click. Uses Gemini free tier at aistudio.google.com.

---

## Demo

### Memory Browser

![Memory Browser](assets/screenshots/memory_browser.png)

*5 lessons with usefulness scores, retrieval counts, confidence analysis, and error breakdown chart.*

### Retrieved Lessons in Action

![Memories Retrieved](assets/screenshots/memories_retrieved.webp)

*3 lessons simultaneously retrieved and injected into a single response.*

### Sidebar Stats

![Sidebar](assets/screenshots/sidebar_stats.png)

*Live stats: 199 interactions, 5 lessons generated.*

---

## Gemini Precision System Prompt

A core design decision in MemoryLoop is that Gemini is not used as a conversational assistant. It is used as a precision training data generator. Every correction it produces becomes a lesson the agent learns from. Imprecise corrections produce bad lessons.

The system prompt enforces 6 rules on every Gemini call:

| Rule | Requirement |
|------|-------------|
| Be exact | State the precise correct answer, not a description |
| Include exact output | Return values, print output, Big-O must be exact |
| 2 sentences max | Sentence 1: correct answer. Sentence 2: mechanism |
| No generic advice | Never: "verify your facts", "check documentation" |
| State the mechanism | Explain WHY, not just WHAT is wrong |
| JSON only | Raw JSON response, zero tolerance for extra text |

**Without precision prompt:**

> "The answer about list.sort() is incorrect. You should check the Python documentation for the correct behavior."

**With precision prompt:**

> "None. list.sort() sorts the list in-place and returns None - use sorted(list) to get a new sorted list returned."

The second correction produces a lesson the agent can actually apply. The first produces noise.

---

## Research Basis

This project implements and extends the **Reflexion** architecture (Shinn et al., 2023 - Stanford/Northeastern):

> "Reflexion: Language Agents with Verbal Reinforcement Learning"
> https://arxiv.org/abs/2303.11366

Key extension over the original paper: persistent external vector store (ChromaDB) instead of in-context storage, enabling lessons to survive across sessions and be retrieved by semantic similarity rather than recency.

Also draws from **MemGPT / Letta** (Packer et al., 2023):
> https://arxiv.org/abs/2310.08560

---

## Tech Stack

- Python 3.10+
- Streamlit
- Ollama (llama3.2)
- ChromaDB
- Sentence Transformers (all-MiniLM-L6-v2)
- Gemini 2.5 Flash (optional)
- Pandas

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tanmay-devhub/MemoryLoop.git
cd MemoryLoop
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

```bash
ollama pull llama3.2
ollama serve
```

### 5. Run the app

```bash
streamlit run app.py
```

App opens at http://localhost:8501

---

## How to Use

**Chat**

Ask questions, mark answers correct or wrong, and submit corrections with error type classification. After every 5 corrections the reflection loop fires and writes a new lesson.

**Memory Browser**

View all lessons with usefulness scores, retrieval counts, and last used timestamps. See confidence analysis (overconfident errors, underconfident correct answers) and error breakdown by category. Run decay check to archive stale lessons.

**Learning Curve**

Run any of 4 eval sets and track accuracy over time. Compare domains side by side. Topic Weakness Analysis shows which domain needs the most corrections for fastest improvement.

---

## Why This Project Matters

Most LLM applications generate responses but do not improve from user feedback. MemoryLoop explores how an AI agent can convert mistakes into persistent lessons and use them during future inference. This makes the project relevant to AI agents, evaluation systems, RAG, memory-augmented generation, and applied LLM engineering.

---

## Limitations

- Current evaluation uses small curated question sets.
- Memory quality depends on the quality of user corrections.
- The system improves only when feedback is specific and accurate.
- Retrieval quality can affect whether the correct lesson is used.
- Future work should compare against larger benchmarks and non-memory baselines.

---

## Future Improvements

- Add larger benchmark evaluation sets
- Add regression testing for memory improvements
- Add support for multiple local LLMs
- Improve lesson ranking and memory pruning
- Add Docker support
- Add hosted demo or demo video
- Add exportable evaluation reports
- Add comparison between memory-enabled and memory-disabled modes

---

## Project Structure

```text
MemoryLoop/
|-- app.py                      # Streamlit UI (Chat, Memory Browser, Learning Curve)
|-- agent.py                    # Core agent logic and LLM inference
|-- memory.py                   # ChromaDB memory storage and retrieval
|-- reflection.py               # Lesson generation from failure patterns
|-- eval.py                     # Evaluation runner and metrics
|-- gemini_judge.py             # Optional Gemini 2.5 Flash auto-judge
|-- requirements.txt            # Python dependencies
|-- .env.example                # Example environment variables
|-- eval_set.json               # General evaluation questions (50)
|-- eval_set_algorithms.json    # Algorithm questions (20)
|-- eval_set_systems.json       # System design questions (20)
|-- eval_set_python.json        # Python deep dive questions (20)
|-- assets/screenshots/         # UI screenshots
```

---

## What I Learned

Building MemoryLoop revealed that the most dangerous LLM failure mode is high-confidence wrong answers. The agent averaged 85% confidence on incorrect responses vs 81% on correct ones, meaning confidence alone is not a reliable quality signal. The reflection loop generated significantly more useful lessons when corrections were specific and included exact expected outputs rather than general descriptions. Memory retrieval improved response quality measurably, but lesson quality matters more than lesson quantity - one precise lesson outperforms five vague ones.

Integrating Gemini as a precision judge revealed an important prompt engineering insight: the same model produces dramatically different output quality depending on whether it is prompted as a conversational assistant or as a specialized data pipeline component. Framing Gemini as a "training data generator" rather than a "helpful assistant" and enforcing strict output constraints via system prompt produced corrections that were significantly more specific and actionable than unguided generation.

---

## License

MIT (c) 2026 Tanmay Chaudhari
