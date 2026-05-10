from datetime import datetime

import pandas as pd
import streamlit as st

from agent import run_agent
from eval import (
    EVAL_SETS,
    compile_results,
    evaluate_single,
    get_eval_history,
    load_eval_questions,
    save_eval_result,
)
from memory import (
    get_active_lessons_count,
    get_all_lessons,
    get_archived_lessons_count,
    get_confidence_stats,
    get_error_breakdown,
    get_interaction_count,
    get_lesson_count,
    get_lessons_by_error_type,
    run_decay_check,
    update_interaction_outcome,
    update_lesson_usage,
)
from reflection import reflect, should_reflect
from gemini_judge import judge_answer, validate_api_key, is_gemini_configured
import os

st.set_page_config(page_title="MemoryLoop", page_icon="🧠", layout="wide")

# ── session state ─────────────────────────────────────────────────────────────
if "interaction_id" not in st.session_state:
    st.session_state.interaction_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "lessons_used" not in st.session_state:
    st.session_state.lessons_used = []
if "lesson_ids_used" not in st.session_state:
    st.session_state.lesson_ids_used = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = None
if "gemini_judgment" not in st.session_state:
    st.session_state.gemini_judgment = None
if "gemini_key_set" not in st.session_state:
    st.session_state.gemini_key_set = False
if "judged_query" not in st.session_state:
    st.session_state.judged_query = None


# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_date(iso_str: str) -> str:
    try:
        return datetime.fromisoformat(iso_str).strftime("%b %d, %Y %H:%M")
    except Exception:
        return iso_str or "Never"


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("MemoryLoop 🧠")
st.sidebar.metric("Total Interactions", get_interaction_count())
st.sidebar.metric("Total Lessons", get_lesson_count())
st.sidebar.divider()
st.sidebar.info(
    "Ollama runs automatically on Windows. If errors appear, "
    "open PowerShell and run: ollama serve"
)
st.sidebar.divider()
st.sidebar.subheader("Gemini Auto-Judge")
st.sidebar.caption(
    "Gemini 2.5 Flash judges every answer automatically "
    "and pre-fills corrections. One click to confirm."
)
gemini_input = st.sidebar.text_input(
    "Gemini API Key:",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Free at aistudio.google.com — no credit card needed",
    key="gemini_api_input",
)
if gemini_input:
    os.environ["GEMINI_API_KEY"] = gemini_input
    if not st.session_state.gemini_key_set:
        st.session_state.gemini_key_set = True
    st.sidebar.success("Gemini 2.5 Flash ready ✓")
else:
    st.sidebar.caption("Add key to enable auto-judgment")

if gemini_input:
    if st.sidebar.button("Test Gemini key"):
        with st.sidebar:
            with st.spinner("Testing..."):
                valid = validate_api_key(gemini_input)
            if valid:
                st.sidebar.success("Gemini 2.5 Flash connected ✓")
            else:
                st.sidebar.error("Invalid key. Check aistudio.google.com")

st.sidebar.divider()
st.sidebar.caption("📚 Based on Reflexion (Stanford, 2023)")
st.sidebar.caption("💰 Cost: $0.00 — fully local")
st.sidebar.caption("🔗 github.com/tanmay-devhub/MemoryLoop")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_memory, tab_eval = st.tabs(["Chat", "Memory Browser", "Learning Curve"])

# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Chat with memory")

    with st.form("chat_form", clear_on_submit=False):
        query = st.text_input("Ask a question:", key="query_input")
        submitted = st.form_submit_button("Send")

    if submitted and query:
        if query != st.session_state.last_query:
            with st.spinner("Thinking..."):
                result = run_agent(query)
            st.session_state.last_answer = result["answer"]
            st.session_state.interaction_id = result["interaction_id"]
            st.session_state.lessons_used = result["lessons_used"]
            st.session_state.lesson_ids_used = result["lesson_ids_used"]
            st.session_state.current_confidence = result["confidence"]
            st.session_state.last_query = query
            st.session_state.feedback_given = False

    if st.session_state.last_answer:
        with st.chat_message("assistant"):
            st.write(st.session_state.last_answer)

        # confidence display
        if st.session_state.current_confidence is not None:
            conf = st.session_state.current_confidence
            if conf >= 0.8:
                st.success(f"Confidence: {conf:.0%}")
            elif conf >= 0.5:
                st.info(f"Confidence: {conf:.0%}")
            else:
                st.warning(f"Confidence: {conf:.0%}")
            st.progress(conf, text="Agent confidence")

        with st.expander("Memories used in this response"):
            if st.session_state.lessons_used:
                for lesson in st.session_state.lessons_used:
                    st.markdown(f"- {lesson['content']}")
            else:
                st.caption("No memories retrieved yet — keep chatting and correcting")

        # ── Gemini Auto-Judge ──────────────────────────────────────────────────
        if (
            st.session_state.last_answer
            and is_gemini_configured()
            and not st.session_state.feedback_given
        ):

            query_changed = (
                st.session_state.get("judged_query") != st.session_state.last_query
            )

            if st.session_state.gemini_judgment is None or query_changed:
                with st.spinner("Gemini 2.5 Flash analyzing answer..."):
                    judgment = judge_answer(
                        st.session_state.last_query, st.session_state.last_answer
                    )
                st.session_state.gemini_judgment = judgment
                st.session_state.judged_query = st.session_state.last_query

            j = st.session_state.gemini_judgment

            if j and j.get("gemini_available"):
                if j.get("judgment") == "CORRECT":
                    st.success(
                        f"✓ Gemini verified: Correct  "
                        f"(confidence: {j.get('confidence', 0)*100:.0f}%)"
                    )
                    st.caption(f"Gemini: {j.get('reasoning', '')}")
                elif j.get("judgment") == "INCORRECT":
                    st.error(
                        "✗ Gemini detected an error — "
                        "correction pre-filled below. "
                        "Review and click Submit."
                    )
                    st.caption(f"Gemini: {j.get('reasoning', '')}")
                elif j.get("judgment") is None:
                    st.caption(
                        f"⚠ Gemini unavailable: "
                        f"{j.get('reasoning', 'unknown error')}"
                    )
            elif j and not j.get("gemini_available"):
                st.caption("💡 Add Gemini API key in sidebar for auto-judgment")
        # ── End Gemini Auto-Judge ──────────────────────────────────────────────

        if not st.session_state.feedback_given and st.session_state.interaction_id:
            st.divider()
            st.caption("Was this response correct?")
            col1, col2 = st.columns([1, 2])

            with col1:
                if st.button("Correct", type="primary"):
                    update_interaction_outcome(
                        st.session_state.interaction_id, outcome="correct"
                    )
                    for lid in st.session_state.lesson_ids_used:
                        update_lesson_usage(lid, was_helpful=True)
                    st.session_state.feedback_given = True
                    st.success("Logged!")
                    st.rerun()

            with col2:
                j = st.session_state.get("gemini_judgment") or {}

                gemini_wrong = (
                    j.get("gemini_available") and j.get("judgment") == "INCORRECT"
                )
                suggested_error = j.get("error_type") or "factual_error"
                suggested_correction = j.get("correction") or ""

                error_options = [
                    "factual_error",
                    "incomplete_answer",
                    "wrong_complexity",
                    "hallucination",
                ]
                default_index = (
                    error_options.index(suggested_error)
                    if suggested_error in error_options
                    else 0
                )

                if gemini_wrong and suggested_correction:
                    st.caption(
                        "✨ Pre-filled by Gemini 2.5 Flash — " "review and confirm"
                    )

                error_type = st.selectbox(
                    "What type of error was this?",
                    options=error_options,
                    index=default_index,
                    format_func=lambda x: {
                        "factual_error": "Factual error — stated something wrong",
                        "incomplete_answer": "Incomplete — missing key details",
                        "wrong_complexity": "Wrong complexity — incorrect Big-O",
                        "hallucination": "Hallucination — made something up",
                    }[x],
                    key="error_type_select",
                )

                correction = st.text_area(
                    "Correct answer:",
                    value=suggested_correction,
                    height=100,
                    placeholder=(
                        "Gemini pre-fills this when answer is wrong. "
                        "You can edit before submitting."
                    ),
                    key="correction_input",
                )

                if st.button(
                    "Submit correction", type="secondary", key="submit_correction_btn"
                ):
                    if correction.strip():
                        update_interaction_outcome(
                            st.session_state.interaction_id,
                            outcome="corrected",
                            correction=correction.strip(),
                            error_type=error_type,
                        )
                        for lid in st.session_state.lesson_ids_used:
                            update_lesson_usage(lid, was_helpful=False)
                        st.session_state.feedback_given = True
                        st.session_state.gemini_judgment = None
                        st.session_state.judged_query = None

                        if should_reflect():
                            with st.spinner("Reflecting on recent failures..."):
                                lesson = reflect(error_type=error_type)
                            if lesson:
                                st.success(f"New lesson learned: {lesson}")
                        else:
                            st.success("Correction saved!")
                        st.rerun()
                    else:
                        st.warning("Please type a correction before submitting.")

# ── TAB 2: Memory Browser ─────────────────────────────────────────────────────
with tab_memory:
    st.subheader("Memory Browser")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lessons learned", get_lesson_count())
    c2.metric("Interactions logged", get_interaction_count())
    c3.metric("Active Lessons", get_active_lessons_count())
    c4.metric("Archived Lessons", get_archived_lessons_count())

    # ── confidence analysis ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Confidence Analysis")
    stats = get_confidence_stats()
    if stats["total_scored"] == 0:
        st.caption(
            "No scored interactions yet. Mark responses as correct or incorrect in the Chat tab."
        )
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(
            "Avg confidence (correct)",
            f"{stats['avg_confidence_correct']:.0%}",
        )
        sc2.metric(
            "Avg confidence (wrong)",
            f"{stats['avg_confidence_incorrect']:.0%}",
        )
        sc3.metric(
            "Overconfident errors",
            stats["overconfident_count"],
            help="Answered with >70% confidence but was wrong",
        )
        sc4.metric(
            "Underconfident correct",
            stats["underconfident_count"],
            help="Answered with <40% confidence but was right",
        )
        st.caption(
            "Overconfident errors are the most dangerous: the agent was certain but wrong. "
            "Focus corrections on those patterns first."
        )

    # ── error breakdown ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Error Breakdown")
    breakdown = get_error_breakdown()
    if any(v > 0 for v in breakdown.values()):
        df_bd = pd.DataFrame(
            {"Count": list(breakdown.values())},
            index=list(breakdown.keys()),
        )
        st.bar_chart(df_bd)
        most_common = max(breakdown.items(), key=lambda x: x[1])
        st.caption(f"Most common error: **{most_common[0]}** ({most_common[1]} times)")
    else:
        st.caption("No corrections submitted yet.")

    # ── lessons ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("All lessons")

    filter_type = st.selectbox(
        "Filter by error type:",
        options=[
            "All",
            "factual_error",
            "incomplete_answer",
            "wrong_complexity",
            "hallucination",
        ],
        key="lesson_filter",
    )

    lessons = (
        get_all_lessons()
        if filter_type == "All"
        else get_lessons_by_error_type(filter_type)
    )

    if not lessons:
        st.info(
            "No lessons yet. Go to the Chat tab, ask questions, and submit "
            "corrections. After 5 corrections the agent writes its first lesson."
        )
    else:
        for i, lesson in enumerate(lessons):
            with st.container(border=True):
                header_col, badge_col = st.columns([5, 1])
                with header_col:
                    st.markdown(f"**Lesson {i + 1}**")
                with badge_col:
                    if not lesson.get("is_active", True):
                        st.caption("archived")
                st.write(lesson["content"])
                st.progress(
                    lesson["usefulness_score"],
                    text=f"Usefulness: {lesson['usefulness_score']:.2f}",
                )
                st.caption(
                    f"Type: {lesson.get('error_type', 'general')}  ·  "
                    f"Retrieved {lesson['times_retrieved']} times  ·  "
                    f"Last used: {_fmt_date(lesson['last_used_at'])}"
                )

    st.divider()

    if st.button("Run decay check"):
        decay_result = run_decay_check()
        st.info(
            f"Checked {decay_result['checked']} lessons. "
            f"Archived: {decay_result['archived']}. "
            f"Kept active: {decay_result['kept']}."
        )

    if st.button("Force reflection now (for testing)"):
        with st.spinner("Reflecting on recent failures..."):
            forced = reflect()
        if forced:
            st.success(f"Lesson generated: {forced}")
        else:
            st.warning("Not enough failures yet (need at least 3 corrections)")

# ── TAB 3: Learning Curve ─────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Learning Curve")

    st.info(
        "Each eval set tests a specific domain. Run the same set multiple times — "
        "after submitting corrections — to watch accuracy improve. "
        "That improvement is proof the memory system is working."
    )
    st.warning(
        "Each run takes 3-5 minutes for the 50-question set, "
        "or ~1-2 minutes for the 20-question sets."
    )

    selected_eval = st.selectbox(
        "Choose eval set:",
        options=list(EVAL_SETS.keys()),
        key="selected_eval_set",
    )

    if st.button("Run selected eval"):
        questions = load_eval_questions(selected_eval)
        if not questions:
            st.error(f"Could not load eval set: {selected_eval}")
        else:
            progress = st.progress(0)
            status = st.empty()
            results = []
            for i, q in enumerate(questions):
                status.text(
                    f"Evaluating question {i + 1} of {len(questions)}: "
                    f"{q['question'][:50]}..."
                )
                results.append(evaluate_single(q))
                progress.progress((i + 1) / len(questions))
            final = compile_results(results, selected_eval)
            save_eval_result(final)
            status.empty()
            progress.empty()
            st.success(f"Eval complete! Accuracy: {final['accuracy'] * 100:.1f}%")
            st.metric("Correct answers", f"{final['correct']} / {final['total']}")

    # ── history tabs ──────────────────────────────────────────────────────────
    history = get_eval_history()
    if history:
        by_set: dict[str, list[dict]] = {}
        for record in history:
            name = record.get("eval_set_name", "General (50 questions)")
            by_set.setdefault(name, []).append(record)

        tab_labels = list(by_set.keys()) + ["Compare All"]
        hist_tabs = st.tabs(tab_labels)

        for i, (set_name, records) in enumerate(by_set.items()):
            with hist_tabs[i]:
                st.subheader(f"Accuracy over time — {set_name}")
                df = pd.DataFrame(records)
                df["run"] = range(1, len(df) + 1)
                st.line_chart(df.set_index("run")["accuracy"])
                st.dataframe(
                    df[["run", "timestamp", "accuracy", "correct", "total"]],
                    hide_index=True,
                )

        with hist_tabs[-1]:
            st.subheader("Compare all eval sets")
            max_runs = max(len(v) for v in by_set.values())
            compare_data: dict[str, list] = {}
            for name, records in by_set.items():
                accs = [r["accuracy"] for r in records]
                compare_data[name] = accs + [None] * (max_runs - len(accs))
            df_compare = pd.DataFrame(compare_data, index=range(1, max_runs + 1))
            df_compare.index.name = "run"
            st.line_chart(df_compare)

        # ── weakness analysis ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Topic Weakness Analysis")
        latest_by_set: dict[str, float] = {}
        for record in history:
            latest_by_set[record.get("eval_set_name", "General (50 questions)")] = (
                record["accuracy"]
            )

        metric_cols = st.columns(len(EVAL_SETS))
        for idx, (name, _) in enumerate(EVAL_SETS.items()):
            with metric_cols[idx]:
                short = name.split("(")[0].strip()
                if name in latest_by_set:
                    st.metric(short, f"{latest_by_set[name]:.0%}")
                else:
                    st.caption(f"**{short}**")
                    st.caption("Not yet evaluated")

        if latest_by_set:
            weakest_name, weakest_acc = min(latest_by_set.items(), key=lambda x: x[1])
            st.error(
                f"Weakest topic: **{weakest_name}** at {weakest_acc:.0%} — "
                "focus your corrections here for fastest improvement"
            )
        else:
            st.info("Run all eval sets to see weakness analysis")
    else:
        st.caption("No eval runs yet. Choose an eval set above and click Run.")
