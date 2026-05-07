import pandas as pd
import streamlit as st

from agent import run_agent
from eval import compile_results, evaluate_single, get_eval_history, load_eval_questions, save_eval_result
from memory import (
    get_all_lessons,
    get_interaction_count,
    get_lesson_count,
    store_interaction,
    update_interaction_outcome,
)
from reflection import reflect, should_reflect

st.set_page_config(page_title="Memory Agent", page_icon="🧠", layout="wide")

# --- Session state init ---
if "interaction_id" not in st.session_state:
    st.session_state.interaction_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "lessons_used" not in st.session_state:
    st.session_state.lessons_used = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# --- Sidebar ---
st.sidebar.title("Memory Agent")
st.sidebar.metric("Total Interactions", get_interaction_count())
st.sidebar.metric("Total Lessons", get_lesson_count())
st.sidebar.divider()
st.sidebar.info(
    "Ollama runs automatically on Windows. If errors appear, "
    "open PowerShell and run: ollama serve"
)

# --- Tabs ---
tab_chat, tab_memory, tab_eval = st.tabs(["Chat", "Memory Browser", "Learning Curve"])

# ── TAB 1: Chat ──────────────────────────────────────────────────────────────
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
            st.session_state.last_query = query
            st.session_state.feedback_given = False

    if st.session_state.last_answer:
        with st.chat_message("assistant"):
            st.write(st.session_state.last_answer)

        with st.expander("Memories used in this response"):
            if st.session_state.lessons_used:
                for lesson in st.session_state.lessons_used:
                    st.markdown(f"- {lesson}")
            else:
                st.caption("No memories retrieved yet — keep chatting and correcting")

        if not st.session_state.feedback_given and st.session_state.interaction_id:
            st.divider()
            st.caption("Was this response correct?")
            col1, col2 = st.columns([1, 2])

            with col1:
                if st.button("Correct", type="primary"):
                    update_interaction_outcome(
                        st.session_state.interaction_id, outcome="correct"
                    )
                    st.session_state.feedback_given = True
                    st.success("Logged!")
                    st.rerun()

            with col2:
                correction = st.text_input("Type the correct answer:", key="correction_input")
                if st.button("Submit correction", type="secondary"):
                    if correction.strip():
                        update_interaction_outcome(
                            st.session_state.interaction_id,
                            outcome="corrected",
                            correction=correction.strip(),
                        )
                        st.session_state.feedback_given = True

                        if should_reflect():
                            with st.spinner("Reflecting on recent failures..."):
                                lesson = reflect()
                            if lesson:
                                st.success(f"New lesson learned: {lesson}")
                        else:
                            st.success("Correction saved!")
                        st.rerun()
                    else:
                        st.warning("Please type a correction before submitting.")

# ── TAB 2: Memory Browser ────────────────────────────────────────────────────
with tab_memory:
    st.subheader("Memory Browser")

    col1, col2 = st.columns(2)
    col1.metric("Lessons learned", get_lesson_count())
    col2.metric("Interactions logged", get_interaction_count())

    st.divider()
    st.subheader("All lessons")

    lessons = get_all_lessons()
    if not lessons:
        st.info(
            "No lessons yet. Go to the Chat tab, ask questions, and submit "
            "corrections. After 5 corrections the agent writes its first lesson."
        )
    else:
        for i, lesson in enumerate(lessons):
            with st.container(border=True):
                st.markdown(f"**Lesson {i + 1}**")
                st.write(lesson["content"])
                st.caption(f"Type: {lesson.get('error_type', 'general')}")

    st.divider()
    if st.button("Force reflection now (for testing)"):
        with st.spinner("Reflecting on recent failures..."):
            result = reflect()
        if result:
            st.success(f"Lesson generated: {result}")
        else:
            st.warning("Not enough failures yet (need at least 3 corrections)")

# ── TAB 3: Learning Curve ────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Learning Curve")

    st.info(
        "The eval set contains 50 fixed questions about Python, CS fundamentals, "
        "programming concepts, and logic. Run it multiple times — after each round of "
        "corrections — to see if accuracy improves. That improvement is proof the "
        "memory system is working."
    )
    st.warning(
        "Running the eval takes 3-5 minutes — Ollama answers 50 questions "
        "plus judges each one."
    )

    if st.button("Run full eval set"):
        questions = load_eval_questions()
        progress = st.progress(0)
        status = st.empty()
        results = []

        for i, q in enumerate(questions):
            status.text(
                f"Evaluating question {i + 1} of {len(questions)}: {q['question'][:50]}..."
            )
            single = evaluate_single(q)
            results.append(single)
            progress.progress((i + 1) / len(questions))

        final = compile_results(results)
        save_eval_result(final)
        status.empty()
        progress.empty()

        st.success(f"Eval complete! Accuracy: {final['accuracy'] * 100:.1f}%")
        st.metric("Correct answers", f"{final['correct']} / {final['total']}")

    history = get_eval_history()
    if history:
        st.subheader("Accuracy over time")
        df = pd.DataFrame(history)
        df["run"] = range(1, len(df) + 1)
        st.line_chart(df.set_index("run")["accuracy"])
        st.dataframe(
            df[["run", "timestamp", "accuracy", "correct", "total"]],
            hide_index=True,
        )
    else:
        st.caption("No eval runs yet. Click the button above to run your first eval.")
