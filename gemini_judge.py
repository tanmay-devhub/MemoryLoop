import os
import json
import time
import google.generativeai as genai

JUDGE_SYSTEM_PROMPT = """You are a precision training data
generator for an AI memory system called MemoryLoop.

Your outputs are used DIRECTLY to train an LLM agent.
Every correction you write becomes a lesson the agent
learns from. Imprecise corrections produce bad lessons.
Vague corrections waste training cycles.

YOUR CORE RULES — never break these:

RULE 1 — BE EXACT, NOT DESCRIPTIVE
Wrong: "The answer about floating point is incorrect."
Right: "0.30000000000000004 — print(0.1 + 0.2) never
outputs 0.3 due to IEEE 754 binary representation."

RULE 2 — INCLUDE THE EXACT CORRECT OUTPUT
Every correction must contain the precise answer.
If it's a return value: state exactly what is returned.
If it's output: state the exact printed string.
If it's complexity: state the exact Big-O with reason.

RULE 3 — 2 SENTENCES MAXIMUM
Sentence 1: The exact correct answer.
Sentence 2: Why (mechanism, not advice).
Never write more than 2 sentences.

RULE 4 — NO GENERIC ADVICE
Never write:
- "Always verify your facts"
- "Be more careful with Python"
- "Check the documentation"
- "This depends on the context"
These produce useless lessons.

RULE 5 — STATE THE MECHANISM
Don't just say what's wrong — explain WHY briefly.
"list.sort() returns None because it mutates the list
in-place — use sorted() to get a new list returned."

RULE 6 — RESPOND ONLY IN JSON
No preamble. No explanation outside JSON.
No markdown fences. Raw JSON only.
The system parsing your output has zero tolerance
for non-JSON text."""


def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-2.5-flash", system_instruction=JUDGE_SYSTEM_PROMPT
        )
    except Exception as e:
        print(f"Gemini init error: {e}")
        return None


def judge_answer(question: str, agent_answer: str) -> dict:
    """
    Uses Gemini 2.5 Flash to judge if the agent answer is correct
    and generates a precise correction if wrong.

    Returns:
    {
      "judgment": "CORRECT" or "INCORRECT",
      "correction": "precise 2-sentence correction or null",
      "error_type": "factual_error|incomplete_answer|wrong_complexity|hallucination or null",
      "reasoning": "one sentence explanation",
      "confidence": 0.0-1.0,
      "gemini_available": True or False
    }
    """
    model = get_gemini_model()

    if not model:
        return {
            "judgment": None,
            "correction": None,
            "error_type": None,
            "reasoning": "Gemini not configured",
            "confidence": 0.0,
            "gemini_available": False,
        }

    prompt = f"""EVALUATE THIS AGENT RESPONSE:

Question asked: {question}

Agent's answer: {agent_answer}

JUDGMENT CRITERIA:
Mark CORRECT only if ALL of these are true:
- Every stated fact is accurate
- Key technical details are present
- No misleading or incomplete information
- Output format is correct (for print/output questions)

Mark INCORRECT if ANY of these are true:
- Any fact is wrong
- Critical detail is missing
- Output format differs from actual Python output
- Complexity is wrong
- Answer is vague where precision is required

ERROR TYPE (only if INCORRECT):
- factual_error: wrong facts stated
- incomplete_answer: missing critical details that change the meaning
- wrong_complexity: Big-O or space/time complexity wrong
- hallucination: referenced something that doesn't exist

CORRECTION FORMAT (only if INCORRECT):
Sentence 1: State the exact correct answer.
Sentence 2: State the mechanism/reason why.
Total: 2 sentences maximum. Be surgical. Be exact.

EXAMPLES OF GOOD CORRECTIONS:
Q: What does list.sort() return?
Bad correction: "list.sort() does not return a list."
Good correction: "None. list.sort() sorts the list in-place and returns None — use sorted(list) if you need a new sorted list returned."

Q: What is print(0.1 + 0.2)?
Bad correction: "The output is not 0.3 due to floating point issues."
Good correction: "0.30000000000000004. Python's IEEE 754 floating point representation means 0.1 + 0.2 never equals exactly 0.3 — use round() or math.isclose() for comparisons."

Q: Space complexity of merge sort?
Bad correction: "Merge sort does not have O(1) space."
Good correction: "O(n). Merge sort requires auxiliary space proportional to input size for the temporary arrays created during the merge step."

Now evaluate the agent response above.
Respond with raw JSON only — no text before or after:
{{
  "judgment": "CORRECT" or "INCORRECT",
  "correction": "exact 2-sentence correction or null",
  "error_type": "one of 4 types or null if correct",
  "reasoning": "one sentence: what specifically is wrong or right",
  "confidence": 0.0 to 1.0
}}"""

    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()

            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            if "{" in raw:
                raw = raw[raw.index("{") :]
            if "}" in raw:
                raw = raw[: raw.rindex("}") + 1]

            result = json.loads(raw.strip())

            required = ["judgment", "correction", "error_type", "reasoning"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            if result["judgment"] not in ["CORRECT", "INCORRECT"]:
                raise ValueError(f"Invalid judgment: {result['judgment']}")

            valid_errors = [
                "factual_error",
                "incomplete_answer",
                "wrong_complexity",
                "hallucination",
                None,
            ]
            if result.get("error_type") not in valid_errors:
                result["error_type"] = "factual_error"

            conf = result.get("confidence", 0.8)
            result["confidence"] = max(0.0, min(1.0, float(conf)))

            if result.get("correction"):
                sentences = result["correction"].split(".")
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) > 2:
                    result["correction"] = sentences[0] + ". " + sentences[1] + "."

            result["gemini_available"] = True
            return result

        except Exception as e:
            print(f"Gemini judge attempt {attempt+1} failed: {e}")
            if attempt == 0:
                time.sleep(3)
            else:
                return {
                    "judgment": None,
                    "correction": None,
                    "error_type": None,
                    "reasoning": f"Judge failed: {str(e)[:50]}",
                    "confidence": 0.0,
                    "gemini_available": True,
                }


def validate_api_key(api_key: str) -> bool:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", system_instruction=JUDGE_SYSTEM_PROMPT
        )
        response = model.generate_content(
            '{"judgment": "CORRECT", "correction": null, '
            '"error_type": null, "reasoning": "test", '
            '"confidence": 1.0}'
        )
        return len(response.text.strip()) > 0
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False


def is_gemini_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", ""))
