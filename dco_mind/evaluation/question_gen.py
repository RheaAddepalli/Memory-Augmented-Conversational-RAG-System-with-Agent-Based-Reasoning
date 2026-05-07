from dco_mind.models.llm import call_llama


def generate_questions(pdf_path):
    prompt = f"""
You are generating evaluation questions for a document.

Generate 20 diverse questions including:
- factual questions
- follow-up questions
- ambiguous queries
- yes/no verification
- questions that may not have answers

Avoid repetition.

Return as a list.
"""

    output = call_llama(prompt)

    # simple split (can refine later)
    questions = [q.strip("- ").strip() for q in output.split("\n") if q.strip()]

    return questions