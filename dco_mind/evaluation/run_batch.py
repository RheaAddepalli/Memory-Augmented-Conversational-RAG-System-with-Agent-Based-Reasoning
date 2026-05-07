import os
from dco_mind.cognition.memory import clear_session
from dco_mind.core.engine import run_pipeline  # your main QA entry
from dco_mind.evaluation.question_gen import generate_questions
from dco_mind.evaluation.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
    compute_recall_at_k,
)

def process_pdf(pdf_path, session_id):
    print(f"\n📄 Processing: {pdf_path}")

    # 1. Generate questions
    questions = generate_questions(pdf_path)

    results = []

    # 2. Run QA for each question
    for q in questions:
        state = run_pipeline(
            question=q,
            pdf_path=pdf_path,
            session_id=session_id
        )
        answer = state.get("answer", "")
        retrieved_docs = state.get("retrieved_docs", [])
        all_chunks = state.get("all_chunks", [])

        # Convert docs → text (safe handling)
        doc_texts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in retrieved_docs
        ]

        # Metrics
        grounding = compute_answer_grounding(answer, doc_texts, q)
        retrieval = compute_retrieval_score(q, retrieved_docs)
        precision = compute_context_precision(q, retrieved_docs)
        recall = compute_recall_at_k(q, retrieved_docs, all_chunks)

        results.append({
            "question": q,
            "answer": answer,
            "decision": state.get("decision_type"),
            "confidence": state.get("confidence"),

            # 🔥 Metrics added
            "grounding": grounding,
            "retrieval_score": retrieval,
            "context_precision": precision,
            "recall": recall,
        })
    # 3. Clear memory between PDFs
    clear_session(session_id)

    return results


def run_batch(pdf_list):
    all_results = {}

    for i, pdf in enumerate(pdf_list):
        session_id = f"session_{i}"
        results = process_pdf(pdf, session_id)
        all_results[pdf] = results

    return all_results


if __name__ == "__main__":
    pdfs = [
        r"C:\Users\Rhearitu\Downloads\rhea AIML resume updated.pdf",
        r"C:\Users\Rhearitu\Downloads\the-story-of-doctor-dolittle.pdf",
        r"C:\Users\Rhearitu\Downloads\ml.pdf"
    ]

    results = run_batch(pdfs)

    # Save results
    import json
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Batch completed")