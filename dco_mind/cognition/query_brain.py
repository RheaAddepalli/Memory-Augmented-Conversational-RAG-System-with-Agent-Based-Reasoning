import re
from dco_mind.config.settings import qa_pipeline, FACTUAL_TOP_K

from dco_mind.models.llm import call_llama, call_llama_streaming
from dco_mind.generation.response_generator import REACT_PROMPT, QA_PROMPT

from dco_mind.retrieval.reranker import rerank_docs, protect_exact_matches
from dco_mind.retrieval.adaptive_search import multi_query_retrieve

from dco_mind.events.events import emit_event

from dco_mind.utils.helpers import (
    clean_artifacts, _is_cop_out_answer,
    clean_chunk_text, validate_and_correct_span
)

from dco_mind.reasoning.context_builder import (
    reorder_by_question, extract_numeric_answer
)

from dco_mind.evaluation.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
)
from dco_mind.cognition.memory import get_memory_context
# ============================================================
# HELPERS
# ============================================================

def should_validate(answer: str, question: str) -> bool:
    answer = answer.strip()
    if len(answer.split()) <= 12:
        return True
    if "\n" in answer or "," in answer:
        return False
    if question.lower().startswith(("what is", "who is", "name", "define")):
        return True
    return False


def _is_echo_answer(answer: str, question: str) -> bool:
    q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    return len(a_tokens - q_tokens) == 0


# ============================================================
# FIX — Generic reasoning/echo filter
# Replaces hardcoded startswith("since", "because", ...) check.
# Uses token overlap: if answer adds fewer than 3 new tokens
# beyond what's in the question, it's likely an echo or
# reasoning preamble rather than a real answer.
# Short answers (<=3 words) are never wiped — they may be
# legitimate single-word or short-phrase answers.
# ============================================================

def clean_reasoning_answer(answer: str, question: str) -> str:
    """
    Remove answers that are pure echoes of the question.
    Generic — no hardcoded trigger words.
    """
    words = answer.strip().split()

    # Never wipe short answers — they may be legitimate facts
    if len(words) <= 3:
        return answer

    q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
    a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    new_tokens = a_tokens - q_tokens

    if len(new_tokens) < 3:
        print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
        return ""

    return answer


# ============================================================
# RoBERTa QA — kept for import compatibility, not called
# ============================================================

def roberta_qa(question: str, chunks: list):
    """Kept for import compatibility. Not called in current pipeline."""
    return "", 0.0


# ============================================================
# ReAct AGENT
# ============================================================

def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score, memory_context=""):
    grounding = 0.0
    """
    Returns:
        answer, model_used, steps, retrieval_score,
        context_precision, grounding, llm_calls, context_chunks
    """
    MAX_STEPS  = 3
    scratchpad = ""
    model_used = "llama_react"
    llm_calls  = 0

    # ── Retrieval ─────────────────────────────────────────────
    # ── Retrieval ─────────────────────────────────────────────
    context_chunks = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    retrieval_score = compute_retrieval_score(question, context_chunks)
    context_precision = compute_context_precision(question, context_chunks)
   

    print(
        f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
        f"retrieval={retrieval_score:.1f}%"
    )

    emit_event(
        request_id,
        "agent_start",
        f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
    )

    # ── Step loop ─────────────────────────────────────────────
    for step in range(MAX_STEPS):
        ranked_chunks = reorder_by_question(question, context_chunks)
        top_chunks    = ranked_chunks[:7]

        print("\n========== DEBUG: TOP CHUNKS ==========")
        for i, chunk in enumerate(top_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:300].replace("\n", " "))
        print("======================================\n")

        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

        print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
        print(context[:200])
        print("=========================================\n")
        print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
        raw = call_llama(
            REACT_PROMPT.format(
                memory_context=memory_context,
                question=question,
                context=context[:2500],
                scratchpad=scratchpad if scratchpad else "None yet"
            ),
                    temperature=0.0
        )
        llm_calls += 1
        print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

        # ── Parse LLM output ──────────────────────────────────
        action       = ""
        action_input = ""
        lines        = raw.split("\n")

        thought_text = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Thought:"):
                thought_text = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action_raw   = line.replace("Action:", "").strip()
                action_lower = action_raw.lower()
                if "final" in action_lower or "answer" in action_lower:
                    action = "final_answer"
                    # capture inline answer e.g. "final_answer: Yes"
                    if ":" in action_raw:
                        inline = action_raw.split(":", 1)[1].strip()
                        if inline:
                            action_input = inline
                elif "search" in action_lower or "more" in action_lower:
                    action = "search_more"
                else:
                    action       = "final_answer"
                    action_input = action_raw
                    print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
            elif line.startswith("Input:"):
                if not action_input:
                    action_input = line.replace("Input:", "").strip()
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        if next_line.startswith(("Thought:", "Action:", "Input:")):
                            break
                        action_input += " " + next_line

        if action_input:
            action_input = clean_artifacts(action_input)
        if not action:
            action = "final_answer"
            action_input = ""
        # 🔥 FIX: recover answer if parsing failed
        action_input = clean_artifacts(action_input) if action_input else ""

        # 🔥 SAFE fallback using system signals (NO HARDCODING)
        if not action_input or len(action_input.strip()) < 2:

            if thought_text:

                if query_type == "VERIFICATION_QA":
                    confidence = round(recall_score / 100, 3)
                    print("[ReAct] ⚠️ Converting Thought → Yes/No using signals")

                    # use retrieval + grounding signals (not keywords)
                    if recall_score >= 50:
                        action_input = "Yes"
                    else:
                        action_input = "No"

                else:
                    # factual / descriptive → safe to keep
                    action_input = thought_text
        if "final_answer" in action:

            print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
            answer = action_input.strip() if action_input else ""
            print(f"[DEBUG] answer after strip: repr={repr(answer)}")

            # 🔥 FIXED fallback
            
            if answer.strip().lower() in [
    "this information is not present in the document.",
    "not present",
    "not found"
]:
                if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
                    print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
                    ranked = reorder_by_question(question, context_chunks)
                    ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
                    answer = call_llama(
                        QA_PROMPT.format(
                            memory_context=memory_context,
                            context=ctx[:2500],
                            question=question
                        ),
                        temperature=0.0
                    )
                    llm_calls += 1
            # cleaning
            if not answer:
                answer = "This information is not present in the document."
            elif answer.lower() == question.strip().lower():
                answer = "This information is not present in the document."
            else:
                cleaned = clean_reasoning_answer(answer, question)
                if cleaned:
                    answer = cleaned

            print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
            # ✅ MUST ADD THIS
            grounding = compute_answer_grounding(answer, context_chunks, question)

            return (
                answer,
                "llama_react",
                step + 1,
                retrieval_score,
                context_precision,
                grounding,
                llm_calls,
                context_chunks
            )
           

        elif "search" in action:
            print("[ReAct] ❌ Skipping search → NOT FOUND")

            return (
                "This information is not present in the document.",
                "llama_react_fail",
                step + 1,
                retrieval_score,
                context_precision,
                0.0,
                llm_calls,
                context_chunks
            )

        else:
            if len(raw) > 20:
                grounding = compute_answer_grounding(raw, context_chunks)
                return (
                    raw, "llama_react_direct", step + 1,
                    retrieval_score, context_precision,
                    grounding, llm_calls, context_chunks
                )

    # ── Max steps — final direct answer ───────────────────────
    emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
    ranked  = reorder_by_question(question, context_chunks)
    ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

    emit_event(request_id, "stream_start", "✍️ Generating final answer...")
    final, ttft = call_llama_streaming(
        QA_PROMPT.format(
            memory_context=memory_context,
            context=ctx[:2500],
            question=question
        ),
        request_id=request_id,
        temperature=0.0
    )
    llm_calls += 1
    grounding = compute_answer_grounding(final, context_chunks)

    return (
        final, "llama_react_final", MAX_STEPS,
        retrieval_score, context_precision,
        grounding, llm_calls, context_chunks
    )

































# #need changes for memory one beacsue  not working -2
# import re
# from dco_mind.config.settings import qa_pipeline, FACTUAL_TOP_K

# from dco_mind.models.llm import call_llama, call_llama_streaming
# from dco_mind.generation.response_generator import REACT_PROMPT, QA_PROMPT

# from dco_mind.retrieval.reranker import rerank_docs, protect_exact_matches
# from dco_mind.retrieval.adaptive_search import multi_query_retrieve

# from dco_mind.events.events import emit_event

# from dco_mind.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )

# from dco_mind.reasoning.context_builder import (
#     reorder_by_question, extract_numeric_answer
# )

# from dco_mind.evaluation.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )

# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     """
#     Remove answers that are pure echoes of the question.
#     Generic — no hardcoded trigger words.
#     """
#     words = answer.strip().split()

#     # Never wipe short answers — they may be legitimate facts
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
#         return ""

#     return answer


# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
#     grounding = 0.0
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     # ── Retrieval ─────────────────────────────────────────────
#     context_chunks = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     retrieval_score = compute_retrieval_score(question, context_chunks)
#     context_precision = compute_context_precision(question, context_chunks)
   

#     print(
#         f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
#         f"retrieval={retrieval_score:.1f}%"
#     )

#     emit_event(
#         request_id,
#         "agent_start",
#         f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
#     )

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")
#         print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         thought_text = ""
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought_text = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                     # capture inline answer e.g. "final_answer: Yes"
#                     if ":" in action_raw:
#                         inline = action_raw.split(":", 1)[1].strip()
#                         if inline:
#                             action_input = inline
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)
#         if not action:
#             action = "final_answer"
#             action_input = ""
#         # 🔥 FIX: recover answer if parsing failed
#         action_input = clean_artifacts(action_input) if action_input else ""

#         # 🔥 SAFE fallback using system signals (NO HARDCODING)
#         if not action_input or len(action_input.strip()) < 2:

#             if thought_text:

#                 if query_type == "VERIFICATION_QA":
#                     confidence = round(recall_score / 100, 3)
#                     print("[ReAct] ⚠️ Converting Thought → Yes/No using signals")

#                     # use retrieval + grounding signals (not keywords)
#                     if recall_score >= 50:
#                         action_input = "Yes"
#                     else:
#                         action_input = "No"

#                 else:
#                     # factual / descriptive → safe to keep
#                     action_input = thought_text
#         if "final_answer" in action:

#             print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
#             answer = action_input.strip() if action_input else ""
#             print(f"[DEBUG] answer after strip: repr={repr(answer)}")

#             # 🔥 FIXED fallback
            
#             if answer.strip().lower() in [
#     "this information is not present in the document.",
#     "not present",
#     "not found"
# ]:
#                 if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
#                     print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
#                     answer = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#             # cleaning
#             if not answer:
#                 answer = "This information is not present in the document."
#             elif answer.lower() == question.strip().lower():
#                 answer = "This information is not present in the document."
#             else:
#                 cleaned = clean_reasoning_answer(answer, question)
#                 if cleaned:
#                     answer = cleaned

#             print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
#             # ✅ MUST ADD THIS
#             grounding = compute_answer_grounding(answer, context_chunks, question)

#             return (
#                 answer,
#                 "llama_react",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 grounding,
#                 llm_calls,
#                 context_chunks
#             )
           

#         elif "search" in action:
#             print("[ReAct] ❌ Skipping search → NOT FOUND")

#             return (
#                 "This information is not present in the document.",
#                 "llama_react_fail",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 0.0,
#                 llm_calls,
#                 context_chunks
#             )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks
#                 )

#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )








































#original but got that  NLP issue where crct retieval but agent ignoring as it is expecting Yes or no keywords only
#  import re
# from dco_mind.config.settings import qa_pipeline, FACTUAL_TOP_K

# from dco_mind.models.llm import call_llama, call_llama_streaming
# from dco_mind.generation.response_generator import REACT_PROMPT, QA_PROMPT

# from dco_mind.retrieval.reranker import rerank_docs, protect_exact_matches
# from dco_mind.retrieval.adaptive_search import multi_query_retrieve

# from dco_mind.events.events import emit_event

# from dco_mind.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )

# from dco_mind.reasoning.context_builder import (
#     reorder_by_question, extract_numeric_answer
# )

# from dco_mind.evaluation.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )

# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     """
#     Remove answers that are pure echoes of the question.
#     Generic — no hardcoded trigger words.
#     """
#     words = answer.strip().split()

#     # Never wipe short answers — they may be legitimate facts
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
#         return ""

#     return answer


# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
#     grounding = 0.0
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     # ── Retrieval ─────────────────────────────────────────────
#     context_chunks = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     retrieval_score = compute_retrieval_score(question, context_chunks)
#     context_precision = compute_context_precision(question, context_chunks)
   

#     print(
#         f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
#         f"retrieval={retrieval_score:.1f}%"
#     )

#     emit_event(
#         request_id,
#         "agent_start",
#         f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
#     )

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")
#         print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                     # capture inline answer e.g. "final_answer: Yes"
#                     if ":" in action_raw:
#                         inline = action_raw.split(":", 1)[1].strip()
#                         if inline:
#                             action_input = inline
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)
#         if not action:
#             action = "final_answer"
#             action_input = ""
#         # 🔥 FIX: recover answer if parsing failed
#         action_input = clean_artifacts(action_input) if action_input else ""

#         if not action_input or len(action_input.strip()) < 2:
#             match = re.search(r'final_answer\s*:\s*(.+)', raw, re.IGNORECASE)
#             if match:
#                 recovered = match.group(1).strip()
#                 if recovered:
#                     print("[ReAct] ⚠️ Recovering answer from raw output")
#                     action_input = recovered
       
#         if "final_answer" in action:

#             print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
#             answer = action_input.strip() if action_input else ""
#             print(f"[DEBUG] answer after strip: repr={repr(answer)}")

#             # 🔥 FIXED fallback
            
#             if answer.strip().lower() in [
#     "this information is not present in the document.",
#     "not present",
#     "not found"
# ]:
#                 if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
#                     print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
#                     answer = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#             # cleaning
#             if not answer:
#                 answer = "This information is not present in the document."
#             elif answer.lower() == question.strip().lower():
#                 answer = "This information is not present in the document."
#             else:
#                 cleaned = clean_reasoning_answer(answer, question)
#                 if cleaned:
#                     answer = cleaned

#             print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
#             # ✅ MUST ADD THIS
#             grounding = compute_answer_grounding(answer, context_chunks, question)

#             return (
#                 answer,
#                 "llama_react",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 grounding,
#                 llm_calls,
#                 context_chunks
#             )
           

#         elif "search" in action:
#             print("[ReAct] ❌ Skipping search → NOT FOUND")

#             return (
#                 "This information is not present in the document.",
#                 "llama_react_fail",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 0.0,
#                 llm_calls,
#                 context_chunks
#             )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks
#                 )

#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )


