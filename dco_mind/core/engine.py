
import time
import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

from dco_mind.config.settings import (
    MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K,DIRECT_QA_TOP_K
)

from dco_mind.reasoning.context_builder import extract_named_entities, expand_answer
from dco_mind.core.state import DocState
from dco_mind.generation.response_generator import QA_PROMPT

from dco_mind.models.llm import call_llama_streaming
from dco_mind.models.embeddings import build_faiss_index
from dco_mind.retrieval.reranker import rerank_docs, protect_exact_matches

from dco_mind.knowledge.ingestion import (
    extract_pdf_parallel, _extraction_cache,
    semantic_chunk, raptor_summarize, _summary_cache
)

from dco_mind.retrieval.adaptive_search import multi_query_retrieve

from dco_mind.cognition.query_brain import (
    react_agent, roberta_qa, clean_reasoning_answer
)

from dco_mind.events.events import emit_event

from dco_mind.utils.helpers import (
    get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
)

from dco_mind.reasoning.context_builder import (
    classify_from_context,
    reorder_by_question,
    normalize_text,
)

from dco_mind.evaluation.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
    compute_recall_at_k,
    compute_confidence,
    compute_grounding_score,
    semantic_similarity,
)
# from dco_mind.cognition.memory import get_memory_context,get_retrieval_query, get_best_memory_match,get_previous_retrieved_chunks,save_to_memory
from dco_mind.cognition.memory import get_memory_context, get_best_memory_match, save_to_memory,get_history

# ============================================================
# LOCAL HELPERS
# ============================================================

def clean_context_for_llm(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        lines = chunk.split("\n")
        filtered_lines = []
        for line in lines:
            line_strip = line.strip()
            is_question_like = (
                line_strip.endswith("?") or
                (len(line_strip.split()) < 15 and "?" in line_strip)
            )
            if is_question_like:
                continue
            filtered_lines.append(line)
        cleaned_chunks.append("\n".join(filtered_lines))
    return cleaned_chunks


def is_grounded(answer, retrieved_texts, threshold=0.6):
    if not answer:
        return False
    answer_words = set(answer.lower().split())
    if not answer_words:
        return False
    best_overlap = 0
    for chunk in retrieved_texts:
        chunk_words = set(chunk.lower().split())
        overlap = len(answer_words & chunk_words) / len(answer_words)
        best_overlap = max(best_overlap, overlap)
    return best_overlap >= threshold


# ============================================================
# NUMERIC INTENT DETECTION
# ============================================================

def _extract_numbers(text: str) -> list:
    return re.findall(r'\b\d+\b', text)


def _detect_numeric_intent(question: str, chunks: list) -> str:
    """
    Detect NAVIGATIONAL or POSITIONAL from chunk structure.
    Generic — no hardcoded section words.
    """
    numbers = _extract_numbers(question)
    if not numbers:
        return "NONE"

    q_words = set(question.lower().split())

    for chunk in chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue
        # chunks may have no newlines — take first 10 tokens as heading proxy
        first_line = " ".join(lines[0].split()[:10]).lower()
        line_words = set(first_line.split())
        overlap    = len(q_words & line_words)
        num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

        if num_found:
            print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

        if num_found and overlap >= 2:  # ← remove word count check entirely
            return "NAVIGATIONAL"
    for chunk in chunks:
        lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l.split()) <= 12]
        if len(short_lines) >= 3:
            first = lines[0].lower() if lines else ""
            if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
                return "POSITIONAL"

    return "NONE"


def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
    numbers = _extract_numbers(question)
    if not numbers:
        return ""

    q_words    = set(question.lower().split())
    candidates = []

    for chunk in all_raw_chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue

        first_line  = " ".join(lines[0].split()[:10])
        first_lower = first_line.lower()

        for num in numbers:
            if not re.search(rf'\b{re.escape(num)}\b', first_lower):
                continue

            line_words = set(first_lower.split())
            overlap    = len(q_words & line_words)

            if overlap >= 2:
                # generic scoring (NO hardcoding)
                has_number = num in first_lower
                score = overlap + (2 if has_number else 0)

                candidates.append((first_line, score))

    if not candidates:
        return ""

    # pick BEST candidate (not first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    best_line, best_score = candidates[0]

    # 🔥 RELATIVE CONFIDENCE (NO HARDCODING)
    if len(candidates) > 1:
        second_score = candidates[1][1]
    else:
        second_score = 0

    # Reject if not clearly better
    if best_score <= second_score:
        print("[NAV] ❌ No clear winner → fallback")
        return ""

    # Optional: also reject very weak absolute matches
    if best_score < 3:
        print("[NAV] ⚠️ Weak match → fallback")
        return ""
    return best_line

def _positional_extract(question: str, retrieved_texts: list) -> str:
    """Extract Nth item from a list. Generic."""
    numbers = _extract_numbers(question)
    if not numbers:
        return ""
    try:
        idx = int(numbers[0]) - 1
    except ValueError:
        return ""
    if idx < 0:
        return ""

    best_chunk, best_count = "", 0
    for chunk in retrieved_texts:
        lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l.split()) <= 12]
        if len(short_lines) > best_count:
            best_count = len(short_lines)
            best_chunk = chunk

    if not best_chunk:
        return ""

    lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
    short_lines = [l for l in lines if len(l.split()) <= 12]
    if idx < len(short_lines):
        return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
    return ""


# ============================================================
# SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# ============================================================

def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
    """
    If answer contains a multi-word identifier (word+number or number+word),
    verify it appears verbatim in retrieved chunks.
    Generic — no hardcoded section words.
    """
    identifiers = re.findall(
        r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
        answer.lower()
    )
    if not identifiers:
        return True

    chunk_text = " ".join(retrieved_texts).lower()
    for ident in identifiers:
        if ident not in chunk_text:
            print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
            return False
    return True


# ============================================================
# REFUSAL — semantic version (local)
# ============================================================

_REFUSAL_ANCHOR = "this information is not available in the provided context"


def _is_refusal_semantic(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return True
    sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
    print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
    return sim > 0.72


# ============================================================
# LANGGRAPH NODES
# ============================================================

def node_extract(state: DocState) -> DocState:
    extract_start    = time.time()
    text, page_count = extract_pdf_parallel(state["pdf_path"])
    extract_time     = time.time() - extract_start
    state["extracted_text"] = text
    state["page_count"]     = page_count
    state["char_count"]     = len(text)
    state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
    state["metrics"]["pages_processed"]      = page_count
    state["metrics"]["characters_processed"] = len(text)
    state["metrics"]["words_processed"]      = len(text.split())
    return state


def node_chunk(state: DocState) -> DocState:
    text = state["extracted_text"]
    state["query_type"] = "QA"
    question = state.get("question", "").strip().lower()
    summary_pattern = (
    r'^(summarize|summarise|summary|'
    r'give me a summary|'
    r'provide a summary|'
    r'summarise this|'
    r'generate a summary)\b'
)
    if re.match(summary_pattern, question) or (len(question.split()) > 12 and "?" not in question):
        state["query_type"] = "FULL_SUMMARY"

    summary_chunks, rag_chunks = semantic_chunk(text)

    # FIX 4 — normalize all chunks after chunking
    # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
    rag_chunks     = [normalize_text(c) for c in rag_chunks]
    summary_chunks = [normalize_text(c) for c in summary_chunks]

    state["summary_chunks"] = summary_chunks
    state["chunks"]         = rag_chunks

    state["metrics"]["summary_chunks"] = len(summary_chunks)
    state["metrics"]["chunks_created"] = len(rag_chunks)
    # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
    state["metrics"]["query_type"]     = state["query_type"]
    print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
    return state




def node_summarize(state: DocState) -> DocState:
    pdf_hash  = get_pdf_hash(state["pdf_path"])
    cache_key = pdf_hash  # ❌ removed doc_type dependency

    # ── Cache check ──────────────────────────────────────────
    if cache_key in _summary_cache:
        cached = _summary_cache[cache_key]
        print("[Summary] ✅ Cache hit")
        emit_event(
            state.get("request_id", ""),
            "agent_action",
            "⚡ Summary loaded from cache instantly!"
        )
        state["answer"] = cached["summary"]
        state["metrics"].update(cached["metrics"])
        state["metrics"]["type"] = "summary"
        return state

    # ── Generate summary ─────────────────────────────────────
    summary_start = time.time()
    raptor_summarize._request_id = state.get("request_id", "")

    # ❌ removed doc_type argument
    summary, map_time, reduce_time = raptor_summarize(
        state["summary_chunks"],   # ← was state["chunks"], must be summary_chunks
        state.get("doc_type", "general")
    )
    summary_time = time.time() - summary_start

    # ── Store result ─────────────────────────────────────────
    state["answer"] = summary

    metrics_snapshot = {
        "summary_time_sec":     round(summary_time, 2),
        "summary_length_words": len(summary.split()),
        "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
        "map_time_sec":         round(map_time, 2),
        "reduce_time_sec":      round(reduce_time, 2),
        "llm_calls":            3,
    }

    state["metrics"].update(metrics_snapshot)
    state["metrics"]["type"] = "summary"

    # ── Cache result ─────────────────────────────────────────
    _summary_cache[cache_key] = {
        "summary": summary,
        "metrics": metrics_snapshot
    }

    print(f"[Summary] Done ({len(summary.split())} words)")
    return state

def node_qa(state: DocState) -> DocState:
    print("\n[DEBUG STATE]")
    print(f"original_question = {repr(state.get('original_question'))}")
    print(f"session_id = {repr(state.get('session_id'))}")
    qa_start_t = time.time()
    question   = state["question"]
    request_id = state.get("request_id", "")
    all_chunks = state["chunks"]
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = "default_session"
    original_q = state.get("original_question", question)

    question = normalize_text(question)
    numeric_intent = "NONE"

    all_raw = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    pdf_hash    = get_pdf_hash(state["pdf_path"])
    faiss_index = build_faiss_index(all_chunks, pdf_hash)

    print(f"[QA] START | {len(all_chunks)} chunks")

    recall_score      = 0.0
    retrieval_score   = 0.0
    context_precision = 0.0
    grounding         = 0.0
    grounding_score = 0.0
    semantic_ground = 0.0
    grounding_now = 0.0
    force_not_found = False
    react_ans = ""
    llm_calls         = 0
    model_used        = "llama"
    confidence        = 0.0
    decision_type     = "accepted"
    retrieved         = []
    retrieved_texts   = []
    answer            = ""
    rewrite_triggered = False
    rewritten_question = ""
    pre_rewrite_docs = []
    post_rewrite_docs = []
    refusal_phrases = [
    "not present",
    "not mentioned",
    "not available",
    "no information",
    "cannot find",
    "context does not mention",
    "don't have any information",
    "not in the document",
    "cannot answer",
    "no mention",
    "there is no mention",
    "does not mention",
    "cannot determine",
    "don't see any information",
    "don't see any mention",
    "no mention of",
    "i don't see",
    "there is no",
    "i don't have",
    "does not explicitly mention",
    "does not specifically",
    "not explicitly mentioned",
]

    # ── STEP 1: RETRIEVE — clean question only ────────────────
    retrieved = multi_query_retrieve(
        question, faiss_index,
        k=50,
        all_chunks=all_chunks,
        query_type="FACTUAL_QA"
    )
    initial_retrieved = retrieved.copy()
    pre_rewrite_docs = [
    {
        "content": d.page_content if hasattr(d, "page_content") else str(d),
        "metadata": getattr(d, "metadata", {})
    }
    for d in initial_retrieved
]

    # 🔥 STEP 2 — RERANK + SAFE FALLBACK
    retrieved, reranker_top, _ = rerank_docs(
        question, retrieved, top_k=8, apply_pruning=True
    )

    retrieved = protect_exact_matches(
        question, retrieved, all_chunks, top_k=8
    )
    retrieved = retrieved[:6]

    if not retrieved or len(retrieved) < 3:
        print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
        retrieved = initial_retrieved[:8]

    retrieved_texts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in retrieved
    ]
    if reranker_top < -5:
        print("[Guard] ⚠️ Weak reranker score — continuing")

    # ── STEP 3: METRICS ───────────────────────────────────────
    retrieval_score   = compute_retrieval_score(question, retrieved)
    context_precision = compute_context_precision(question, retrieved)
    recall_score      = compute_recall_at_k(
        question, retrieved, all_chunks, k=len(retrieved)
    )

    is_numeric_question = len(_extract_numbers(question)) > 0
    if recall_score < 40:
        print("[QA] ⚠️ Expanding retrieval (low recall)")
        expanded = multi_query_retrieve(
            question, faiss_index, k=30,
            all_chunks=all_chunks, query_type="FACTUAL_QA"
        )
        if len(expanded) > len(retrieved):
            retrieved = expanded
            retrieved_texts = [
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in retrieved
            ]
        numeric_intent = _detect_numeric_intent(question, retrieved_texts)
        is_numeric_question = len(_extract_numbers(question)) > 0
        if numeric_intent == "NONE" and is_numeric_question:
            print("[Navigate] Retrying intent detection on full document...")
            numeric_intent = _detect_numeric_intent(question, all_raw)
    else:
        print("[QA] ✅ High recall — trusting retrieved context")

    if numeric_intent == "NAVIGATIONAL":
        print("[Routing] Numeric intent → NAVIGATIONAL")
        title = _navigate_full_chunks(question, all_raw)
        if title:
            print(f"[Navigate] Extracted title: '{title}'")
            grounding  = compute_answer_grounding(title, retrieved_texts, question)
            try:
                confidence = compute_confidence(
                    reranker_top=reranker_top,
                    recall_score=recall_score,
                    answer=answer,
                    context_chunks=retrieved_texts,
                    question=question
                )
            except Exception as e:
                print("\n[CONFIDENCE CRASH]")
                print(f"answer={repr(answer)}")
                print(f"question={repr(question)}")
                print(f"retrieved_texts type={type(retrieved_texts)}")
                print(f"retrieved_texts len={len(retrieved_texts) if retrieved_texts else 0}")
                print(f"ERROR={e}")
                raise
            qa_time    = time.time() - qa_start_t
            state["answer"]     = title
            state["query_type"] = "NAVIGATIONAL"
            _write_metrics(state, "navigational", "navigational",
                           grounding, confidence, retrieval_score,
                           context_precision, recall_score, llm_calls,
                           retrieved, qa_time)
            return state
        print("[Navigate] Falling back to QA")

    # ── Normal routing ────────────────────────────────────────
    query_type = classify_from_context(original_q, retrieved_texts)
    state["query_type"] = query_type
    print(f"[Routing] Context-based → {query_type}")

    # ── STEP 4: ANSWER GENERATION ─────────────────────────────
    if query_type == "FULL_SUMMARY":
        llm_context_chunks = clean_context_for_llm(retrieved_texts)
        ranked = retrieved_texts
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        structured_context = "\n".join(
            f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
        )
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(
                memory_context="",
                context=structured_context[:2500],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        if answer is None:                          
            answer = ""                             
        answer     = clean_artifacts(str(answer)).strip()  
   
        model_used = "llama_summary"
        llm_calls  = 1
        grounding_score = compute_answer_grounding(answer, retrieved_texts, question)
        semantic_ground = semantic_similarity(answer, retrieved_texts)

    elif query_type == "MULTIPART_QA":
    
        llm_context_chunks = clean_context_for_llm(retrieved_texts)

        ranked = reorder_by_question(
            question,
            llm_context_chunks
        )
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        emit_event(request_id, "agent_start", f"🤖 MULTIPART | {len(retrieved)} chunks")
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        react_ans, _ = call_llama_streaming(
            QA_PROMPT.format(
                memory_context="",
                context=context[:4000],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        if react_ans is None:                       
            react_ans = ""                          
        react_ans = clean_artifacts(str(react_ans)).strip()  
      
        answer = react_ans

        model_used = "llama_multipart"
        llm_calls  = 1
        grounding_score = compute_answer_grounding(answer, retrieved_texts, question)
        semantic_ground = semantic_similarity(answer, retrieved_texts)

    else:
        if not retrieved_texts:
            print("[QA] ❌ No context → NOT FOUND")
            state["answer"] = "This information is not present in the document."
            _write_metrics(state, "not_found", "no_context",
                           0.0, 0.0, retrieval_score, context_precision,
                           recall_score, llm_calls, retrieved,
                           time.time() - qa_start_t)
            return state

        ranked = reorder_by_question(
            question,
            retrieved_texts
        )

        if query_type == "MULTIPART_QA":
            retrieved_texts = ranked[:6]
        else:
            retrieved_texts = ranked[:4]

        context = "\n\n---\n\n".join(retrieved_texts)
        memory_context = get_memory_context(session_id)

        if (
            query_type in ("FACTUAL_QA", "VERIFICATION_QA")
            and recall_score >= 60
        ):
            print("[QA] ⚡ Direct QA (no ReAct)")
        

            if query_type == "FACTUAL_QA":


                prompt = f"""Previous Conversation:
{memory_context}
Answer strictly using the provided context.
For factual fields like names, numbers, dates, marks, or values,
return the exact answer as it appears.
For conceptual questions, provide a concise grounded answer
based only on the context.
For numeric comparisons, compute the comparison explicitly.
Do not hallucinate unsupported information.
Answer using the context.

If the answer appears implicitly, infer it carefully from the retrieved text.

Only say "This information is not present in the document"
when the retrieved context contains no relevant evidence at all.
Context:
{context[:2000]}
Question:
{original_q}
Answer:
"""
            else:

                prompt = f"""Previous Conversation:
{memory_context}

Use the retrieved context as the primary source of truth.

Reason carefully over the evidence before answering.

For yes/no questions:
- infer the answer from the retrieved evidence
- answer directly when evidence strongly implies the conclusion
- briefly explain the reasoning when useful

Do not refuse if the answer can be inferred from the context.

Only say "This information is not present in the document"
when the retrieved context contains no meaningful evidence at all.

Context:
{context[:2000]}

Question:
{original_q}

Answer:
"""
                print("\n" + "="*80)
                print("FINAL CHUNKS SENT TO LLM")
                print("="*80)

                for i, chunk in enumerate(retrieved_texts):
                    print(f"\nCHUNK {i+1}\n")
                    print(chunk[:800])

                print("\nFINAL CONTEXT:\n")
                print(context[:4000])

                print("\nQUESTION:")
                print(original_q)

                print("="*80 + "\n")

            react_ans, _ = call_llama_streaming(
                prompt,
                request_id=request_id,
                temperature=0.0
            )
            llm_calls += 1
            model_used = "llama_direct"
         

            FOLLOWUP_WORDS = {
    "it", "this", "that",
    "he", "she", "they",
    "them", "his", "her", "their"
}
            tokens = re.findall(r"\b\w+\b", question.lower())

            # history = get_history(session_id)
            history = get_history(session_id)

            previous_question = ""

            if history:
                previous_question = history[-1].get("question", "")

            has_memory = len(history) > 0

            FOLLOWUP_PRONOUNS = {
                "it", "this", "that",
                "he", "she", "they",
                "them", "his", "her",
                "their", "its",
                "these", "those"
            }

            pronoun_count = sum(
                1 for t in tokens
                if t in FOLLOWUP_PRONOUNS
            )

            meaningful_tokens = [
                t for t in tokens
                if (
                    t not in FOLLOWUP_PRONOUNS
                    and len(t) > 2
                )
            ]

            semantic_density = len(meaningful_tokens)

            is_followup = (
                has_memory
                and (
                    pronoun_count > 0
                    or semantic_density <= 2
                )
            )

            is_refusal = any(
                p in react_ans.lower()
                for p in refusal_phrases
            )

            best_memory = get_best_memory_match(
                session_id,
                question
            )

            grounding_now = (
    compute_grounding_score(
        react_ans,
        retrieved_texts
    ) * 100
)



            ambiguous_followup = (
                is_followup
                and confidence > 0.50
                and grounding_now < 85
            )
            memory_trigger = (
    has_memory
    and (
        is_refusal
        or recall_score < 60
        or ambiguous_followup
    )
)

      
            print("\n[MEMORY DEBUG]")
            print("history_len =", len(history))
            print("has_memory =", has_memory)
            print("is_followup =", is_followup)
            print("grounding_now =", grounding_now)
            print("confidence =", confidence)
            print("ambiguous_followup =", ambiguous_followup)
            print("memory_trigger =", memory_trigger)
            if is_followup and (is_refusal or memory_trigger):

                print("[MemoryRewrite] Triggered")
                rewrite_triggered = True

                history = get_memory_context(session_id)
                print("\n[MEMORY HISTORY]")
                print(history)
                print("[END MEMORY HISTORY]\n")


                rewrite_prompt = f"""
                Previous Question:
                {previous_question}

                Current Question:
                {question}

                Rewrite the current question into a standalone question.

                Rules:
                - Use ONLY the previous question for resolving references.
                - Do NOT use hidden assumptions.
                - Do NOT introduce new entities.
                - Preserve the exact meaning.
                - Keep the rewrite concise.
                - If the current question is already standalone, return it unchanged.

                Only return the rewritten question.
                """

                rewritten_question, _ = call_llama_streaming(
                    rewrite_prompt,
                    request_id=request_id,
                    temperature=0.0
                )

                rewritten_question = clean_artifacts(
                    rewritten_question
                ).strip()

                print(f"[MemoryRewrite] {rewritten_question}")
          
                # ------------------------------------------------
                # RE-RETRIEVE using rewritten standalone question
                # ------------------------------------------------

                new_retrieved = multi_query_retrieve(
                    rewritten_question,
                    faiss_index,
                    k=50,
                    all_chunks=all_chunks,
                    query_type="FACTUAL_QA"
                )

                new_retrieved, _, _ = rerank_docs(
                    rewritten_question,
                    new_retrieved,
                    top_k=8,
                    apply_pruning=True
                )

                new_retrieved = protect_exact_matches(
                    rewritten_question,
                    new_retrieved,
                    all_chunks,
                    top_k=8
                )

                new_retrieved_texts = [
                    d.page_content if hasattr(d, "page_content") else str(d)
                    for d in new_retrieved
                ]

                new_context = "\n".join(new_retrieved_texts)

                retry_prompt = f"""
                Previous Conversation:
                {history}

                Use the retrieved context as the primary source of truth.
                Use conversation history only when relevant to resolve follow-up references.

                Context:
                {new_context[:2000]}

                Question:
                {rewritten_question}

                Answer:
                """

                react_ans, _ = call_llama_streaming(
                    retry_prompt,
                    request_id=request_id,
                    temperature=0.0
                )

                # IMPORTANT:
                retrieved = new_retrieved
                retrieved_texts = new_retrieved_texts
                post_rewrite_docs = [
                    {
                        "content": d.page_content if hasattr(d, "page_content") else str(d),
                        "metadata": getattr(d, "metadata", {})
                    }
                    for d in new_retrieved
                    ]

                llm_calls += 2
        else:
            react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
                original_q,
                faiss_index,
                query_type,
                all_chunks,
                request_id,
                recall_score,
            
                memory_context=memory_context
            )
        print("\n[DEBUG RAW LLM OUTPUT]")
        print(f"type={type(react_ans)}")
        print(f"value={repr(react_ans)}")

   
        if react_ans is None:
            print("[ERROR] react_ans is None")
            react_ans = ""

        react_ans = clean_artifacts(
            str(react_ans)
        ).strip().strip('"').strip("'")
        print(f"[QA] Answer: '{react_ans[:60]}'")

        words         = react_ans.split()
        content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

        print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")


        is_refusal_detected = any(
            p in react_ans.lower()
            for p in refusal_phrases
        )

        weak_retrieval = (
            recall_score < 85
            and grounding_now < 35
        )

        if is_refusal_detected and weak_retrieval:

            print("[QA] ⚠️ Refusal detected")

            force_not_found = True

            answer = "This information is not present in the document."
            react_ans = answer
            state["answer"] = answer

            decision_type = "not_found"

            grounding = 0.0
            confidence = 0.0

        grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
        semantic_ground = semantic_similarity(
    react_ans,
    retrieved_texts
)
        print(
    f"[Grounding Fusion] "
    f"lexical={grounding_score:.3f} "
    f"semantic={semantic_ground:.3f} "
    f"recall={recall_score:.1f}"
)
      
        if force_not_found:

            answer = "This information is not present in the document."
            decision_type = "not_found"

        elif not react_ans or react_ans.strip() == "":
            answer        = "This information is not present in the document."
            decision_type = "empty_answer"

        else:
            if query_type == "VERIFICATION_QA":
                normalized = react_ans.strip().lower()

                if recall_score >= 40:
                    answer        = react_ans
                    decision_type = "accepted"
                else:
                    answer        = "This information is not present in the document."
                    decision_type = "low_recall"

       
            elif query_type == "FACTUAL_QA":

                allow_memory_override = (
                    rewrite_triggered
                    and recall_score >= 80
                    and confidence >= 0.45
                )

                grounded_enough = (
                    grounding_score >= 45
                    or semantic_ground >= 0.45
                    or (
                        recall_score >= 80
                        and semantic_ground >= 0.30
                    )
                    or allow_memory_override
                )

                if grounded_enough:
                    answer = react_ans
                    decision_type = "accepted"

                # FIX 1:
                # Do NOT overwrite correct answers when recall is strong.
                elif recall_score >= 85 and len(react_ans.strip()) > 15:
                    print("[Guard] High recall answer preserved")
                    answer = react_ans
                    decision_type = "accepted_high_recall"

                else:
                    answer = "This information is not present in the document."
                    decision_type = "low_grounding"

            else:
                ans_words = len(react_ans.strip().split())
                if recall_score < 25:
                    answer        = "This information is not present in the document."
                    decision_type = "low_recall"
                elif ans_words <= 3:
                    if recall_score >= 40:
                        answer        = react_ans
                        decision_type = "accepted"
                    else:
                        answer        = "This information is not present in the document."
                        decision_type = "weak_short"
                elif grounding_score < 40 and recall_score < 40:
                    answer        = "This information is not present in the document."
                    decision_type = "low_grounding"
                elif grounding_score >= 50 or recall_score >= 50:
                    answer        = react_ans
                    decision_type = "accepted"
                else:
                    answer        = "This information is not present in the document."
                    decision_type = "uncertain"

    if query_type == "VERIFICATION_QA":
        grounding = recall_score
    else:
        grounding = max(grounding_score, semantic_ground * 100)

    confidence = compute_confidence(
    reranker_top=reranker_top,
    recall_score=recall_score,
    answer=answer,
    context_chunks=retrieved_texts,
    question=question
)
    qa_time    = time.time() - qa_start_t

    print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
          f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
          f"confidence={confidence:.3f} | decision={decision_type}")

    if not answer.strip():
        answer = "Could not find a relevant answer in the PDF."
    else:
        answer = normalize_answer(answer)


    state["answer"] = answer
    state["rewrite_triggered"] = rewrite_triggered
    state["rewritten_query"] = rewritten_question
    state["pre_rewrite_docs"] = pre_rewrite_docs
    state["post_rewrite_docs"] = post_rewrite_docs
    _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time)

    if (
        not force_not_found
        and decision_type.startswith("accepted")
        and (
            grounding >= 50
            or recall_score >= 90
        )
    ):
        print(f"[DEBUG] SAVE session_id = {session_id}")
        save_to_memory(
            session_id=session_id,
            question=original_q,
            answer=answer,
            query_type=query_type,
            grounding=grounding,
            confidence=confidence,
            retrieved_chunks=retrieved_texts[:8]
        )
    else:

        # Save conversational anchor only
        # Do NOT save hallucinated/refused answers

        save_to_memory(
        session_id=session_id,
        question=original_q,
        answer="",
        query_type=query_type,
        grounding=grounding,
        confidence=confidence,
        retrieved_chunks=retrieved_texts[:8]
)

        print("[Memory] Saved question-only context")
        print("\n[DEBUG] Saving retrieved chunks:\n")

        for i, ch in enumerate(retrieved_texts[:8]):
            print(f"\n--- SAVED CHUNK {i} ---\n")
            print(ch[:300])

        print("[Memory] ✅ Saved trusted answer")
    return state

def node_validate(state: DocState) -> DocState:
    answer = state["answer"]
    retry  = state.get("retry_count", 0)

    # ── Retry for empty/very weak answers ────────────────────
    if len(answer.strip()) < 3 and retry < 2:
        state["retry_count"] = retry + 1
        state["answer"]      = ""
        return state

    total_time    = time.time() - state["start_time"]
    output_words  = len(answer.split())
    output_tokens = output_words * 1.3

    extract_time  = state["metrics"].get("extraction_time_sec", 0)
    llm_time      = max(total_time - extract_time, 1)
    tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

    m = state["metrics"]

    # ── Core metrics ─────────────────────────────────────────
    m["response_time_sec"]    = round(total_time, 2)
    m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
    m["pages_processed"]      = state.get("page_count", 0)
    m["characters_processed"] = state.get("char_count", 0)
    m["words_processed"]      = len(state.get("extracted_text", "").split())

    # ── Type-specific metrics ────────────────────────────────
    if m.get("type") == "summary":
        m["summary_time_sec"]     = m.get("summary_time_sec", 0)
        m["summary_length_words"] = len(answer.split())

    if m.get("type") == "qa":
        m["qa_time_sec"]      = m.get("qa_time_sec", 0)
        m["confidence_score"] = m.get("confidence_score", 0)

    # ── Performance ──────────────────────────────────────────
    m["ttft_sec"]        = round(total_time, 2)
    m["e2e_latency_sec"] = round(total_time, 2)
    m["tps"]             = tps

   

    # ── Context info ─────────────────────────────────────────
    m["query_type"]     = state.get("query_type", "")
    m["chunks_created"] = m.get("chunks_created", 0)
    m["retry_count"]    = retry

    # ── Model + retrieval metrics ────────────────────────────
    m["model_used"]        = m.get("model_used", "llama_react")
    m["llm_calls"]         = m.get("llm_calls", 0)
    m["retrieval_score"]   = m.get("retrieval_score", 0)
    m["context_precision"] = m.get("context_precision", 0)
    m["answer_grounding"]  = m.get("answer_grounding", 0)
    m["recall_at_k"]       = m.get("recall_at_k", 0)

    # ── Summary-specific ─────────────────────────────────────
    if m.get("type") == "summary":
        m["parallel_workers"] = m.get("parallel_workers", 0)
        m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
        m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

    # ── QA-specific ──────────────────────────────────────────
    if m.get("type") == "qa":
        m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
        m["decision_type"]    = m.get("decision_type", "accepted")
        m["confidence_raw"]   = m.get("confidence_raw", 0.0)

    state["metrics"] = m
    return state




def _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time):
    m = state.setdefault("metrics", {})
    m["qa_time_sec"]       = round(qa_time, 2)
    m["confidence_score"]  = round(confidence * 100, 2)
    m["retrieval_score"]   = retrieval_score
    m["context_precision"] = context_precision
    m["answer_grounding"]  = grounding
    m["recall_at_k"]       = recall_score
    m["llm_calls"]         = llm_calls
    m["model_used"]        = model_used
    m["chunks_retrieved"]  = len(retrieved)
    m["retrieved_docs"] = [
        {
            "content": d.page_content if hasattr(d, "page_content") else str(d),
            "metadata": getattr(d, "metadata", {})
        }
        for d in retrieved
    ]
    m["type"]              = "qa"
    m["decision_type"]     = decision_type
    m["confidence_raw"]    = round(confidence, 4)
    m["rewrite_triggered"] = state.get(
    "rewrite_triggered",
    False
)

    m["rewritten_query"] = state.get(
        "rewritten_query",
        ""
    )

    m["pre_rewrite_docs"] = state.get(
        "pre_rewrite_docs",
        []
    )

    m["post_rewrite_docs"] = state.get(
        "post_rewrite_docs",
        []
    )



def run_pipeline(question: str, pdf_path: str, session_id: str):
    if not session_id:
        session_id = "default_session"
    state = {
        "question": question,
        "pdf_path": pdf_path,
        "session_id": session_id,
        "start_time": time.time(),
        "metrics": {},
    }

    # Execute pipeline manually
    state = node_extract(state)
    state = node_chunk(state)
    state = node_qa(state)
    state = node_validate(state)

    # 🔥 IMPORTANT: expose for evaluation
    state["retrieved_docs"] = state["metrics"].get("retrieved_docs", [])
    state["all_chunks"] = state.get("chunks", [])

    return state









