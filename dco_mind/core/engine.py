
import time
import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

from dco_mind.config.settings import (
    MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
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
    semantic_similarity,
)
from dco_mind.cognition.memory import get_memory_context

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
    summary_pattern = r'^(summarize|summarise|give|provide|generate|write|create)\b'
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
    qa_start_t = time.time()
    question   = state["question"]
    request_id = state.get("request_id", "")
    all_chunks = state["chunks"]

    # FIX 4 — normalize question the same way chunks were normalized
    question = normalize_text(question)
    numeric_intent = "NONE"   # 🔥 DEFAULT FIX (MANDATORY)
    # Raw strings for navigational scanning — must NOT be cleaned
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
    llm_calls         = 0
    model_used        = "llama"
    confidence        = 0.0
    decision_type     = "accepted"
    retrieved         = []
    retrieved_texts   = []
    answer            = ""

    # ── STEP 1: RETRIEVE ─────────────────────────────────────
    retrieved = multi_query_retrieve(
        question, faiss_index,
        k=50,
        all_chunks=all_chunks,
        query_type="FACTUAL_QA"
    )


   
    # 🔥 STEP 2 — RERANK + SAFE FALLBACK

    retrieved, reranker_top, _ = rerank_docs(
        question, retrieved, top_k=8, apply_pruning=True
    )

    retrieved = protect_exact_matches(
        question, retrieved, all_chunks, top_k=8
    )

    # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
    if not retrieved or len(retrieved) < 3:
        print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
        retrieved = all_chunks[:8]

    # 🔥 FIX 2 — safe conversion
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

    # ── FIX 6 — k expansion: low recall OR numeric question ───
    is_numeric_question = len(_extract_numbers(question)) > 0
    if recall_score < 25 or is_numeric_question:
        print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
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
        # 🔥 ENSURE numeric_intent ALWAYS DEFINED
        numeric_intent = _detect_numeric_intent(question, retrieved_texts)

        # fallback check (keep this)
        is_numeric_question = len(_extract_numbers(question)) > 0

        if numeric_intent == "NONE" and is_numeric_question:
            print("[Navigate] Retrying intent detection on full document...")
            numeric_intent = _detect_numeric_intent(question, all_raw)
    # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
    if numeric_intent == "NAVIGATIONAL":
        print("[Routing] Numeric intent → NAVIGATIONAL")

        title = _navigate_full_chunks(question, all_raw)

        if title:
            print(f"[Navigate] Extracted title: '{title}'")

            grounding  = compute_answer_grounding(title, retrieved_texts, question)
            confidence = round(grounding / 100, 3)
            qa_time    = time.time() - qa_start_t

            state["answer"]     = title
            state["query_type"] = "NAVIGATIONAL"

            _write_metrics(
                state,
                "navigational",
                "navigational",
                grounding,
                confidence,
                retrieval_score,
                context_precision,
                recall_score,
                llm_calls,
                retrieved,
                qa_time
            )

            return state

        print("[Navigate] Falling back to QA")
    # ── Normal routing ────────────────────────────────────────
    query_type = classify_from_context(question, retrieved_texts)
    state["query_type"] = query_type
    print(f"[Routing] Context-based → {query_type}")

    # ── STEP 4: ANSWER GENERATION ─────────────────────────────

    if query_type == "FULL_SUMMARY":
        retrieved_texts = clean_context_for_llm(retrieved_texts)
        ranked  = reorder_by_question(question, retrieved_texts)
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        structured_context = "\n".join(
            f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
        )
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        # answer, _ = call_llama_streaming(
        #     QA_PROMPT.format(context=structured_context[:2500], question=question),
        #     request_id=request_id, temperature=0.0
        # )
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(
                memory_context="",
                context=structured_context[:2500],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        answer     = clean_artifacts(answer).strip()
        model_used = "llama_summary"
        llm_calls  = 1

    elif query_type == "MULTIPART_QA":
        retrieved_texts = clean_context_for_llm(retrieved_texts)
        ranked  = reorder_by_question(question, retrieved_texts)
        # FIX 3 — 4000 chars for MULTIPART to capture all list items
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        emit_event(request_id, "agent_start",
                   f"🤖 MULTIPART | {len(retrieved)} chunks")
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        # answer, _ = call_llama_streaming(
        #     QA_PROMPT.format(context=context[:4000], question=question),
        #     request_id=request_id, temperature=0.0
        # )
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(
                memory_context="",
                context=context[:4000],
                question=question
            ),
            request_id=request_id, temperature=0.0
        )
        answer     = clean_artifacts(answer).strip()
        model_used = "llama_multipart"
        llm_calls  = 1

    else:
        
        # FACTUAL_QA / VERIFICATION_QA

        if not retrieved_texts:
            print("[QA] ❌ No context → NOT FOUND")
            # state["answer"] = "This information is not present in the document."   #changing to match quac keyword 
            state["answer"] = "CANNOTANSWER"
            _write_metrics(state, "not_found", "no_context",
                        0.0, 0.0, retrieval_score, context_precision,
                        recall_score, llm_calls, retrieved,
                        time.time() - qa_start_t)
            return state

        # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

        context = "\n".join(retrieved_texts)

        if (
            query_type == "FACTUAL_QA"
            and len(question.split()) <= 10
            and not is_numeric_question
            and recall_score >= 60
        ):
            print("[QA] ⚡ Direct QA (no ReAct)")

            memory_context = get_memory_context(state.get("session_id", ""))

            react_ans, _ = call_llama_streaming(
                f"""
            Previous Conversation:
            {memory_context}

            Interpret the current question in light of prior conversation when relevant.
            Use retrieved context as the authoritative source of factual information.

            Answer the question using ONLY the given context.
            Return ONLY the exact answer phrase from the context.
            Do NOT explain. Do NOT say 'not found' unless absolutely missing.

            Context:
            {context[:2000]}

            Question: {question}

            Answer:
            """,
                request_id=request_id,
                temperature=0.0
            )
            llm_calls += 1
            model_used = "llama_direct"

        else:
            react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
    question,
    faiss_index,
    query_type,
    all_chunks,
    request_id,
    recall_score,
    memory_context=get_memory_context(state.get("session_id", ""))
)
 
        react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
        # react_ans = clean_reasoning_answer(react_ans, question)
        print(f"[QA] Answer: '{react_ans[:60]}'")

        # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
        words         = react_ans.split()
        content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

     
        print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")

        refusal_phrases = [
            "not present",
            "not mentioned",
            "not available",
            "no information",
            "cannot find",
            "not in the document",
            "cannotanswer"
        ]


       
  
        # ============================================================
        # 🔥 UNIFIED DECISION LAYER (FINAL CLEAN VERSION)
        # ============================================================

       

        # Step 1: Early failure check (MUST be before everything)
        # 🔥 Refusal detection (keep this as-is)
        if any(p in react_ans.lower() for p in refusal_phrases):
            print("[QA] ⚠️ Refusal detected")
            qa_time = time.time() - qa_start_t
            # state["answer"] = "This information is not present in the document."
            state["answer"] = "CANNOTANSWER"
            _write_metrics(state, "not_found", "not_found",
                        75.0, 0.75, retrieval_score, context_precision,
                        recall_score, llm_calls, retrieved, qa_time)
            return state


        # 🔥 Compute grounding BEFORE decisions
        grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)


        # ============================================================
        # 🔥 FINAL UNIFIED DECISION LAYER
        # ============================================================

        # Step 1: Early failure check
        if not react_ans or react_ans.strip() == "":
            # answer = "This information is not present in the document."
            answer = "CANNOTANSWER"
            decision_type = "empty_answer"

        else:
            # --------------------------------------------------------
            # 🔹 VERIFICATION_QA
            # --------------------------------------------------------
            if query_type == "VERIFICATION_QA":

                # normalize to Yes/No
                if react_ans.lower() not in ["yes", "no"]:
                    react_ans = "Yes" if recall_score >= 50 else "No"

                ans_words = len(react_ans.strip().split())

                if ans_words <= 3:
                    if recall_score >= 40:
                        answer = react_ans
                        decision_type = "accepted"
                    else:
                        # answer = "This information is not present in the document."
                        answer = "CANNOTANSWER"
                        decision_type = "weak_short"

                elif grounding_score >= 60 and recall_score >= 40:
                    answer = react_ans
                    decision_type = "accepted"

                else:
                    # answer = "This information is not present in the document."
                    answer = "CANNOTANSWER"
                    decision_type = "verification_failed"

            # --------------------------------------------------------
            # 🔹 FACTUAL_QA
            # --------------------------------------------------------
            elif query_type == "FACTUAL_QA":

                if grounding_score < 50:
                    # answer = "This information is not present in the document."
                    answer = "CANNOTANSWER"
                    decision_type = "low_grounding"
                else:
                    answer = react_ans
                    decision_type = "accepted"

            # --------------------------------------------------------
            # 🔹 GENERAL FALLBACK
            # --------------------------------------------------------
            else:

                ans_words = len(react_ans.strip().split())

                if recall_score < 25:
                    # answer = "This information is not present in the document."
                    answer = "CANNOTANSWER"
                    decision_type = "low_recall"

                elif ans_words <= 3:
                    if recall_score >= 40:
                        answer = react_ans
                        decision_type = "accepted"
                    else:
                        # answer = "This information is not present in the document."
                        answer = "CANNOTANSWER"
                        decision_type = "weak_short"

                elif grounding_score < 40 and recall_score < 40:
                    # answer = "This information is not present in the document."
                    answer = "CANNOTANSWER"
                    decision_type = "low_grounding"

                elif grounding_score >= 50 or recall_score >= 50:
                    answer = react_ans
                    decision_type = "accepted"

                else:
                    # answer = "This information is not present in the document."
                    answer = "CANNOTANSWER"
                    decision_type = "uncertain"
    # ── STEP 5: METRICS ───────────────────────────────────────
    grounding  = compute_answer_grounding(answer, retrieved_texts, question)
    confidence = round(grounding / 100, 3)
    qa_time    = time.time() - qa_start_t

    print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
          f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
          f"confidence={confidence:.3f} | decision={decision_type}")

    if not answer.strip():
        answer = "Could not find a relevant answer in the PDF."
    else:
        answer = normalize_answer(answer)

    state["answer"] = answer
    _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time)
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

    # ❌ REMOVED doc_type (no longer used anywhere)
    # m["doc_type"] = state.get("doc_type", "general")

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
    m["retrieved_docs"] = retrieved
    m["type"]              = "qa"
    m["decision_type"]     = decision_type
    m["confidence_raw"]    = round(confidence, 4)



def run_pipeline(question: str, pdf_path: str, session_id: str):
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





























