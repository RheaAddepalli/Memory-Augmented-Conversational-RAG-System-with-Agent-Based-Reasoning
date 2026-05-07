

import logging
import os
import json
import time
import uuid
import sys
import datetime
from flask import Response, send_from_directory
from flask import request, jsonify
from dco_mind.cognition.memory import save_to_memory, get_retrieval_query, get_memory_context, clear_session

from dco_mind.models.llm import call_llama, _ttft_tracker
from dco_mind.core.state import DocState

from dco_mind.knowledge.ingestion import _extraction_cache, _summary_cache

from dco_mind.models.embeddings import _faiss_cache

from dco_mind.events.events import emit_event, cleanup_queue
from dco_mind.config.settings import OLLAMA_MODEL, device



def register_routes(app, workflow):
    """Register all Flask routes onto the given app instance."""

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model":  OLLAMA_MODEL,
            "device": "cuda" if device == 0 else "cpu",
            "faiss_cached":     len(_faiss_cache),
            "extracted_pdfs":   len(_extraction_cache),
            "summaries_cached": len(_summary_cache),
        })

    @app.route("/process", methods=["POST"])
    def process():
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        pdf_path   = data.get("pdf_path", "").strip()
        question   = data.get("question", "").strip()
        request_id = data.get("request_id", "")

        if not pdf_path:
            return jsonify({"error": "Missing required field: pdf_path"}), 400
        if not question:
            return jsonify({"error": "Missing required field: question"}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question too long (max 2000 chars)"}), 400
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF not found: {pdf_path}"}), 404
        if not pdf_path.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

        initial_state: DocState = {
            "pdf_path":       pdf_path,
            "question":       question,
            "extracted_text": "",
            "chunks":         [],
            "summary_chunks": [],
            "faiss_index":    None,
            "answer":         "",
            "metrics":        {},
            "doc_type":       "general",
            "query_type":     "",
            "retry_count":    0,
            "start_time":     time.time(),
            "page_count":     0,
            "char_count":     0,
            "request_id":     request_id
        }

        try:
            request_start = time.time()
            emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
            result      = workflow.invoke(initial_state)
            m           = result["metrics"]
            total_e2e   = round(time.time() - request_start, 2)
            m["e2e_latency_sec"] = total_e2e

            real_ttft = _ttft_tracker.pop(request_id, None)
            if real_ttft is not None:
                m["ttft_sec"] = real_ttft
            else:
                m["ttft_sec"] = m.get("ttft_sec", total_e2e)

            print(f"[LATENCY] E2E={total_e2e}s | TTFT={m['ttft_sec']}s | TPS={m.get('tps',0)}")
            if m.get("type") == "qa":
                print(f"[QUALITY] Grounding={m.get('answer_grounding',0)}% | "
                      f"Retrieval={m.get('retrieval_score',0)}% | "
                      f"Precision={m.get('context_precision',0)}% | "
                      f"Recall={m.get('recall_at_k',0)}% | "
                      f"Confidence={m.get('confidence_score',0)}%")

            emit_event(request_id, "done", "✅ Processing complete!")
            cleanup_queue(request_id)

            v1 = {
                "type":                 m.get("type", ""),
                "response_time_sec":    round(m.get("response_time_sec", 0), 2),
                "extraction_time_sec":  round(m.get("extraction_time_sec", 0), 2),
                "pages_processed":      m.get("pages_processed", 0),
                "characters_processed": m.get("characters_processed", 0),
                "words_processed":      m.get("words_processed", 0),
            }
            if m.get("type") == "summary":
                v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
                v1["summary_length_words"] = m.get("summary_length_words", 0)
            if m.get("type") == "qa":
                v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
                v1["confidence_score"] = round(m.get("confidence_score", 0), 2)

            v2 = {
                "ttft_sec":          round(m.get("ttft_sec", 0), 2),
                "e2e_latency_sec":   round(m.get("e2e_latency_sec", 0), 2),
                "tps":               m.get("tps", 0),
                "doc_type":          m.get("doc_type", "general"),
                "query_type":        m.get("query_type", ""),
                "model_used":        m.get("model_used", ""),
                "chunks_created":    m.get("chunks_created", 0),
                "retry_count":       m.get("retry_count", 0),
                "llm_calls":         m.get("llm_calls", 0),
                "retrieval_score":   m.get("retrieval_score", 0),
                "context_precision": m.get("context_precision", 0),
                "answer_grounding":  m.get("answer_grounding", 0),
                "recall_at_k":       m.get("recall_at_k", 0),
            }
            if m.get("type") == "summary":
                v2["parallel_workers"] = m.get("parallel_workers", 0)
                v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
                v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
            if m.get("type") == "qa":
                v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)
                v2["decision_type"]    = m.get("decision_type", "accepted")
                v2["confidence_raw"]   = m.get("confidence_raw", 0.0)

            return jsonify({
                "answer":     result["answer"],
                "metrics":    {**v1, **v2},
                "metrics_v1": v1,
                "metrics_v2": v2,
            })

        except Exception as e:
            return jsonify({
                "answer":     f"Error: {str(e)}",
                "metrics":    {},
                "metrics_v1": {},
                "metrics_v2": {}
            })

    def is_refusal_answer(answer: str) -> bool:
        if not answer or len(answer.strip()) < 3:
            return True
        prompt = (
            f"Does this answer say that the requested information "
            f"was not found, not present, or not available?\n\n"
            f"Answer: {answer}\n\n"
            f"Reply with YES or NO only."
        )
        try:
            result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
            is_refusal = "YES" in result
            print(f"[Refusal] '{answer[:60]}' → {'REFUSAL' if is_refusal else 'FACTUAL'}")
            return is_refusal
        except Exception:
            return False

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        try:
            data        = request.json or {}
            session_name = data.get("session_name", "").strip()
            if session_name:
                import builtins, datetime as _dt
                _LOG_DIR = r"C:\xampp\htdocs\GenAI-Doc-old\dco_mind\evaluation\results\backend_logs"
                _new_log = open(os.path.join(_LOG_DIR, f"run_logs_backend_{session_name}.txt"), "a", encoding="utf-8", buffering=1)
                _rp = builtins._real_print  # the original print saved in settings.py
                def _tee(*args, **kwargs):
                    _rp(*args, **kwargs)
                    msg = kwargs.get("sep", " ").join(str(a) for a in args) + kwargs.get("end", "\n")
                    _new_log.write(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
                    _new_log.flush()
                builtins.print = _tee

            pdf_path    = data.get("pdf_path", "").strip()
            run_desc    = data.get("run_description", f"run_{int(time.time())}")

            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

            pdf_name = os.path.basename(pdf_path).lower().strip()
            print(f"[DEBUG] PDF NAME: {pdf_name}")

            dataset_map = {
                "rhea aiml resume updated.pdf": "datasets/resume.json",
                "the-story-of-doctor-dolittle.pdf": "datasets/story.json",
                "ml.pdf": "datasets/ml.json"
            }

            print(f"[DEBUG] DATASET MAP KEYS: {list(dataset_map.keys())}")

            dataset_file = dataset_map.get(pdf_name)

            if not dataset_file:
                return jsonify({"error": f"No dataset mapping for {pdf_name}"}), 404

            dataset_file = os.path.join(os.path.dirname(__file__), "..", dataset_file)
            dataset_file = os.path.abspath(dataset_file)

            print(f"[DEBUG] FULL DATASET PATH: {dataset_file}")

            if not os.path.exists(dataset_file):
                return jsonify({"error": f"Dataset file missing: {dataset_file}"}), 404

            with open(dataset_file, "r") as f:
                dataset = json.load(f)

            questions = dataset.get("questions", [])
            if not questions:
                return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

            results       = []
            pass_count    = 0
            partial_count = 0
            fail_count    = 0

            print(f"[Evaluate] Starting evaluation run: {run_desc}")
            print(f"[Evaluate] PDF: {pdf_path}")
            print(f"[Evaluate] Questions: {len(questions)}")

            for item in questions:

                # ============================================================
                # CASE 1 — CONVERSATION
                # ============================================================
                if "conversation" in item:

                    session_id = f"conv_{item['id']}"
                    clear_session(session_id)

                    turn_keywords = item.get("turn_keywords", [])

                    for turn_idx, q_text in enumerate(item["conversation"]):

                        if turn_keywords and turn_idx < len(turn_keywords):
                            kw_list = [kw.lower() for kw in turn_keywords[turn_idx]]
                        else:
                            kw_list = [kw.lower() for kw in item.get("expected_keywords", [])]

                        q = {
                            "id":                 f"{item['id']}_t{turn_idx}",
                            "question":           q_text,
                            "query_type":         item.get("query_type", "FACTUAL_QA"),
                            "expected_keywords":  kw_list,
                            "pass_threshold":     item.get("pass_threshold", 1),
                            "max_score":          max(1, len(kw_list)),
                            "hallucination_test": item.get("hallucination_test", False),
                            "roberta_test":       item.get("roberta_test", False),
                            "skip":               item.get("skip", False),
                            "tests":              item.get("tests", ""),
                        }

                        try:
                            q_id       = q["id"]
                            question   = get_retrieval_query(session_id, q_text)
                            query_type = q["query_type"]
                            keywords   = q["expected_keywords"]
                            threshold  = q["pass_threshold"]
                            max_score  = q["max_score"]
                            is_halluc  = q["hallucination_test"]
                            is_roberta = q["roberta_test"]

                            if q["skip"]:
                                print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
                                continue

                            print(f"[Evaluate] Q{q_id}: {question[:60]}...")

                            initial_state: DocState = {
                                "pdf_path":       pdf_path,
                                "question":       question,
                                "extracted_text": "",
                                "chunks":         [],
                                "summary_chunks": [],
                                "faiss_index":    None,
                                "answer":         "",
                                "metrics":        {},
                                "doc_type":       "general",
                                "query_type":     "",
                                "retry_count":    0,
                                "start_time":     time.time(),
                                "page_count":     0,
                                "char_count":     0,
                                "request_id":     ""
                            }

                            try:
                                result     = workflow.invoke(initial_state)
                                answer     = result.get("answer", "")
                                save_to_memory(session_id, question, answer)
                                metrics    = result.get("metrics", {})
                                model_used = metrics.get("model_used", "unknown")
                                recall_k   = metrics.get("recall_at_k", 0)
                                grounding  = metrics.get("answer_grounding", 0)
                                confidence = metrics.get("confidence_score", 0)
                                print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
                            except Exception as invoke_err:
                                print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
                                answer     = f"ERROR: {str(invoke_err)}"
                                model_used, recall_k, grounding, confidence = "error", 0, 0, 0

                            answer_lower = answer.lower()
                            short_answer = len(answer.split()) <= 3
                            hallucination_detected = not is_halluc and grounding < 60 and not short_answer

                            if hallucination_detected:
                                print(f"[Eval] ❌ Low grounding ({grounding:.1f}%) → hallucination detected")

                            if is_halluc:
                                found_safe = is_refusal_answer(answer)
                                if found_safe:
                                    verdict, keyword_hits, hallucination_detected = "PASS", max_score, False
                                else:
                                    verdict, keyword_hits, hallucination_detected = "FAIL", 0, True
                                    print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")

                            elif hallucination_detected:
                                verdict, keyword_hits = "FAIL", 0
                                print(f"[Eval] ❌ Forced FAIL due to low grounding")

                            else:
                                keyword_hits = sum(1 for kw in keywords if kw in answer_lower)
                                if keyword_hits >= threshold:
                                    verdict = "PASS"
                                elif keyword_hits >= max(1, threshold // 2):
                                    verdict = "PARTIAL"
                                else:
                                    verdict = "FAIL"

                            if verdict == "PASS":       pass_count += 1
                            elif verdict == "PARTIAL":  partial_count += 1
                            else:                       fail_count += 1

                            verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
                            print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
                                  f"hits={keyword_hits}/{max_score} | model={model_used} | "
                                  f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

                            results.append({
                                "id":               q_id,
                                "question":         question,
                                "query_type":       query_type,
                                "verdict":          verdict,
                                "keyword_hits":     keyword_hits,
                                "max_score":        max_score,
                                "pass_threshold":   threshold,
                                "model_used":       model_used,
                                "recall_at_k":      round(recall_k, 1),
                                "answer_grounding": round(grounding, 1),
                                "confidence":       round(confidence, 1),
                                "hallucination":    hallucination_detected,
                                "roberta_test":     is_roberta,
                                "actual_answer":    answer[:300],
                                "tests":            q.get("tests", ""),
                                "conversation_id":  item["id"],
                                "turn":             turn_idx,
                            })

                        except Exception as q_err:
                            print(f"[Evaluate] Q{q_id} unexpected error: {q_err}")

                # ============================================================
                # CASE 2 — PLAIN QUESTION
                # ============================================================
                else:
                    q_id       = item.get("id", "?")
                    query_type = item.get("query_type", "FACTUAL_QA")
                    keywords   = [kw.lower() for kw in item.get("expected_keywords", [])]
                    threshold  = item.get("pass_threshold", 1)
                    max_score  = item.get("max_score", max(1, len(keywords)))
                    is_halluc  = item.get("hallucination_test", False)
                    is_roberta = item.get("roberta_test", False)

                    if item.get("skip", False):
                        print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
                        continue

                    question = item.get("question", "")
                    print(f"[Evaluate] Q{q_id}: {question[:60]}...")

                    initial_state: DocState = {
                        "pdf_path":       pdf_path,
                        "question":       question,
                        "extracted_text": "",
                        "chunks":         [],
                        "summary_chunks": [],
                        "faiss_index":    None,
                        "answer":         "",
                        "metrics":        {},
                        "doc_type":       "general",
                        "query_type":     "",
                        "retry_count":    0,
                        "start_time":     time.time(),
                        "page_count":     0,
                        "char_count":     0,
                        "request_id":     ""
                    }

                    try:
                        result     = workflow.invoke(initial_state)
                        answer     = result.get("answer", "")
                        metrics    = result.get("metrics", {})
                        model_used = metrics.get("model_used", "unknown")
                        recall_k   = metrics.get("recall_at_k", 0)
                        grounding  = metrics.get("answer_grounding", 0)
                        confidence = metrics.get("confidence_score", 0)
                        print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
                    except Exception as invoke_err:
                        print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
                        answer     = f"ERROR: {str(invoke_err)}"
                        model_used, recall_k, grounding, confidence = "error", 0, 0, 0

                    answer_lower = answer.lower()
                    short_answer = len(answer.split()) <= 3
                    hallucination_detected = not is_halluc and grounding < 60 and not short_answer

                    if hallucination_detected:
                        print(f"[Eval] ❌ Low grounding ({grounding:.1f}%) → hallucination detected")

                    if is_halluc:
                        found_safe = is_refusal_answer(answer)
                        if found_safe:
                            verdict, keyword_hits, hallucination_detected = "PASS", max_score, False
                        else:
                            verdict, keyword_hits, hallucination_detected = "FAIL", 0, True
                            print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")

                    elif hallucination_detected:
                        verdict, keyword_hits = "FAIL", 0
                        print(f"[Eval] ❌ Forced FAIL due to low grounding")

                    else:
                        keyword_hits = sum(1 for kw in keywords if kw in answer_lower)
                        if keyword_hits >= threshold:
                            verdict = "PASS"
                        elif keyword_hits >= max(1, threshold // 2):
                            verdict = "PARTIAL"
                        else:
                            verdict = "FAIL"

                    if verdict == "PASS":       pass_count += 1
                    elif verdict == "PARTIAL":  partial_count += 1
                    else:                       fail_count += 1

                    verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
                    print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
                          f"hits={keyword_hits}/{max_score} | model={model_used} | "
                          f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

                    results.append({
                        "id":               q_id,
                        "question":         question,
                        "query_type":       query_type,
                        "verdict":          verdict,
                        "keyword_hits":     keyword_hits,
                        "max_score":        max_score,
                        "pass_threshold":   threshold,
                        "model_used":       model_used,
                        "recall_at_k":      round(recall_k, 1),
                        "answer_grounding": round(grounding, 1),
                        "confidence":       round(confidence, 1),
                        "hallucination":    hallucination_detected,
                        "roberta_test":     is_roberta,
                        "actual_answer":    answer[:300],
                        "tests":            item.get("tests", ""),
                        "conversation_id":  None,
                        "turn":             None,
                    })

            total = len(results)
            if total == 0:
                return jsonify({"error": "All questions were skipped or failed"}), 400

            pass_rate = round(pass_count / total * 100, 1)
            run_summary = {
                "run_id":          run_desc,
                "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
                "pdf":             os.path.basename(pdf_path),
                "total_questions": total,
                "pass":            pass_count,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       pass_rate,
                "results":         results
            }

            results_path = os.path.join(os.path.dirname(__file__), "..", "grounding_results.json")
            try:
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        existing = json.load(f)
                else:
                    existing = {"runs": []}
                existing["runs"].append(run_summary)
                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"[Evaluate] Results saved to grounding_results.json")
            except Exception as save_err:
                print(f"[Evaluate] Warning: could not save results: {save_err}")

            print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
                  f"Fail={fail_count} | Pass rate={pass_rate}%")

            return jsonify({
                "run_id":          run_desc,
                "total_questions": total,
                "pass":            pass_count,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       f"{pass_rate}%",
                "results":         results
            })

        except Exception as fatal_err:
            print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500

    UPLOAD_FOLDER = "uploads"

    @app.route("/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        file = request.files["file"]
        if not file.filename.endswith(".pdf"):
            return jsonify({"status": "error", "message": "Only PDFs allowed"}), 400
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(save_path)
        return jsonify({"status": "success", "filename": filename})

    @app.route("/ask", methods=["POST"])
    def ask():
        data       = request.json or {}
        question   = data.get("question", "").strip()
        filename   = data.get("filename", "").strip()
        request_id = data.get("request_id", "")
        session_id = data.get("session_id", "")
        pdf_path   = os.path.join(UPLOAD_FOLDER, filename)

        if not question:
            return jsonify({"error": "Missing question"}), 400
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"File not found: {filename}"}), 404

        original_question = question
        question = get_retrieval_query(session_id, question)
        emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

        initial_state: DocState = {
            "pdf_path":       pdf_path,
            "question":       question,
            "extracted_text": "",
            "chunks":         [],
            "summary_chunks": [],
            "faiss_index":    None,
            "answer":         "",
            "metrics":        {},
            "doc_type":       "general",
            "query_type":     "",
            "retry_count":    0,
            "start_time":     time.time(),
            "page_count":     0,
            "char_count":     0,
            "request_id":     request_id
        }
        try:
            request_start = time.time()
            emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
            result    = workflow.invoke(initial_state)
            m         = result["metrics"]
            total_e2e = round(time.time() - request_start, 2)
            m["e2e_latency_sec"] = total_e2e
            real_ttft = _ttft_tracker.pop(request_id, None)
            m["ttft_sec"] = real_ttft if real_ttft is not None else total_e2e
            emit_event(request_id, "done", "✅ Complete!")
            cleanup_queue(request_id)
            save_to_memory(session_id, original_question, result["answer"])
            return jsonify({
                "answer":          result["answer"],
                "metrics":         m,
                "rewritten_query": question if question != original_question else None
            })
        except Exception as e:
            return jsonify({"answer": f"Error: {str(e)}", "metrics": {}})

    @app.route("/clear_session", methods=["POST"])
    def clear_session_route():
        data       = request.json or {}
        session_id = data.get("session_id", "")
        if session_id:
            clear_session(session_id)
        return jsonify({"status": "cleared"})

    @app.route("/stream", methods=["GET"])
    def stream():
        request_id = request.args.get("request_id", "")

        def event_generator():
            import tempfile, time
            event_file = os.path.join(tempfile.gettempdir(), f"docmind_{request_id}.jsonl")
            timeout    = time.time() + 300
            seen_lines = 0
            while time.time() < timeout:
                if os.path.exists(event_file):
                    with open(event_file, "r") as f:
                        lines = f.readlines()
                    for line in lines[seen_lines:]:
                        seen_lines += 1
                        yield f"data: {line.strip()}\n\n"
                        if json.loads(line.strip()).get("type") == "done":
                            return
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'message': ''})}\n\n"
                time.sleep(0.3)

        return Response(event_generator(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


























