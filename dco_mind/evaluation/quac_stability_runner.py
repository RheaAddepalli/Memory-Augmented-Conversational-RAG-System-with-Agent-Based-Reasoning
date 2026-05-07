"""
DocMind — QuAC Dataset Stability Evaluation
Evaluates on QuAC-format conversations using F1 scoring.
- Normal questions: F1 score against gold answer
- CANNOTANSWER questions: refusal detection (hallucination test)
- Single reusable temp file for context (no mess)
- Same logging, stability report, save results as main runner
"""

import requests
import json
import time
import sys
import os
import re
import datetime
import tempfile
from collections import defaultdict

# ============================================================
# LOGGING SETUP
# ============================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

_LOG_DIR = r"C:\xampp\htdocs\GenAI-Doc-old\dco_mind\evaluation\results"
os.makedirs(_LOG_DIR, exist_ok=True)
_log_file = open(os.path.join(_LOG_DIR, "run_logs_quac_runner.txt"), "a", encoding="utf-8")
_log_file.write(f"""
################################################################################
##                                                                            ##
##   QUAC SESSION START : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                  ##
##                                                                            ##
################################################################################
""")
_log_file.flush()

sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)
# ============================================================


# ── CONFIG ──────────────────────────────────────────────────
FLASK_URL       = "http://127.0.0.1:5000/evaluate"
BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
QUAC_FILE       = os.path.join(BASE_DIR, "datasets", "quac_eval_10.json")
TEMP_CONTEXT    = os.path.join(BASE_DIR, "datasets", "_quac_temp_context.txt")
RESULTS_FILE    = os.path.join(BASE_DIR, "evaluation", "results", "quac_stability_results.json")
NUM_RUNS        = 1

# F1 thresholds
F1_PASS_THRESHOLD    = 0.5
F1_PARTIAL_THRESHOLD = 0.3
# ────────────────────────────────────────────────────────────


# ============================================================
# F1 SCORING
# ============================================================

def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1

    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def f1_verdict(f1_score: float) -> str:
    if f1_score >= F1_PASS_THRESHOLD:
        return "PASS"
    elif f1_score >= F1_PARTIAL_THRESHOLD:
        return "PARTIAL"
    else:
        return "FAIL"


def is_refusal(answer: str) -> bool:
    """Check if model correctly refused to answer."""
    refusal_phrases = [
        "not present",
        "not mentioned",
        "not available",
        "no information",
        "cannot find",
        "not in the document",
        "cannot answer",
        "not found",
        "not stated",
    ]
    answer_lower = answer.lower()
    return any(p in answer_lower for p in refusal_phrases)


# ============================================================
# QUAC DATASET LOADING
# ============================================================

def load_quac(quac_path: str) -> list:
    """
    Load QuAC file and return flat list of paragraphs.
    Each paragraph: { context, qas: [{question, gold_answer, is_hallucination}] }
    """
    with open(quac_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paragraphs = []
    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "").strip()
            qas = []
            for qa in para.get("qas", []):
                question    = qa.get("question", "").strip()
                answers     = qa.get("answers", [])
                gold_answer = answers[0].get("text", "").strip() if answers else ""
                is_halluc   = (gold_answer.upper() == "CANNOTANSWER")
                qas.append({
                    "id":               qa.get("id", ""),
                    "question":         question,
                    "gold_answer":      gold_answer,
                    "is_hallucination": is_halluc,
                    "followup":         qa.get("followup", ""),
                    "yesno":            qa.get("yesno", ""),
                })
            if context and qas:
                paragraphs.append({"context": context, "qas": qas})

    print(f"[QuAC] Loaded {len(paragraphs)} paragraphs from {os.path.basename(quac_path)}")
    return paragraphs


# ============================================================
# TEMP FILE MANAGEMENT
# ============================================================

def write_temp_context(context: str) -> str:
    """Write context to reusable temp file. Returns path."""
    with open(TEMP_CONTEXT, "w", encoding="utf-8") as f:
        f.write(context)
    return TEMP_CONTEXT


def cleanup_temp():
    """Remove temp context file."""
    if os.path.exists(TEMP_CONTEXT):
        os.remove(TEMP_CONTEXT)
        print(f"[QuAC] Cleaned up temp file")


# ============================================================
# DATASET BUILDING FOR /evaluate
# ============================================================

def build_quac_dataset(qas: list) -> dict:
    """
    Convert QuAC qas into the dataset format /evaluate expects.
    Uses gold answer as single expected keyword for routing —
    actual scoring is done here with F1, not in api_routes.
    """
    questions = []
    for i, qa in enumerate(qas):
        if qa["is_hallucination"]:
            # Hallucination test — expect refusal
            questions.append({
                "id":                i + 1,
                "question":         qa["question"],
                "expected_keywords": ["not present", "cannot answer", "not found"],
                "pass_threshold":    1,
                "hallucination_test": True,
            })
        else:
            # Normal question — use gold answer words as keywords
            # (api_routes will do basic scoring, we override with F1 here)
            gold_words = [
                w for w in _normalize_text(qa["gold_answer"]).split()
                if len(w) > 3
            ][:5]  # top 5 meaningful words
            questions.append({
                "id":                i + 1,
                "question":         qa["question"],
                "expected_keywords": gold_words if gold_words else [qa["gold_answer"][:30]],
                "pass_threshold":    1,
                "hallucination_test": False,
            })

    return {"questions": questions}


# ============================================================
# EVALUATE ONE PARAGRAPH
# ============================================================

def evaluate_paragraph(para_idx: int, para: dict, run_num: int,
                        session_name: str) -> tuple:
    """
    Evaluate one QuAC paragraph.
    Returns (results_list, elapsed_time)
    """
    context = para["context"]
    qas     = para["qas"]

    # Write context to temp file
    temp_path = write_temp_context(context)

    # Build dataset for this paragraph
    dataset = build_quac_dataset(qas)

    # Write dataset to temp JSON (api_routes reads from file)
    temp_dataset_path = TEMP_CONTEXT.replace(".txt", "_dataset.json")
    with open(temp_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(f"\n{'='*60}")
    print(f"  PARAGRAPH {para_idx + 1} | RUN {run_num}")
    print(f"  Questions : {len(qas)} "
          f"({sum(1 for q in qas if q['is_hallucination'])} hallucination)")
    print(f"{'='*60}")

    payload = {
        "pdf_path":        temp_path,
        "run_description": f"{session_name}_para{para_idx+1}_run_{run_num}",
        "request_id":      f"quac_para{para_idx+1}_run_{run_num}",
        "session_name":    session_name,
        "dataset_path":    temp_dataset_path,
    }

    start = time.time()
    try:
        resp = requests.post(FLASK_URL, json=payload, timeout=7200)
        elapsed = round(time.time() - start, 1)

        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}: {resp.text[:300]}")
            return [], elapsed

        api_results = resp.json().get("results", [])

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        elapsed = round(time.time() - start, 1)
        return [], elapsed

    # ── F1 SCORING OVERRIDE ───────────────────────────────
    final_results = []
    for i, qa in enumerate(qas):
        api_result  = api_results[i] if i < len(api_results) else {}
        actual_ans  = api_result.get("actual_answer", "")
        model_used  = api_result.get("model_used", "unknown")
        recall      = api_result.get("recall_at_k", 0.0)
        grounding   = api_result.get("answer_grounding", 0.0)
        confidence  = api_result.get("confidence", 0.0)

        if qa["is_hallucination"]:
            # CANNOTANSWER — check if model correctly refused
            if is_refusal(actual_ans):
                verdict  = "PASS"
                f1_score = 1.0
                print(f"[Q{i+1}] ✅ HALLUCINATION PASS — correctly refused")
            else:
                verdict  = "FAIL"
                f1_score = 0.0
                print(f"[Q{i+1}] ❌ HALLUCINATION FAIL — model answered: '{actual_ans[:60]}'")
        else:
            # Normal question — F1 scoring
            f1_score = compute_f1(actual_ans, qa["gold_answer"])
            verdict  = f1_verdict(f1_score)
            icon     = "✅" if verdict == "PASS" else ("⚠️" if verdict == "PARTIAL" else "❌")
            print(f"[Q{i+1}] {icon} {verdict} | F1={f1_score:.3f} | "
                  f"Gold: '{qa['gold_answer'][:40]}' | "
                  f"Got: '{actual_ans[:40]}'")

        final_results.append({
            "id":               i + 1,
            "question":         qa["question"],
            "gold_answer":      qa["gold_answer"],
            "actual_answer":    actual_ans,
            "f1_score":         f1_score,
            "verdict":          verdict,
            "is_hallucination": qa["is_hallucination"],
            "model_used":       model_used,
            "recall_at_k":      recall,
            "answer_grounding": grounding,
            "confidence":       confidence,
            "para_idx":         para_idx,
        })

    # Cleanup temp dataset file
    if os.path.exists(temp_dataset_path):
        os.remove(temp_dataset_path)

    return final_results, elapsed


# ============================================================
# RESULTS + REPORTING
# ============================================================

def load_existing_results() -> dict:
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
            if "sessions" not in data:
                data["sessions"] = []
            return data
        except Exception as e:
            print(f"[Warning] Could not read results: {e}")
    return {"sessions": []}


def get_session_name(existing: dict) -> str:
    next_num  = len(existing.get("sessions", [])) + 1
    auto_name = f"QuAC-Test-{next_num}"
    try:
        user_input = input(
            f"\nEnter session name (press Enter for auto '{auto_name}'): "
        ).strip()
        session_name = user_input if user_input else auto_name
    except (EOFError, KeyboardInterrupt):
        session_name = auto_name
    os.environ["DCO_RUN_NAME"] = session_name
    return session_name


def compute_summary(all_results: list) -> dict:
    """Compute overall metrics across all paragraphs."""
    normal_results = [r for r in all_results if not r["is_hallucination"]]
    halluc_results = [r for r in all_results if r["is_hallucination"]]

    total       = len(all_results)
    total_pass  = sum(1 for r in all_results if r["verdict"] == "PASS")
    total_part  = sum(1 for r in all_results if r["verdict"] == "PARTIAL")
    total_fail  = sum(1 for r in all_results if r["verdict"] == "FAIL")

    avg_f1_normal = (
        round(sum(r["f1_score"] for r in normal_results) / len(normal_results), 4)
        if normal_results else 0.0
    )

    halluc_pass = sum(1 for r in halluc_results if r["verdict"] == "PASS")
    halluc_acc  = (
        round(halluc_pass / len(halluc_results) * 100, 1)
        if halluc_results else 0.0
    )

    return {
        "total_questions":     total,
        "total_pass":          total_pass,
        "total_partial":       total_part,
        "total_fail":          total_fail,
        "overall_accuracy":    round(total_pass / total * 100, 1) if total else 0.0,
        "avg_f1_normal":       avg_f1_normal,
        "hallucination_accuracy": halluc_acc,
        "normal_questions":    len(normal_results),
        "hallucination_questions": len(halluc_results),
    }


def print_report(summary: dict, all_results: list, session_name: str,
                 total_time: float):
    print(f"\n{'='*60}")
    print(f"  QUAC EVALUATION REPORT")
    print(f"  Session : {session_name}")
    print(f"{'='*60}")
    print(f"\n  {'Q':<4} {'Verdict':<8} {'F1':>6}  {'Halluc':<8}  Question")
    print(f"  {'─'*4} {'─'*8} {'─'*6}  {'─'*8}  {'─'*40}")

    for r in all_results:
        icon    = "✅" if r["verdict"] == "PASS" else ("⚠️" if r["verdict"] == "PARTIAL" else "❌")
        halluc  = "🚨 YES" if r["is_hallucination"] else "NO"
        q_short = r["question"][:45]
        print(f"  {r['id']:<4} {icon} {r['verdict']:<6} {r['f1_score']:>6.3f}  {halluc:<8}  {q_short}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY METRICS")
    print(f"{'='*60}")
    print(f"  Session              : {session_name}")
    print(f"  Total questions      : {summary['total_questions']}")
    print(f"  Normal questions     : {summary['normal_questions']}")
    print(f"  Hallucination tests  : {summary['hallucination_questions']}")
    print(f"")
    print(f"  ✅ Overall Accuracy  : {summary['overall_accuracy']}%  "
          f"({summary['total_pass']}/{summary['total_questions']} PASS)")
    print(f"  ⚠️  Partial rate      : "
          f"{round(summary['total_partial']/summary['total_questions']*100,1)}%  "
          f"({summary['total_partial']}/{summary['total_questions']} PARTIAL)")
    print(f"  ❌ Fail rate         : "
          f"{round(summary['total_fail']/summary['total_questions']*100,1)}%  "
          f"({summary['total_fail']}/{summary['total_questions']} FAIL)")
    print(f"")
    print(f"  📊 Avg F1 (normal)   : {summary['avg_f1_normal']:.4f}")
    print(f"  🚨 Halluc accuracy   : {summary['hallucination_accuracy']}%")
    print(f"  ⏱️  Total time        : {round(total_time, 1)}s")
    print(f"{'='*60}")


def save_session(session: dict, existing: dict):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing["sessions"].append(session)
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  Results saved → {RESULTS_FILE}")
        print(f"  Total sessions stored: {len(existing['sessions'])}")
    except Exception as e:
        print(f"[ERROR] Could not save results: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\n=== QUAC RUN START: {datetime.datetime.now()} ===\n")

    # Load existing results and get session name
    existing     = load_existing_results()
    session_name = get_session_name(existing)

    # Load QuAC dataset
    if not os.path.exists(QUAC_FILE):
        print(f"[ERROR] QuAC file not found: {QUAC_FILE}")
        return

    paragraphs = load_quac(QUAC_FILE)
    if not paragraphs:
        print("[ERROR] No paragraphs loaded from QuAC file")
        return

    print(f"\n{'='*60}")
    print(f"  DocMind QuAC Evaluation")
    print(f"  Session    : {session_name}")
    print(f"  Dataset    : {os.path.basename(QUAC_FILE)}")
    print(f"  Paragraphs : {len(paragraphs)}")
    print(f"  Runs       : {NUM_RUNS}")
    print(f"  F1 PASS    : ≥ {F1_PASS_THRESHOLD}")
    print(f"  F1 PARTIAL : ≥ {F1_PARTIAL_THRESHOLD}")
    print(f"{'='*60}")

    total_start  = time.time()
    all_results  = []

    for run_num in range(1, NUM_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run_num} / {NUM_RUNS}")
        print(f"{'='*60}")

        run_results = []
        for para_idx, para in enumerate(paragraphs):
            results, elapsed = evaluate_paragraph(
                para_idx, para, run_num, session_name
            )
            run_results.extend(results)
            print(f"[Para {para_idx+1}] Done in {elapsed}s")

        all_results = run_results  # for single run; extend for multi-run

    total_time = time.time() - total_start

    # Compute and print summary
    summary = compute_summary(all_results)
    print_report(summary, all_results, session_name, total_time)

    # Save results
    session = {
        "name":        session_name,
        "date":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":     os.path.basename(QUAC_FILE),
        "num_runs":    NUM_RUNS,
        "summary":     summary,
        "results":     all_results,
        "total_time":  round(total_time, 1),
    }
    save_session(session, existing)

    # Cleanup temp file
    cleanup_temp()

    print(f"\n=== QUAC RUN COMPLETE: {datetime.datetime.now()} ===\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        _log_file.close()