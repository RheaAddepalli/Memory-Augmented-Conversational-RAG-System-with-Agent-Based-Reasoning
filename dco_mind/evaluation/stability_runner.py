
"""
DocMind — Multi-Run Stability Evaluation
Runs /evaluate N times and reports:
  - Per-question accuracy across runs
  - Overall accuracy
  - Hallucination rate
  - Instability rate (questions that change verdict across runs)
  - All results saved to stability_results_final.json (append, never overwrite)
  - Each session named by user (auto-increments if no name given)
"""

import requests
import json
import time
import sys
import os
import datetime
from collections import defaultdict

# ============================================================
# LOGGING SETUP — at top before anything else
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

# CHANGE 1: updated log filename
_log_file = open(
    os.path.join(_LOG_DIR, "run_logs_runner_final.txt"),
    "a",
    encoding="utf-8"
)
_log_file.write(f"""
################################################################################
##                                                                            ##
##   SESSION START : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                        ##
##                                                                            ##
################################################################################
""")
_log_file.flush()

sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)
# ============================================================


# ── CONFIG ──────────────────────────────────────────────────
FLASK_URL = "http://127.0.0.1:5000/evaluate"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PDF_PATHS = [
    r"C:\Users\Rhearitu\Downloads\rhea resume-ziroh labs.pdf.pdf",
    r"C:\Users\Rhearitu\Downloads\the-story-of-doctor-dolittle.pdf",
    r"C:\Users\Rhearitu\Downloads\ml.pdf"
]

# CHANGE 3: updated dataset filenames
PDF_DATASET_MAP = {
    "rhea resume-ziroh labs.pdf.pdf": os.path.join(BASE_DIR, "datasets", "resume2.json"),
    "the-story-of-doctor-dolittle.pdf": os.path.join(BASE_DIR, "datasets", "story2.json"),
    "ml.pdf": os.path.join(BASE_DIR, "datasets", "ml2.json")
}
NUM_RUNS = 1

# CHANGE 2: updated results filename
RESULTS_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "results",
    "stability_results_final.json"
)
# ────────────────────────────────────────────────────────────


def load_existing_results() -> dict:
    """Load existing results file or create fresh structure."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
            if "sessions" not in data:
                data["sessions"] = []
            return data
        except Exception as e:
            print(f"[Warning] Could not read {RESULTS_FILE}: {e} — starting fresh")
    return {"sessions": []}


def get_next_session_number(existing: dict) -> int:
    return len(existing.get("sessions", [])) + 1


def get_session_name(next_num: int) -> str:
    auto_name = f"Test {next_num}"
    try:
        user_input = input(
            f"\nEnter session name (press Enter for auto '{auto_name}'): "
        ).strip()
        session_name = user_input if user_input else auto_name
    except (EOFError, KeyboardInterrupt):
        session_name = auto_name

    os.environ["DCO_RUN_NAME"] = session_name
    return session_name


def load_dataset_meta(pdf_path: str) -> dict:
    meta = {}
    pdf_name = os.path.basename(pdf_path)

    dataset_file = None
    for key in PDF_DATASET_MAP:
        if key.lower() in pdf_name.lower():
            dataset_file = PDF_DATASET_MAP[key]
            break

    if not dataset_file:
        raise ValueError(f"No dataset mapped for: {pdf_name}")

    print(f"[DEBUG] RESOLVED DATASET PATH: {dataset_file}")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    try:
        with open(dataset_file, "r") as f:
            dataset = json.load(f)

        questions = (
            dataset
            if isinstance(dataset, list)
            else dataset.get("questions", [])
        )

        for q in questions:
            meta[q["id"]] = {
                "hallucination_test": q.get("hallucination_test", False),
                "skip":               q.get("skip", False),
            }

    except Exception as e:
        print(f"[Error] Could not read dataset: {e}")

    return meta


def run_evaluate(pdf_path: str, run_num: int, session_name: str) -> dict:
    payload = {
        "pdf_path":        pdf_path,
        "run_description": f"{session_name}_run_{run_num}",
        "request_id":      f"{os.path.basename(pdf_path)}_run_{run_num}",
        "session_name":    session_name
    }

    print(f"\n{'='*60}")
    print(f"  RUN {run_num} / {NUM_RUNS}")
    print(f"  PDF : {os.path.basename(pdf_path)}")
    print(f"{'='*60}")

    try:
        resp = requests.post(FLASK_URL, json=payload, timeout=7200)
        if resp.status_code != 200:
            print(f"[ERROR] Run {run_num} returned HTTP {resp.status_code}")
            print(f"[ERROR] Response: {resp.text[:500]}")
            return {}
        try:
            data = resp.json()
        except Exception as json_err:
            print(f"[ERROR] Run {run_num} — invalid JSON response: {json_err}")
            print(f"[ERROR] Raw response: {resp.text[:500]}")
            return {}
        return data
    except requests.exceptions.Timeout:
        print(f"[ERROR] Run {run_num} timed out")
        return {}
    except Exception as e:
        print(f"[ERROR] Run {run_num} failed: {e}")
        return {}


def build_run_record(run_num: int, result: dict, elapsed: float) -> dict:
    questions_detail = []
    for r in result.get("results", []):
        questions_detail.append({
            "id":               r.get("id"),
            "question":         r.get("question", ""),
            "conversation_id":  r.get("conversation_id", None),
            "turn":             r.get("turn", None),
            "verdict":          r.get("verdict", "FAIL"),
            # REMOVED: keyword_hits, max_score, pass_threshold
            # NEW semantic metrics
            "answer_score":           r.get("answer_score", 0.0),
            "retrieval_recall":       r.get("retrieval_recall", 0.0),
            "retrieval_precision":    r.get("retrieval_precision", 0.0),
            "followup_question":      r.get("followup_question", False),
            "followup_score":         r.get("followup_score", None),
            "rewrite_triggered":      r.get("rewrite_triggered", False),
            "rewrite_gain":           r.get("rewrite_gain", 0.0),
            "hallucination_detected": r.get("hallucination_detected", False),
            "rewritten_query":        r.get("rewritten_query", ""),
            "query_rewrite":          r.get("query_rewrite", ""),
            "retrieved_docs":         r.get("retrieved_docs", []),
            "pass_numeric":           r.get("pass_numeric", 0),
            # KEPT existing useful fields
            "model_used":       r.get("model_used", "unknown"),
            "recall_at_k":      r.get("recall_at_k", 0.0),
            "answer_grounding": r.get("answer_grounding", 0.0),
            "confidence":       r.get("confidence", 0.0),
            "actual_answer":    r.get("actual_answer", ""),
            "tests":            r.get("tests", ""),
        })

    return {
        "run_number": run_num,
        "pass":       result.get("pass", 0),
        "partial":    result.get("partial", 0),
        "fail":       result.get("fail", 0),
        "pass_rate":  float(str(result.get("pass_rate", "0")).replace("%", "")),
        "time_sec":   elapsed,
        "questions":  questions_detail,
    }


def compute_stability_report(all_run_records: list,
                              dataset_meta: dict,
                              run_times: list,
                              total_runs: int) -> dict:
    q_verdicts  = defaultdict(list)
    q_questions = {}
    q_halluc    = {}

    # NEW: per-question score trackers
    q_answer_scores    = defaultdict(list)
    q_retrieval_scores = defaultdict(list)
    q_followup_scores  = defaultdict(list)
    q_numeric_scores = defaultdict(list)
    q_rewrite_triggered = defaultdict(list)
    q_halluc_detected = defaultdict(list)

    for run_record in all_run_records:
        for q in run_record["questions"]:
            if q.get("conversation_id") is not None:
                qid = f"{q['conversation_id']}_{q['turn']}"
            else:
                qid = str(q["id"])

            q_verdicts[qid].append(q["verdict"])
            q_questions[qid] = q["question"]

            base_id = q.get("id")
            if base_id in dataset_meta:
                q_halluc[qid] = dataset_meta[base_id]["hallucination_test"]
            else:
                # CHANGE: use hallucination_detected instead of hallucination
                q_halluc[qid] = q.get("hallucination_detected", False)

            # NEW: collect score arrays
            q_answer_scores[qid].append(q.get("answer_score", 0.0))
            q_retrieval_scores[qid].append(q.get("retrieval_recall", 0.0))
            if q.get("followup_score") is not None:
                q_followup_scores[qid].append(q.get("followup_score", 0.0))
            q_numeric_scores[qid].append(
                q.get("pass_numeric", 0)
            )
            q_rewrite_triggered[qid].append(
                q.get("rewrite_triggered", False)
            )
            q_halluc_detected[qid].append(
                q.get("hallucination_detected", False)
            )

    total_questions = len(q_verdicts)
    total_answers   = total_questions * total_runs
    total_pass      = sum(v.count("PASS")    for v in q_verdicts.values())
    total_partial   = sum(v.count("PARTIAL") for v in q_verdicts.values())
    total_fail      = sum(v.count("FAIL")    for v in q_verdicts.values())

    overall_accuracy = round(
        total_pass / max(total_answers, 1) * 100,
        1
    )

    partial_rate = round(
        total_partial / max(total_answers, 1) * 100,
        1
    )

    fail_rate = round(
        total_fail / max(total_answers, 1) * 100,
        1
    )

    q_stable         = {qid: len(set(v)) == 1 for qid, v in q_verdicts.items()}
    unstable_qs      = sorted([qid for qid, s in q_stable.items() if not s])
    instability_rate = round(
        len(unstable_qs) / max(total_questions, 1) * 100,
        1
    )

    halluc_q_ids = [qid for qid, is_h in q_halluc.items() if is_h]

    if halluc_q_ids:

        halluc_count = sum(
            sum(q_halluc_detected[qid])
            for qid in halluc_q_ids
        )

        halluc_total = len(halluc_q_ids) * total_runs

        halluc_rate = round(
            halluc_count / halluc_total * 100,
            1
        )

    else:
        halluc_rate = 0.0

    avg_run_time = round(sum(run_times) / len(run_times), 1)

    per_question = []
    for qid in sorted(q_verdicts.keys()):
        verdicts   = q_verdicts[qid]
        pass_count = verdicts.count("PASS")
        part_count = verdicts.count("PARTIAL")
        fail_count = verdicts.count("FAIL")
        pass_pct = round(
            pass_count / max(len(verdicts), 1) * 100,
            1
        )
        per_question.append({
            "id":            qid,
            "question":      q_questions[qid],
            "pass_count":    pass_count,
            "partial_count": part_count,
            "fail_count":    fail_count,
            "pass_pct":      pass_pct,
            "stable":        q_stable[qid],
            "halluc_test":   q_halluc.get(qid, False),
            # NEW: per-question avg scores
            "avg_answer_score": round(
                sum(q_answer_scores[qid]) /
                max(len(q_answer_scores[qid]), 1),
                4
            ),

            "avg_retrieval_recall": round(
                sum(q_retrieval_scores[qid]) /
                max(len(q_retrieval_scores[qid]), 1),
                4
            ),

            "avg_followup_score": round(
                sum(q_followup_scores[qid]) /
                max(len(q_followup_scores[qid]), 1),
                4
            ) if q_followup_scores[qid] else None,

            "avg_pass_numeric": round(
                sum(q_numeric_scores[qid]) /
                max(len(q_numeric_scores[qid]), 1),
                4
            ),

            "rewrite_trigger_rate": round(
                sum(q_rewrite_triggered[qid]) /
                max(len(q_rewrite_triggered[qid]), 1) * 100,
                1
            ),
            })

    return {
        "total_questions":    total_questions,
        "total_answers":      total_answers,
        "overall_accuracy":   overall_accuracy,
        "partial_rate":       partial_rate,
        "fail_rate":          fail_rate,
        "instability_rate":   instability_rate,
        "unstable_q_ids":     unstable_qs,
        "hallucination_rate": halluc_rate,
        "avg_run_time_sec":   avg_run_time,
        # NEW: global averages
        "avg_answer_score": round(
            sum(
                sum(v) / max(len(v), 1)
                for v in q_answer_scores.values()
            ) / max(total_questions, 1),
            4
        ),
        "avg_retrieval_recall": round(
            sum(
                sum(v) / max(len(v), 1)
                for v in q_retrieval_scores.values()
           ) / max(total_questions, 1),
            4
        ),
        "per_question":       per_question,
    }


def print_stability_report(report: dict, total_runs: int, session_name: str):
    total_questions = report["total_questions"]
    total_answers   = report["total_answers"]
    per_question    = report["per_question"]

    print(f"\n{'='*60}")
    print(f"  STABILITY REPORT  ({total_runs} runs × {total_questions} questions)")
    print(f"{'='*60}")
    print(f"\n{'Q':>3}  {'Verdict Distribution':<30}  {'Pass%':>6}  {'Stable':>9}  Question")
    print(f"{'─'*3}  {'─'*30}  {'─'*6}  {'─'*9}  {'─'*40}")

    for pq in per_question:
        dist       = f"P={pq['pass_count']} Pr={pq['partial_count']} F={pq['fail_count']}"
        stable_str = "✅ STABLE" if pq["stable"] else "⚠️ UNSTABLE"
        q_short    = pq["question"][:45]
        print(f"{pq['id']:>3}  {dist:<30}  {pq['pass_pct']:>5.0f}%  {stable_str:>11}  {q_short}")

    total_pass    = sum(pq["pass_count"]    for pq in per_question)
    total_partial = sum(pq["partial_count"] for pq in per_question)
    total_fail    = sum(pq["fail_count"]    for pq in per_question)
    unstable_qs   = report["unstable_q_ids"]

    print(f"\n{'='*60}")
    print(f"  SUMMARY METRICS")
    print(f"{'='*60}")
    print(f"  Session              : {session_name}")
    print(f"  Runs completed       : {total_runs} / {total_runs}")
    print(f"  Questions per run    : {total_questions}")
    print(f"  Total answers scored : {total_answers}")
    print(f"")
    print(f"  ✅ Overall Accuracy  : {report['overall_accuracy']}%  ({total_pass}/{total_answers} PASS)")
    print(f"  ⚠️  Partial rate      : {report['partial_rate']}%  ({total_partial}/{total_answers} PARTIAL)")
    print(f"  ❌ Fail rate         : {report['fail_rate']}%  ({total_fail}/{total_answers} FAIL)")
    print(f"")
    # NEW: avg score lines
    print(f"  🧠 Avg Answer Score : {report['avg_answer_score']}")
    print(f"  📚 Avg Retrieval    : {report['avg_retrieval_recall']}")
    print(f"")
    print(f"  🔁 Instability rate  : {report['instability_rate']}%  "
          f"({len(unstable_qs)}/{total_questions} unstable questions)")
    if unstable_qs:
        print(f"     Unstable Q IDs   : {unstable_qs}")
    print(f"")
    print(f"  🚨 Hallucination rate: {report['hallucination_rate']}%  "
          f"(hallucination tests only)")
    print(f"")
    print(f"  ⏱️  Avg run time      : {report['avg_run_time_sec']}s")
    print(f"{'='*60}")


def save_session(session: dict, existing: dict):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    # existing["sessions"].append(session)
    replaced = False

    for i, s in enumerate(existing["sessions"]):

        if s.get("name") == session["name"]:

            existing["sessions"][i] = session
            replaced = True
            break

    if not replaced:
        existing["sessions"].append(session)
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  Results saved → {RESULTS_FILE}")
        print(f"  Total sessions stored: {len(existing['sessions'])}")
    except Exception as e:
        print(f"[ERROR] Could not save results: {e}")


def main():
    print(f"\n=== RUN START: {datetime.datetime.now()} ===\n")
    existing     = load_existing_results()
    next_num     = get_next_session_number(existing)
    session_name = get_session_name(next_num)
    # ============================================================
    # Reserve session immediately
    # so interrupted runs don't reuse same Test number
    # ============================================================

    placeholder_session = {
        "name": session_name,
        "reserved": True,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    existing["sessions"].append(placeholder_session)

    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    for pdf_path in PDF_PATHS:
        print("\n" + "="*80)
        print(f"STARTING PDF: {os.path.basename(pdf_path)}")
        print("="*80 + "\n")

        try:
            dataset_meta = load_dataset_meta(pdf_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] Skipping {os.path.basename(pdf_path)}: {e}")
            continue

        print(f"\n{'='*60}")
        print(f"  DocMind Stability Evaluation")
        print(f"  Session : {session_name}")
        print(f"  PDF     : {os.path.basename(pdf_path)}")
        print(f"  Runs    : {NUM_RUNS}")
        print(f"{'='*60}")

        skipped  = [qid for qid, m in dataset_meta.items() if m.get("skip")]
        # skip_ids = set(skipped)
        if skipped:
            print(f"  ⏭️  Skipping Q IDs  : {skipped} (skip=true in dataset)")

        all_run_records = []
        run_times       = []

        for i in range(1, NUM_RUNS + 1):
            start   = time.time()
            result  = run_evaluate(pdf_path, i, session_name)
            elapsed = round(time.time() - start, 1)
            run_times.append(elapsed)

            if result:
                run_record = build_run_record(i, result, elapsed)
                all_run_records.append(run_record)
                print(f"\n[Run {i}] Pass={run_record['pass']} "
                      f"Partial={run_record['partial']} "
                      f"Fail={run_record['fail']} "
                      f"Rate={run_record['pass_rate']}% "
                      f"Time={elapsed}s")
            else:
                print(f"[Run {i}] FAILED — skipping")

        if not all_run_records:
            print("No successful runs. Skipping this PDF.")
            continue

        actual_runs = len(all_run_records)
        report = compute_stability_report(
            all_run_records, dataset_meta, run_times, actual_runs
        )
        print_stability_report(report, actual_runs, session_name)

        session = {
            "name":             session_name,
            "date":             time.strftime("%Y-%m-%d %H:%M:%S"),
            "pdf":              os.path.basename(pdf_path),
            "num_runs":         actual_runs,
            "runs":             all_run_records,
            "stability_report": report,
        }
        save_session(session, existing)


if __name__ == "__main__":
    try:
        main()
    finally:
        _log_file.close()



















