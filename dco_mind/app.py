# import os
# _f = open(r"C:\xampp\htdocs\GenAI-Doc-old\dco_mind\evaluation\run_logs_backend.txt", "a", encoding="utf-8")
# _f.write("=== APP STARTED ===\n")
# _f.flush()
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOGGING SETUP — using logging module, not sys.stdout patch
# ============================================================
import sys, os, logging

# _LOG_DIR = r"C:\xampp\htdocs\GenAI-Doc-old\dco_mind\evaluation"
# os.makedirs(_LOG_DIR, exist_ok=True)
# _LOG_PATH = os.path.join(_LOG_DIR, "run_logs_backend.txt")

# # Root logger captures everything
# _logger = logging.getLogger()
# _logger.setLevel(logging.DEBUG)

# # File handler
# _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8", mode="a")
# _fh.setLevel(logging.DEBUG)
# _fh.setFormatter(logging.Formatter("%(message)s"))

# # Console handler (keeps terminal output)
# _ch = logging.StreamHandler(sys.__stdout__)
# _ch.setLevel(logging.DEBUG)
# _ch.setFormatter(logging.Formatter("%(message)s"))

# _logger.addHandler(_fh)
# _logger.addHandler(_ch)

# # Redirect print() → logging
# class _PrintToLogger:
#     def __init__(self, level=logging.INFO):
#         self.level = level
#         self._buf = ""
#     def write(self, msg):
#         self._buf += msg
#         while "\n" in self._buf:
#             line, self._buf = self._buf.split("\n", 1)
#             if line.strip():
#                 logging.log(self.level, line)
#     def flush(self):
#         if self._buf.strip():
#             logging.log(self.level, self._buf)
#             self._buf = ""

# sys.stdout = _PrintToLogger(logging.INFO)
# sys.stderr = _PrintToLogger(logging.ERROR)


from flask import Flask
from flask_cors import CORS
from langgraph.graph import StateGraph, END

# ✅ YOUR STRUCTURE
from dco_mind.core.state import DocState
from dco_mind.core.engine import (
    node_extract,
    node_chunk,
    node_summarize,
    node_qa,
    node_validate
)
from dco_mind.interface.api_routes import register_routes


# ============================================================
# FLASK SETUP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================
def route_query(state: DocState) -> str:
    return "summarize" if state["query_type"] == "FULL_SUMMARY" else "qa"

def route_validate(state: DocState) -> str:
    # 🔥 HARD STOP: never retry if answer exists
    if state.get("answer") and state["answer"].strip():
        return "done"

    # Retry ONLY if truly empty
    if state.get("retry_count", 0) < 1:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "retry"

    return "done"

def build_workflow():
    workflow = StateGraph(DocState)
    workflow.add_node("extract",   node_extract)
    workflow.add_node("chunk",     node_chunk)
    workflow.add_node("summarize", node_summarize)
    workflow.add_node("qa",        node_qa)
    workflow.add_node("validate",  node_validate)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "chunk")
    workflow.add_conditional_edges("chunk", route_query,
        {"summarize": "summarize", "qa": "qa"})
    workflow.add_edge("summarize", "validate")
    workflow.add_edge("qa",        "validate")
    workflow.add_conditional_edges("validate", route_validate,
        {"retry": "qa", "done": END})
    return workflow.compile()

print("Building LangGraph workflow...")
workflow = build_workflow()
print("Workflow ready!")

# ============================================================
# REGISTER ROUTES
# ============================================================
register_routes(app, workflow)

if __name__ == "__main__":
    print("Flask server running on http://127.0.0.1:5000")
    # app.run(port=5000, threaded=True)
    app.run(port=5000, threaded=True, debug=False, use_reloader=False)





























# import warnings  gpt issue with writing terminal logs in backend.txt 
# warnings.filterwarnings("ignore")

# from flask import Flask
# from flask_cors import CORS
# from langgraph.graph import StateGraph, END

# # ✅ YOUR STRUCTURE
# from dco_mind.core.state import DocState
# from dco_mind.core.engine import (
#     node_extract,
#     node_chunk,
#     node_summarize,
#     node_qa,
#     node_validate
# )
# from dco_mind.interface.api_routes import register_routes


# # ============================================================
# # FLASK SETUP
# # ============================================================
# app = Flask(__name__)
# CORS(app)

# # ============================================================
# # LANGGRAPH WORKFLOW
# # ============================================================
# def route_query(state: DocState) -> str:
#     return "summarize" if state["query_type"] == "FULL_SUMMARY" else "qa"

# def route_validate(state: DocState) -> str:
#     # 🔥 HARD STOP: never retry if answer exists
#     if state.get("answer") and state["answer"].strip():
#         return "done"

#     # Retry ONLY if truly empty
#     if state.get("retry_count", 0) < 1:
#         state["retry_count"] = state.get("retry_count", 0) + 1
#         return "retry"

#     return "done"

# def build_workflow():
#     workflow = StateGraph(DocState)
#     workflow.add_node("extract",   node_extract)
#     workflow.add_node("chunk",     node_chunk)
#     workflow.add_node("summarize", node_summarize)
#     workflow.add_node("qa",        node_qa)
#     workflow.add_node("validate",  node_validate)
#     workflow.set_entry_point("extract")
#     workflow.add_edge("extract", "chunk")
#     workflow.add_conditional_edges("chunk", route_query,
#         {"summarize": "summarize", "qa": "qa"})
#     workflow.add_edge("summarize", "validate")
#     workflow.add_edge("qa",        "validate")
#     workflow.add_conditional_edges("validate", route_validate,
#         {"retry": "qa", "done": END})
#     return workflow.compile()

# print("Building LangGraph workflow...")
# workflow = build_workflow()
# print("Workflow ready!")

# # ============================================================
# # REGISTER ROUTES
# # ============================================================
# register_routes(app, workflow)

# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000, threaded=True)






















# #without memory.py code html code
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>DocMind — Agentic Document Intelligence</title>

# <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">

# <!-- ✅ External CSS -->
# <link rel="stylesheet" href="css/style.css">
# </head>

# <body>

# <div class="container">

#   <div class="header">
#     <h1>DocMind</h1>
#     <p>Agentic Document Intelligence System</p>
#     <div class="badge">LangGraph · ReAct · FAISS · RAPTOR · RoBERTa · LLaMA 3.2</div>
#   </div>

#   <div class="card">
#     <div class="card-title">01 — Upload Document</div>
#     <div class="upload-area" id="uploadArea">
#       <input type="file" id="pdfUpload" accept="application/pdf">
#       <div class="upload-icon">📄</div>
#       <div class="upload-text">Drop PDF here or <span>browse</span></div>
#       <div class="small-note">
#         Supports digital + scanned PDFs · Any size
#       </div>
#     </div>
#     <div class="upload-status" id="uploadStatus"></div>
#   </div>

#   <div class="card">
#     <div class="card-title">02 — Ask Question or Request Summary</div>
#     <div class="input-row">
#       <input type="text" class="question-input" id="questionInput"
#         placeholder="e.g. 'give summary' or 'list all ethical theories' or 'why is AI dangerous?'">
#       <button class="submit-btn" id="submitBtn">
#         Run Agent →
#       </button>
#     </div>
#   </div>

#   <div class="card" id="agentPanel">
#     <div class="agent-header">
#       <div class="agent-spinner" id="agentSpinner"></div>
#       <div class="agent-title">AGENT REASONING TRACE</div>
#     </div>
#     <div class="timeline" id="timeline"></div>
#   </div>

#   <div class="card" id="answerPanel">
#     <div class="card-title">Answer</div>
#     <div class="answer-box" id="answerBox"></div>
#   </div>

#   <div class="card" id="metricsPanel">
#     <div class="card-title">Benchmark Metrics</div>
#     <div class="metrics-grid" id="metricsGrid"></div>
#   </div>

# </div>

# <!-- ✅ External JS -->
# <script src="js/app.js"></script>

# </body>
# </html>




















































# (venv_dco) PS C:\xampp\htdocs\GenAI-Doc-old> tree "C:\xampp\htdocs\GenAI-Doc-old\dco_mind" /F /A
# Folder PATH listing for volume OS
# Volume serial number is 0C10-0E14
# C:\XAMPP\HTDOCS\GENAI-DOC-OLD\DCO_MIND
# |   app.py
# |   grounding_results.json
# |   requirements.txt
# |   
# +---cognition
# |   |   intent_model.py
# |   |   memory.py
# |   |   query_brain.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           memory.cpython-311.pyc
# |           query_brain.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---config
# |   |   settings.py
# |   |   
# |   \---__pycache__
# |           settings.cpython-311.pyc
# |           
# +---core
# |   |   engine.py
# |   |   state.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           engine.cpython-311.pyc
# |           state.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---datasets
# |       ml.json
# |       resume.json
# |       story.json
# |       
# +---evaluation
# |   |   metrics.py
# |   |   question_gen.py
# |   |   run_batch.py
# |   |   run_logs_backend.txt
# |   |   run_logs_runner.txt
# |   |   stability_runner.py
# |   |   __init__.py
# |   |   
# |   +---results
# |   |       stability_results.json
# |   |       
# |   \---__pycache__
# |           metrics.cpython-311.pyc
# |           question_gen.cpython-311.pyc
# |           run_batch.cpython-311.pyc
# |           stability_runner.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---events
# |   |   events.py
# |   |   
# |   \---__pycache__
# |           events.cpython-311.pyc
# |           
# +---generation
# |   |   response_generator.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           response_generator.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---interface
# |   |   api_routes.py
# |   |   session_manager.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           api_routes.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---knowledge
# |   |   indexing.py
# |   |   ingestion.py
# |   |   knowledge_store.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           ingestion.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---models
# |   |   embeddings.py
# |   |   llm.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           embeddings.cpython-311.pyc
# |           llm.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---reasoning
# |   |   context_builder.py
# |   |   evidence_linker.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           context_builder.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---retrieval
# |   |   adaptive_search.py
# |   |   query_expansion.py
# |   |   reranker.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           adaptive_search.cpython-311.pyc
# |           reranker.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# +---utils
# |   |   helpers.py
# |   |   __init__.py
# |   |   
# |   \---__pycache__
# |           helpers.cpython-311.pyc
# |           __init__.cpython-311.pyc
# |           
# \---__pycache__
#         app.cpython-311.pyc
        
# (venv_dco) PS C:\xampp\htdocs\GenAI-Doc-old> 











# dco_mind/
# │
# ├── app.py
# ├── requirements.txt
# │
# ├── cognition/                 🧠 (NEW IDENTITY)
# │   ├── memory.py              # session memory
# │   ├── query_brain.py         # classify + rewrite
# │   └── intent_model.py        # optional upgrade later
# │
# ├── knowledge/                 📚 (NOT "services")
# │   ├── ingestion.py           # extraction + chunking merged
# │   ├── indexing.py
# │   └── knowledge_store.py
# │
# ├── retrieval/                 ⚙️ (BUT smarter)
# │   ├── adaptive_search.py     # replaces retrieval.py
# │   ├── query_expansion.py
# │   └── reranker.py
# │
# ├── reasoning/                 🧬 (NEW layer)
# │   ├── context_builder.py
# │   └── evidence_linker.py     # multi-hop ready
# │
# ├── generation/
# │   ├── response_generator.py
# │
# ├── core/
# │   ├── engine.py              🚀 (replaces qa_pipeline)
# │   └── state.py
# │
# ├── interface/
# │   ├── api_routes.py          # instead of api/routes.py
# │   └── session_manager.py
# │
# ├── models/                    (you can reuse this mostly)
# │   ├── llm.py
# │   ├── embeddings.py
# │
# ├── utils/
# ├── cache/
# └── data/

# │
# ├── evaluation/               📊 (NEW — VERY IMPORTANT)
# │   ├── evaluator.py
# │   ├── metrics.py
# |   |- question_gen.py
# |   |-run_batch.py
# │   └── datasets/



# # story.json
# {
#   "questions": [

#     {
#       "id": 1,
#       "question": "Why did people stop visiting the doctor?",
#       "expected_keywords": ["animals", "house", "untidy"],
#       "pass_threshold": 1
#     },

#     {
#       "id": 2,
#       "conversation": [
#         "Who is the main character in the story?",
#         "Where does he live?",
#         "Who lives with him?",
#         "What problem does she complain about?"
#       ],
#       "expected_keywords": ["Dolittle", "Puddleby", "sister", "animals"],
#       "pass_threshold": 1
#     },
#     {
  
#       "id": 3,
#       "question": "How did the doctor learn that animals can communicate?",
#       "expected_keywords": ["parrot", "Polynesia", "language", "teach"],
#       "pass_threshold": 1
#     },

#     {
#       "id": 4,
#       "conversation": [
#         "Why did the doctor stop treating people?",
#         "Who helped him understand animal language?",
#         "What did he learn from her?",
#         "How did this change his career?"
#       ],
#       "expected_keywords": ["Polynesia", "animals", "language", "animal doctor"],
#       "pass_threshold": 1
#     },


#     {
#       "id": 5,
#       "question": "How did the doctor become rich after returning from his journey?",
#       "expected_keywords": ["pushmi", "pullyu", "fair", "show", "sixpence"],
#       "pass_threshold": 1
#     },

#     {
#       "id": 6,
#       "conversation": [
#         "What did the doctor do before returning home?",
#         "Why did he stop that activity?",
#         "What did he do after coming back to Puddleby?",
#         "How did his life change compared to before?"
#       ],
#       "expected_keywords": ["show", "money", "home", "rich", "animals"],
#       "pass_threshold": 1
#     },
#     {
 
#       "id": 7,
#       "question": "Why was the fisherman left alone on the rock?",
#       "expected_keywords": ["refused", "pirate", "Barbary", "Dragon"],
#       "pass_threshold": 1
#     },

#     {
#       "id": 8,
#       "conversation": [
#         "Why was the fisherman in trouble initially?",
#         "How did the doctor help him?",
#         "How did the people react when he returned home?",
#         "What reward was given to the doctor and Jip?"
#       ],
#       "expected_keywords": ["rescued", "ship", "cheer", "watch", "gold", "collar"],
#       "pass_threshold": 1
#     },
#     {
 
#       "id": 9,
#       "question": "Why did the eagles fail to find the fisherman?",
#       "expected_keywords": ["could not see", "no sign", "search", "failed"],
#       "pass_threshold": 1
#     },

#     {
#       "id": 10,
#       "conversation": [
#         "What was the initial plan to find the fisherman?",
#         "Why did that plan fail?",
#         "What alternative method was used afterward?",
#         "What clue helped in finding the fisherman?"
#       ],
#       "expected_keywords": ["eagles", "failed", "smell", "snuff", "handkerchief"],
#       "pass_threshold": 1
#     }







#   ]
# }