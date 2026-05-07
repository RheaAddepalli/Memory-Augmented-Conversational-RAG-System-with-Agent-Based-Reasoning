
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOGGING SETUP — using logging module, not sys.stdout patch
# ============================================================
import sys, os, logging



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























