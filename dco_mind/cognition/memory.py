import time
from collections import defaultdict
from dco_mind.models.llm import call_llama

# ============================================================
# SESSION STORE
# ============================================================
# { session_id: [ {question, answer, timestamp}, ... ] }
_session_store = defaultdict(list)

MAX_HISTORY = 5  # keep last 5 turns per session

import re

def extract_entities(text):
    # simple but scalable: proper nouns
    return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
# ============================================================
# SAVE TO MEMORY
# ============================================================
def save_to_memory(session_id: str, question: str, answer: str):
    if not session_id:
        return
    _session_store[session_id].append({
        "question":  question,
        "answer":    answer,
        "entities": extract_entities(answer), 
        "timestamp": time.time()
    })
    # Keep only last MAX_HISTORY turns
    if len(_session_store[session_id]) > MAX_HISTORY:
        _session_store[session_id] = _session_store[session_id][-MAX_HISTORY:]


# ============================================================
# GET HISTORY
# ============================================================
def get_history(session_id: str) -> list:
    if not session_id:
        return []
    return _session_store.get(session_id, [])


# ============================================================
# CLEAR SESSION
# ============================================================
def clear_session(session_id: str):
    if session_id in _session_store:
        del _session_store[session_id]


# ============================================================
# REWRITE QUERY USING HISTORY
# ============================================================
# In memory.py — replace rewrite_query entirely
# def get_retrieval_query(session_id: str, question: str) -> str:      for test 3 issue 
#     # 🔥 DO NOT MODIFY QUERY FOR RETRIEVAL
#     return question
def get_retrieval_query(session_id: str, question: str) -> str:
    """
    Enriches the retrieval query with the last answer
    so FAISS finds relevant chunks without extra LLM calls.
    """
    if not session_id:
        return question

    history = get_history(session_id)
    if not history:
        return question

    # Take the last answer as context for retrieval
    last_answer = history[-1]["answer"][:200]

    # Combine last answer + current question as retrieval query
    # This gives FAISS enough context to find the right chunks
    enriched = f"{last_answer} {question}"

    print(f"[Memory] Enriched query: '{enriched[:100]}'")
    return enriched
# ============================================================
# BUILD MEMORY CONTEXT FOR PROMPT
# ============================================================
def get_memory_context(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""

    context = "Previous conversation:\n"
    for turn in history[-3:]:
        context += f"Q: {turn['question']}\nA: {turn['answer'][:300]}\n\n"
    return context.strip()
















# #for this changed routes.py js,html nd  changed in app.py threaded =True
#  the one with what i got  problem with rewrite query 
# import time
# from collections import defaultdict
# from dco_mind.models.llm import call_llama

# # ============================================================
# # SESSION STORE
# # ============================================================
# # { session_id: [ {question, answer, timestamp}, ... ] }
# _session_store = defaultdict(list)

# MAX_HISTORY = 5  # keep last 5 turns per session


# # ============================================================
# # SAVE TO MEMORY
# # ============================================================
# def save_to_memory(session_id: str, question: str, answer: str):
#     if not session_id:
#         return
#     _session_store[session_id].append({
#         "question":  question,
#         "answer":    answer,
#         "timestamp": time.time()
#     })
#     # Keep only last MAX_HISTORY turns
#     if len(_session_store[session_id]) > MAX_HISTORY:
#         _session_store[session_id] = _session_store[session_id][-MAX_HISTORY:]


# # ============================================================
# # GET HISTORY
# # ============================================================
# def get_history(session_id: str) -> list:
#     if not session_id:
#         return []
#     return _session_store.get(session_id, [])


# # ============================================================
# # CLEAR SESSION
# # ============================================================
# def clear_session(session_id: str):
#     if session_id in _session_store:
#         del _session_store[session_id]


# # ============================================================
# # REWRITE QUERY USING HISTORY
# # ============================================================
# def rewrite_query(session_id: str, question: str) -> str:
#     history = get_history(session_id)

#     # No history → return original question as is
#     if not history:
#         return question

#     # Build history context string
#     history_text = ""
#     for turn in history[-3:]:  # use last 3 turns only
#         history_text += f"User: {turn['question']}\nAssistant: {turn['answer'][:200]}\n\n"

#     prompt = f"""You are a query rewriter. Given the conversation history and a new question, rewrite the new question to be self-contained and clear.

# Conversation History:
# {history_text}

# New Question: {question}

# Rules:
# - If the question is already clear and self-contained, return it unchanged
# - If the question refers to something from history (like "it", "that", "the same"), rewrite it with full context
# - Return ONLY the rewritten question, nothing else

# Rewritten Question:"""

#     try:
#         rewritten = call_llama(prompt, num_ctx=1024, temperature=0.0).strip()
#         # Safety check — if rewrite is too long or weird, use original
#         if len(rewritten) > 500 or len(rewritten) < 3:
#             return question
#         print(f"[Memory] Query rewritten: '{question}' → '{rewritten}'")
#         return rewritten
#     except Exception:
#         return question


# # ============================================================
# # BUILD MEMORY CONTEXT FOR PROMPT
# # ============================================================
# def get_memory_context(session_id: str) -> str:
#     history = get_history(session_id)
#     if not history:
#         return ""

#     context = "Previous conversation:\n"
#     for turn in history[-3:]:
#         context += f"Q: {turn['question']}\nA: {turn['answer'][:300]}\n\n"
#     return context.strip()
