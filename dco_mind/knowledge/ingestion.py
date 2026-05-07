#chunking.py 



import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import pytesseract
from PIL import Image


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dco_mind.generation.response_generator import SECTION_SUMMARY_PROMPT, MERGE_PROMPT

from dco_mind.config.settings import (
    SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE,
    CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
)

from dco_mind.models.llm import call_llama, call_llama_streaming
from dco_mind.events.events import emit_event


_summary_cache: dict = {}

# ============================================================
# CORE: TF-IDF EXTRACTIVE SUMMARY
# ============================================================
def extractive_summary(chunk: str, top_n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    sentences = [s for s in sentences
                 if not re.match(r'^\[\d+\]', s)
                 and not re.match(r'^\d+\.\s+[A-Z]', s)
                 and s.count('[') < 3]
    if not sentences:
        return chunk[:1000]
    if len(sentences) <= top_n:
        return " ".join(sentences)
    try:
        vectorizer   = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
        return " ".join(sentences[i] for i in top_indices)
    except Exception:
        return " ".join(sentences[:top_n])


# ============================================================
# DUAL CHUNKING
# ============================================================
def merge_short_lines(text: str) -> str:
    lines = text.split("\n")
    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # detect short header-like lines
        if (
            0 < len(line) < 80
            and not line.endswith(".")
            and i + 1 < len(lines)
        ):
            next_line = lines[i + 1].strip()

            # merge if next line is meaningful
            if next_line and len(next_line) > 20:
                merged.append(f"{line} — {next_line}")
                i += 2
                continue

        merged.append(line)
        i += 1

    return "\n".join(merged)
def semantic_chunk(text: str):
    # Summary chunks — simple version (fast, works for summary)
    clean_text = re.sub(r'--- PAGE \d+ ---\n?', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    summary_chunks = summary_splitter.split_text(clean_text)

    # RAG chunks — page-aware version (works for QA)
    text_for_rag = merge_short_lines(text)
    rag_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_pages = [p.strip() for p in text_for_rag.split("--- PAGE") if p.strip()]
    page_texts = []
    for page in raw_pages:
        lines = page.split("\n", 1)
        page_text = lines[1].strip() if len(lines) > 1 else lines[0].strip()
        if page_text:
            page_texts.append(page_text)

    MIN_PAGE_CHARS = 200
    merged_pages = []
    i = 0
    while i < len(page_texts):
        current = page_texts[i]
        if len(current) < MIN_PAGE_CHARS and i + 1 < len(page_texts):
            merged_pages.append(current + "\n\n" + page_texts[i + 1])
            i += 2
        else:
            merged_pages.append(current)
            i += 1

    rag_chunks = []
    for page_text in merged_pages:
        rag_chunks.extend(rag_splitter.split_text(page_text))

    if not rag_chunks:
        rag_chunks = rag_splitter.split_text(clean_text)

    print(f"[Chunk] Summary chunks: {len(summary_chunks)} | RAG chunks: {len(rag_chunks)}")
    return summary_chunks, rag_chunks
# ============================================================
# CORE: RAPTOR HIERARCHICAL SUMMARY
# ============================================================
def raptor_summarize(chunks: list, doc_type: str):
    if not chunks:
        return "No content to summarize.", 0, 0

    map_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
        mini_summaries = [f.result() for f in futures if f.result().strip()]
    map_time = time.time() - map_start
    print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

    if not mini_summaries:
        return "Could not generate summary.", map_time, 0

    reduce_start  = time.time()
    BATCH_CHARS = RAPTOR_BATCH_CHARS
    batches       = []
    current_batch = ""
    for summary in mini_summaries:
        if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
            batches.append(current_batch.strip())
            current_batch = summary + "\n\n"
        else:
            current_batch += summary + "\n\n"
    if current_batch.strip():
        batches.append(current_batch.strip())

    request_id = getattr(raptor_summarize, '_request_id', "")

    partial_summaries = []
    print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

    if len(batches) == 1:
        emit_event(request_id, "stream_start", "✍️ Generating summary...")
        final_summary, _ = call_llama_streaming(
            SECTION_SUMMARY_PROMPT.format(text=batches[0]),
            request_id, temperature=0.7)
        partial_summaries = [final_summary]
    else:
        for i, batch in enumerate(batches):
            prompt = SECTION_SUMMARY_PROMPT.format(text=batch)
            result = call_llama(prompt, temperature=0.7)
            partial_summaries.append(result)
            print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
            emit_event(request_id, "agent_action",
                       f"📝 Processed section {i+1}/{len(batches)}...")

        merged = "\n\n---\n\n".join(partial_summaries)
        emit_event(request_id, "stream_start", "✍️ Generating final summary...")
        final_summary, _ = call_llama_streaming(
            MERGE_PROMPT.format(summaries=merged[:12000]),
            request_id, temperature=0.7)

    reduce_time = time.time() - reduce_start
    total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
    print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
    return final_summary, map_time, reduce_time










# extraction.py




from dco_mind.config.settings import MAX_WORKERS
from dco_mind.utils.helpers import get_pdf_hash
_extraction_cache: dict = {}

# ============================================================
# CORE: EXTRACT SINGLE PAGE
# ============================================================
def extract_page(args):
    page, page_num = args
    try:
        text = page.get_text().strip()
        if len(text) > 10:
            return page_num, text

        if page.rect.width < 10 or page.rect.height < 10:
            return page_num, ""

        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            t = pytesseract.image_to_string(img).strip()
            real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
            if real_words > 5:
                return page_num, t
        except Exception:
            pass

        best_text = ""
        best_len  = 0
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle, expand=True)
            try:
                t = pytesseract.image_to_string(rotated).strip()
                real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
                if real_words > best_len:
                    best_len  = real_words
                    best_text = t
            except Exception:
                continue

        return page_num, best_text.strip()
    except Exception:
        return page_num, ""


def extract_pdf_parallel(pdf_path: str):
    pdf_hash = get_pdf_hash(pdf_path)
    if pdf_hash in _extraction_cache:
        text, page_count = _extraction_cache[pdf_hash]
        print(f"[Extract] ✅ Cache hit — skipping OCR ({page_count} pages)")
        return text, page_count

    doc        = fitz.open(pdf_path)
    page_count = len(doc)
    pages      = [(doc[i], i) for i in range(page_count)]
    results    = {}
    workers    = min(MAX_WORKERS, page_count)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_page, p): p[1] for p in pages}
        for future in as_completed(futures):
            page_num, text    = future.result()
            results[page_num] = text
    doc.close()

    page_texts = []
    for i in sorted(results.keys()):
        t = results[i].strip()
        if t:
            t = re.sub(r'[^\x00-\x7F]+', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            page_texts.append(f"--- PAGE {i} ---\n{t}")

    full_text = "\n\n".join(page_texts)
    _extraction_cache[pdf_hash] = (full_text, page_count)
    print(f"[Extract] Done — {page_count} pages, {len(full_text)} chars")
    return full_text, page_count
