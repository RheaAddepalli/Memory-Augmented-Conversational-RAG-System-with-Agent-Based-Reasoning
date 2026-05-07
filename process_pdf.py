import sys
import os
import json
import hashlib
import concurrent.futures
import logging
import re
import time
from metrics import compute_metrics
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    pipeline
)

# --------------------------------------------------
# BASIC SAFE CONFIG (XAMPP / Apache friendly)
# --------------------------------------------------
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# OCR CONFIG
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --------------------------------------------------
# MODEL LOAD (ONCE)
# --------------------------------------------------
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# --------------------------------------------------
# NORMAL TEXT EXTRACTION
# --------------------------------------------------
def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

# --------------------------------------------------
# OCR FALLBACK
# --------------------------------------------------
def ocr_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    except Exception:
        pass
    return text.strip()

# --------------------------------------------------
# CLEAN OCR + NOISE
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9.,;:()\n ]+', ' ', text)
    lines = text.splitlines()
    clean_lines = [l for l in lines if len(l.strip()) > 30]
    text = " ".join(clean_lines)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --------------------------------------------------
# CHUNKING
# --------------------------------------------------
def split_text(text, max_words=500):
    words = text.split()
    chunks, chunk = [], []

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_words:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return [c[:4000] for c in chunks]

# --------------------------------------------------
# ✅ FIX 2 APPLIED HERE (NO PROMPT)
# --------------------------------------------------
def summarize_chunk(chunk):
    try:
        result = summarizer(
            chunk,
            max_length=160,
            min_length=80,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception:
        return ""

def summarize_text(text):
    chunks = split_text(text)
    summaries = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for s in executor.map(summarize_chunk, chunks):
            if s:
                summaries.append(s)

    return " ".join(summaries).strip()

# --------------------------------------------------
# CACHE
# --------------------------------------------------
def get_pdf_hash(pdf_path):
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# def load_cached_summary(pdf_hash):
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     if os.path.exists(path):
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f).get("summary")
#     return None

# def save_summary(pdf_hash, summary):
#     os.makedirs("saved_summaries", exist_ok=True)
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

# # --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"summary": "", "error": "No PDF provided"}))
            sys.exit(0)

        pdf_path = sys.argv[1]
        start_total = time.time()

        if not os.path.exists(pdf_path):
            print(json.dumps({"summary": "", "error": "PDF not found"}))
            sys.exit(0)

        pdf_hash = get_pdf_hash(pdf_path)

        # cached = load_cached_summary(pdf_hash)
        # if cached:
        #     print(json.dumps({"summary": cached, "cached": True}, ensure_ascii=False))
        #     sys.exit(0)

        start_extract = time.time()

        text = extract_text(pdf_path)

        if len(text) < 100:
            text = ocr_pdf(pdf_path)

        extract_time = time.time() - start_extract

       

        text = clean_text(text)
        chunks = split_text(text)
        num_chunks = len(chunks)

        if len(text) < 100:
            print(json.dumps({"summary": "", "error": "No readable text"}))
            sys.exit(0)

        start_summary = time.time()

        summary = summarize_text(text)

        summary_time = time.time() - start_summary
        total_time = time.time() - start_total
        # save_summary(pdf_hash, summary)
        map_time = summary_time
        reduce_time = 0
        metrics = compute_metrics(
            text,
            summary,
            num_chunks,
            {
                "total": total_time,
                "extraction": extract_time,
                "summary": summary_time,
                "map": map_time,
                "reduce": reduce_time
            }
        )
        print(json.dumps({
    "summary": summary,
    "metrics": metrics
}, ensure_ascii=False))
        sys.stdout.flush()

    except Exception as e:
        print(json.dumps({"summary": "", "error": str(e)}))
        sys.stdout.flush()



























# no chnages just removed chaching now gonna add metrics to it 
# import sys
# import os
# import json
# import hashlib
# import concurrent.futures
# import logging
# import re

# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image

# from transformers import (
#     BartTokenizer,
#     BartForConditionalGeneration,
#     pipeline
# )

# # --------------------------------------------------
# # BASIC SAFE CONFIG (XAMPP / Apache friendly)
# # --------------------------------------------------
# logging.getLogger("transformers").setLevel(logging.ERROR)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.stdout.reconfigure(encoding="utf-8")

# # --------------------------------------------------
# # OCR CONFIG
# # --------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# # --------------------------------------------------
# # MODEL LOAD (ONCE)
# # --------------------------------------------------
# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
# model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# summarizer = pipeline(
#     "summarization",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1
# )

# # --------------------------------------------------
# # NORMAL TEXT EXTRACTION
# # --------------------------------------------------
# def extract_text(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text("text") + "\n"
#     return text.strip()

# # --------------------------------------------------
# # OCR FALLBACK
# # --------------------------------------------------
# def ocr_pdf(pdf_path):
#     text = ""
#     try:
#         doc = fitz.open(pdf_path)
#         for page in doc:
#             pix = page.get_pixmap(dpi=200)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             text += pytesseract.image_to_string(img)
#     except Exception:
#         pass
#     return text.strip()

# # --------------------------------------------------
# # CLEAN OCR + NOISE
# # --------------------------------------------------
# def clean_text(text):
#     text = re.sub(r'[^A-Za-z0-9.,;:()\n ]+', ' ', text)
#     lines = text.splitlines()
#     clean_lines = [l for l in lines if len(l.strip()) > 30]
#     text = " ".join(clean_lines)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# # --------------------------------------------------
# # CHUNKING
# # --------------------------------------------------
# def split_text(text, max_words=500):
#     words = text.split()
#     chunks, chunk = [], []

#     for w in words:
#         chunk.append(w)
#         if len(chunk) >= max_words:
#             chunks.append(" ".join(chunk))
#             chunk = []

#     if chunk:
#         chunks.append(" ".join(chunk))

#     return [c[:4000] for c in chunks]

# # --------------------------------------------------
# # ✅ FIX 2 APPLIED HERE (NO PROMPT)
# # --------------------------------------------------
# def summarize_chunk(chunk):
#     try:
#         result = summarizer(
#             chunk,
#             max_length=160,
#             min_length=80,
#             do_sample=False
#         )
#         return result[0]["summary_text"]
#     except Exception:
#         return ""

# def summarize_text(text):
#     chunks = split_text(text)
#     summaries = []

#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         for s in executor.map(summarize_chunk, chunks):
#             if s:
#                 summaries.append(s)

#     return " ".join(summaries).strip()

# # --------------------------------------------------
# # CACHE
# # --------------------------------------------------
# def get_pdf_hash(pdf_path):
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# # def load_cached_summary(pdf_hash):
# #     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
# #     if os.path.exists(path):
# #         with open(path, "r", encoding="utf-8") as f:
# #             return json.load(f).get("summary")
# #     return None

# # def save_summary(pdf_hash, summary):
# #     os.makedirs("saved_summaries", exist_ok=True)
# #     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
# #     with open(path, "w", encoding="utf-8") as f:
# #         json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

# # # --------------------------------------------------
# # MAIN
# # --------------------------------------------------
# if __name__ == "__main__":
#     try:
#         if len(sys.argv) < 2:
#             print(json.dumps({"summary": "", "error": "No PDF provided"}))
#             sys.exit(0)

#         pdf_path = sys.argv[1]

#         if not os.path.exists(pdf_path):
#             print(json.dumps({"summary": "", "error": "PDF not found"}))
#             sys.exit(0)

#         pdf_hash = get_pdf_hash(pdf_path)

#         # cached = load_cached_summary(pdf_hash)
#         # if cached:
#         #     print(json.dumps({"summary": cached, "cached": True}, ensure_ascii=False))
#         #     sys.exit(0)

#         text = extract_text(pdf_path)

#         if len(text) < 100:
#             text = ocr_pdf(pdf_path)

#         text = clean_text(text)

#         if len(text) < 100:
#             print(json.dumps({"summary": "", "error": "No readable text"}))
#             sys.exit(0)

#         summary = summarize_text(text)

#         # save_summary(pdf_hash, summary)

#         print(json.dumps({"summary": summary, "cached": False}, ensure_ascii=False))
#         sys.stdout.flush()

#     except Exception as e:
#         print(json.dumps({"summary": "", "error": str(e)}))
#         sys.stdout.flush()






























# #!/usr/bin/env python - with caching  final one but now changiing project  so gonna make it without caching nd add some metrics 
# # process_pdf.py

# import sys
# import os
# import json
# import hashlib
# import concurrent.futures
# import logging
# import re

# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image

# from transformers import (
#     BartTokenizer,
#     BartForConditionalGeneration,
#     pipeline
# )

# # --------------------------------------------------
# # BASIC SAFE CONFIG (XAMPP / Apache friendly)
# # --------------------------------------------------
# logging.getLogger("transformers").setLevel(logging.ERROR)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.stdout.reconfigure(encoding="utf-8")

# # --------------------------------------------------
# # OCR CONFIG
# # --------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# # --------------------------------------------------
# # MODEL LOAD (ONCE)
# # --------------------------------------------------
# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
# model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# summarizer = pipeline(
#     "summarization",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1
# )

# # --------------------------------------------------
# # NORMAL TEXT EXTRACTION
# # --------------------------------------------------
# def extract_text(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text("text") + "\n"
#     return text.strip()

# # --------------------------------------------------
# # OCR FALLBACK
# # --------------------------------------------------
# def ocr_pdf(pdf_path):
#     text = ""
#     try:
#         doc = fitz.open(pdf_path)
#         for page in doc:
#             pix = page.get_pixmap(dpi=200)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             text += pytesseract.image_to_string(img)
#     except Exception:
#         pass
#     return text.strip()

# # --------------------------------------------------
# # CLEAN OCR + NOISE
# # --------------------------------------------------
# def clean_text(text):
#     text = re.sub(r'[^A-Za-z0-9.,;:()\n ]+', ' ', text)
#     lines = text.splitlines()
#     clean_lines = [l for l in lines if len(l.strip()) > 30]
#     text = " ".join(clean_lines)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# # --------------------------------------------------
# # CHUNKING
# # --------------------------------------------------
# def split_text(text, max_words=500):
#     words = text.split()
#     chunks, chunk = [], []

#     for w in words:
#         chunk.append(w)
#         if len(chunk) >= max_words:
#             chunks.append(" ".join(chunk))
#             chunk = []

#     if chunk:
#         chunks.append(" ".join(chunk))

#     return [c[:4000] for c in chunks]

# # --------------------------------------------------
# # ✅ FIX 2 APPLIED HERE (NO PROMPT)
# # --------------------------------------------------
# def summarize_chunk(chunk):
#     try:
#         result = summarizer(
#             chunk,
#             max_length=160,
#             min_length=80,
#             do_sample=False
#         )
#         return result[0]["summary_text"]
#     except Exception:
#         return ""

# def summarize_text(text):
#     chunks = split_text(text)
#     summaries = []

#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         for s in executor.map(summarize_chunk, chunks):
#             if s:
#                 summaries.append(s)

#     return " ".join(summaries).strip()

# # --------------------------------------------------
# # CACHE
# # --------------------------------------------------
# def get_pdf_hash(pdf_path):
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# def load_cached_summary(pdf_hash):
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     if os.path.exists(path):
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f).get("summary")
#     return None

# def save_summary(pdf_hash, summary):
#     os.makedirs("saved_summaries", exist_ok=True)
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

# # --------------------------------------------------
# # MAIN
# # --------------------------------------------------
# if __name__ == "__main__":
#     try:
#         if len(sys.argv) < 2:
#             print(json.dumps({"summary": "", "error": "No PDF provided"}))
#             sys.exit(0)

#         pdf_path = sys.argv[1]

#         if not os.path.exists(pdf_path):
#             print(json.dumps({"summary": "", "error": "PDF not found"}))
#             sys.exit(0)

#         pdf_hash = get_pdf_hash(pdf_path)

#         cached = load_cached_summary(pdf_hash)
#         if cached:
#             print(json.dumps({"summary": cached, "cached": True}, ensure_ascii=False))
#             sys.exit(0)

#         text = extract_text(pdf_path)

#         if len(text) < 100:
#             text = ocr_pdf(pdf_path)

#         text = clean_text(text)

#         if len(text) < 100:
#             print(json.dumps({"summary": "", "error": "No readable text"}))
#             sys.exit(0)

#         summary = summarize_text(text)

#         save_summary(pdf_hash, summary)

#         print(json.dumps({"summary": summary, "cached": False}, ensure_ascii=False))
#         sys.stdout.flush()

#     except Exception as e:
#         print(json.dumps({"summary": "", "error": str(e)}))
#         sys.stdout.flush()



























# index.html 

# <?php
# session_start();

# if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
#     header("Location: login.php");
#     exit;
# }
# ?>
# <!DOCTYPE html>
# <html lang="en">

# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>PDF Summarizer</title>

#     <style>
#         body {
#             font-family: 'Segoe UI', Tahoma, sans-serif;
#             text-align: center;
#             margin: 40px;
#             background: linear-gradient(135deg, #e0f7fa, #f8f9fa);
#         }

#         h1 {
#             color: #00695c;
#             margin-bottom: 10px;
#         }

#         input[type="file"],
#         select,
#         button {
#             padding: 10px;
#             margin: 10px;
#             font-size: 16px;
#             border-radius: 8px;
#             border: 1px solid #ccc;
#             outline: none;
#         }

#         button {
#             background-color: #26a69a;
#             color: white;
#             border: none;
#             cursor: pointer;
#             transition: 0.3s;
#         }

#         button:hover {
#             background-color: #00796b;
#         }

#         #summaryOutput {
#             margin-top: 30px;
#             font-weight: bold;
#             color: #004d40;
#             white-space: pre-wrap;
#             background: #e0f2f1;
#             padding: 15px;
#             border-radius: 10px;
#             width: 80%;
#             margin-left: auto;
#             margin-right: auto;
#             box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
#         }

#         #loadingMessage {
#             margin-top: 10px;
#             font-weight: bold;
#             color: #f57c00;
#             display: none;
#         }

#         label {
#             color: #00796b;
#             font-weight: bold;
#         }

#         a.logout {
#             position: absolute;
#             top: 20px;
#             right: 30px;
#             text-decoration: none;
#             color: white;
#             background-color: #dc3545;
#             padding: 8px 15px;
#             border-radius: 5px;
#         }
#     </style>
# </head>

# <body>

#     <a href="logout.php" class="logout">Logout</a>

#     <h1>📄 Smart PDF Summarizer</h1>

#     <!-- Upload PDF -->
#     <input type="file" id="pdfUpload" accept="application/pdf">
#     <br>

#     <!-- Ask if user wants to save -->
#     <label for="saveChoice">Do you want to save this PDF in our database for faster summaries next time?</label>

#     <select id="saveChoice">
#         <option value="yes">✅ Yes, save it</option>
#         <option value="no">❌ No, delete after summary</option>
#     </select>

#     <br>

#     <button onclick="uploadPDF()">Upload & Summarize</button>

#     <p id="loadingMessage">⚙ Processing your PDF, please wait...</p>

#     <h2>Summary</h2>
#     <p id="summaryOutput">Waiting for file upload...</p>

#     <script>
#         function uploadPDF() {
#             const fileInput = document.getElementById("pdfUpload");
#             const pdfFile = fileInput.files[0];
#             const saveChoice = document.getElementById("saveChoice").value;

#             if (!pdfFile) {
#                 alert("Please select a PDF file first.");
#                 return;
#             }

#             const formData = new FormData();
#             formData.append("file", pdfFile);
#             formData.append("saveChoice", saveChoice);

#             const loading = document.getElementById("loadingMessage");
#             const output = document.getElementById("summaryOutput");

#             loading.style.display = "block";
#             output.innerText = "";

#             fetch("upload.php", {
#                 method: "POST",
#                 body: formData
#             })
#                 .then(response => response.json())
#                 .then(data => {
#                     loading.style.display = "none";

#                     console.log("Response from PHP:", data);

#                     if (data.status === "success") {
#                         if (data.summary && data.summary.trim() !== "") {
#                             output.innerText = data.summary;
#                         } else {
#                             output.innerText = "⚠ Summary is empty — check debug_output.txt for details.";
#                         }
#                     } else {
#                         output.innerText = "❌ " + (data.message || "An error occurred.");
#                     }
#                 })
#                 .catch(error => {
#                     loading.style.display = "none";
#                     console.error("Error:", error);
#                     output.innerText = "Error uploading or summarizing file.";
#                 });
#         }
#     </script>

# </body>

# </html>











# Getting answers for both scanned also 
# #!/usr/bin/env python
# # process_pdf.py

# import sys
# import os
# import json
# import hashlib
# import concurrent.futures
# import logging

# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image

# from transformers import (
#     BartTokenizer,
#     BartForConditionalGeneration,
#     pipeline
# )

# # --------------------------------------------------
# # BASIC SAFE CONFIG (XAMPP / Apache friendly)
# # --------------------------------------------------
# logging.getLogger("transformers").setLevel(logging.ERROR)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.stdout.reconfigure(encoding="utf-8")

# # --------------------------------------------------
# # OCR CONFIG (REFERENCE FROM YOUR GEMINI CODE)
# # --------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# # --------------------------------------------------
# # MODEL LOAD (ONCE)
# # --------------------------------------------------
# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
# model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# summarizer = pipeline(
#     "summarization",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1  # CPU only (safe locally)
# )

# # --------------------------------------------------
# # NORMAL TEXT EXTRACTION (FAST)
# # --------------------------------------------------
# def extract_text(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text("text") + "\n"
#     return text.strip()

# # --------------------------------------------------
# # OCR FALLBACK (ONLY IF NEEDED)
# # --------------------------------------------------
# def ocr_pdf(pdf_path):
#     text = ""
#     try:
#         doc = fitz.open(pdf_path)
#         for page in doc:
#             pix = page.get_pixmap(dpi=200)  # same DPI as your reference code
#             img = Image.frombytes(
#                 "RGB",
#                 [pix.width, pix.height],
#                 pix.samples
#             )
#             text += pytesseract.image_to_string(img)
#     except Exception:
#         pass

#     return text.strip()

# # --------------------------------------------------
# # CHUNKING (SAFE FOR BART)
# # --------------------------------------------------
# def split_text(text, max_words=500):
#     words = text.split()
#     chunks, chunk = [], []

#     for w in words:
#         chunk.append(w)
#         if len(chunk) >= max_words:
#             chunks.append(" ".join(chunk))
#             chunk = []

#     if chunk:
#         chunks.append(" ".join(chunk))

#     # extra safety: limit chunk size
#     return [c[:4000] for c in chunks]

# # --------------------------------------------------
# # SUMMARIZATION
# # --------------------------------------------------
# def summarize_chunk(chunk):
#     try:
#         result = summarizer(
#             chunk,
#             max_length=160,
#             min_length=60,
#             do_sample=False
#         )
#         return result[0]["summary_text"]
#     except Exception:
#         return ""

# def summarize_text(text):
#     chunks = split_text(text)
#     summaries = []

#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         for s in executor.map(summarize_chunk, chunks):
#             if s:
#                 summaries.append(s)

#     return " ".join(summaries).strip()

# # --------------------------------------------------
# # CACHE (OPTIONAL)
# # --------------------------------------------------
# def get_pdf_hash(pdf_path):
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# def load_cached_summary(pdf_hash):
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     if os.path.exists(path):
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f).get("summary")
#     return None

# def save_summary(pdf_hash, summary):
#     os.makedirs("saved_summaries", exist_ok=True)
#     path = os.path.join("saved_summaries", f"{pdf_hash}.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

# # --------------------------------------------------
# # MAIN (WEB + TERMINAL SAFE)
# # --------------------------------------------------
# if __name__ == "__main__":
#     try:
#         if len(sys.argv) < 2:
#             print(json.dumps({"summary": "", "error": "No PDF provided"}))
#             sys.exit(0)

#         pdf_path = sys.argv[1]

#         if not os.path.exists(pdf_path):
#             print(json.dumps({"summary": "", "error": "PDF not found"}))
#             sys.exit(0)

#         pdf_hash = get_pdf_hash(pdf_path)

#         cached = load_cached_summary(pdf_hash)
#         if cached:
#             print(json.dumps(
#                 {"summary": cached, "cached": True},
#                 ensure_ascii=False
#             ))
#             sys.exit(0)

#         # 1️⃣ Try normal text extraction
#         text = extract_text(pdf_path)

#         # 2️⃣ OCR ONLY if text is missing
#         if len(text) < 100:
#             text = ocr_pdf(pdf_path)

#         if len(text) < 100:
#             print(json.dumps(
#                 {"summary": "", "error": "No readable text"}
#             ))
#             sys.exit(0)

#         # 3️⃣ Summarize
#         summary = summarize_text(text)

#         save_summary(pdf_hash, summary)

#         # 4️⃣ Web-safe output
#         print(json.dumps(
#             {"summary": summary, "cached": False},
#             ensure_ascii=False
#         ))
#         sys.stdout.flush()

#     except Exception as e:
#         print(json.dumps(
#             {"summary": "", "error": str(e)}
#         ))
#         sys.stdout.flush()




















#giving answers but for scanned pdfs not working as no ocr is there 
#  import sys
# import os
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import numpy as np
# import cv2
# import re

# from transformers import pipeline, logging
# logging.set_verbosity_error()

# # Set pytesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# device = 0 if torch.cuda.is_available() else -1

# # Load models
# summarizer = pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6",
#     device=device
# )

# qa_pipeline = pipeline(
#     "question-answering",
#     model="distilbert-base-uncased-distilled-squad",
#     device=device
# )

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     doc.close()
#     return text

# def summarize_text(text):
#     if not text.strip():
#         return "PDF has no readable text."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summarized = ""
#     for chunk in chunks:
#         summary = summarizer(
#             chunk,
#             max_length=250,
#             min_length=30,
#             do_sample=False
#         )[0]["summary_text"]
#         summarized += summary + " "
#     return summarized.strip()

# def answer_question(context, question):
#     return qa_pipeline(question=question, context=context)["answer"]

# if __name__ == "__main__":

#     if len(sys.argv) < 3:
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     question = sys.argv[2]

#     if not os.path.exists(pdf_path):
#         sys.exit(1)

#     context = extract_text_from_pdf(pdf_path)

#     if "summary" in question.lower():
#         result = summarize_text(context)
#     else:
#         result = answer_question(context, question)

#     # 🔒 ABSOLUTE PATH OUTPUT (CRITICAL)
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     output_path = os.path.join(BASE_DIR, "output.txt")

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(result)

#     # For terminal usage
#     print(result)




















# import sys
# import os
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import numpy as np
# import cv2
# import re

# from transformers import pipeline, logging
# logging.set_verbosity_error()  # suppress model warnings

# # Set pytesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# device = 0 if torch.cuda.is_available() else -1

# # Load the summarization and question-answering pipelines
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     doc.close()
#     return text

# # Function to summarize
# def summarize_text(text):
#     if len(text.strip()) == 0:
#         return "❌ PDF has no readable text."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summarized = ""
#     for chunk in chunks:
#         summary = summarizer(chunk, max_length=250, min_length=30, do_sample=False)[0]['summary_text']
#         summarized += summary + " "
#     return summarized.strip()

# # Function to answer a question from PDF text
# def answer_question(context, question):
#     result = qa_pipeline(question=question, context=context)
#     return result['answer']

# # Main
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("⚠ Missing arguments. Usage: python process_pdf.py <pdf_path> <question>")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     question = sys.argv[2]

#     if not os.path.exists(pdf_path):
#         print(f"⚠ File not found: {pdf_path}")
#         sys.exit(1)

#     context = extract_text_from_pdf(pdf_path)

#     # If question is like 'summary', do summary
#     if "summary" in question.lower():
#         print(summarize_text(context))
#     else:
#         print(answer_question(context, question))

















# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load NLP pipelines (force PyTorch only)
# summarizer = pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn",
#     framework="pt",
#     device=0 if device == "cuda" else -1
# )

# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     framework="pt",
#     device=0 if device == "cuda" else -1
# )

# # Helper: Preprocess images for OCR
# def preprocess_for_ocr(pil_img):
#     img = np.array(pil_img.convert('L'))
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
#     return Image.fromarray(img)

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     lines = []

#     for page in doc:
#         text = page.get_text("text").strip()
#         if len(text) > 20:
#             lines.extend([line.strip() for line in text.splitlines() if len(line.strip()) > 5])
#         else:
#             pix = page.get_pixmap(dpi=300)
#             img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
#             ocr_text = pytesseract.image_to_string(img, lang="eng")
#             clean_lines = [
#                 line.strip()
#                 for line in ocr_text.splitlines()
#                 if line.strip() and len(line.split()) >= 5 and any(c.isalpha() for c in line)
#             ]
#             lines.extend(clean_lines)

#     final_lines = [line for line in lines if len(line) > 15]
#     return final_lines

# # Filter out code-style lines
# def is_code_line(line):
#     if not line.strip() or len(line.strip()) < 10:
#         return True
#     symbols = ['{', '}', ';', '==', 'def ', 'class ', '=', 'torch.', 'nn.', 'cv2.', 'plt.', '(', ')', '[', ']']
#     if any(sym in line for sym in symbols):
#         return True
#     alpha_ratio = sum(c.isalpha() for c in line) / max(1, len(line))
#     return alpha_ratio < 0.4

# def filter_natural_language(lines):
#     return [line for line in lines if not is_code_line(line)]

# # Summarize text
# def summarize_text(text):
#     if not text.strip():
#         return "No valid natural language text found to summarize."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summaries = []
#     for chunk in chunks:
#         result = summarizer(chunk, max_length=250, min_length=60, do_sample=False)
#         summaries.append(result[0]['summary_text'])
#     return "\n\n".join(summaries)

# # Answer a question from text
# def answer_question(text, question):
#     if not text.strip():
#         return "No content found to answer the question."
#     context = text[:4000]  # Keep context short enough
#     result = qa_pipeline({'question': question, 'context': context})
#     return result['answer']

# # ---------- Main Execution ----------
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("❌ Error: Missing PDF path")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     if not os.path.isfile(pdf_path):
#         print(f"❌ Error: File not found - {pdf_path}")
#         sys.exit(1)

#     query = sys.argv[2] if len(sys.argv) > 2 else "summarize"

#     # Extract and filter text
#     lines = extract_text_from_pdf(pdf_path)
#     filtered_text = "\n".join(filter_natural_language(lines))

#     # Summarize or answer
#     if "summary" in query.lower() or "summarize" in query.lower():
#         summary = summarize_text(filtered_text)
#         print(summary)
#     else:
#         answer = answer_question(filtered_text, query)
#         print(answer)
































# import sys
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import numpy as np
# import cv2
# import re

# from transformers import pipeline

# # Set pytesseract path (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# #print(f"Device set to use {device}")

# # Load NLP pipelines
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if device == "cuda" else -1)

# # Helper: Preprocess images for OCR
# def preprocess_for_ocr(pil_img):
#     img = np.array(pil_img.convert('L'))
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
#     return Image.fromarray(img)

# # Extract text from PDF using text or OCR
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     lines = []

#     for page in doc:
#         text = page.get_text("text").strip()
#         if len(text) > 20:
#             lines.extend([line.strip() for line in text.splitlines() if len(line.strip()) > 5])
#         else:
#             pix = page.get_pixmap(dpi=300)
#             img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
#             ocr_text = pytesseract.image_to_string(img, lang="eng")
#             clean_lines = [
#                 line.strip()
#                 for line in ocr_text.splitlines()
#                 if line.strip() and len(line.split()) >= 5 and any(c.isalpha() for c in line)
#             ]
#             lines.extend(clean_lines)

#     final_lines = [line for line in lines if len(line) > 15]
#     return final_lines

# # Filter natural language
# def is_code_line(line):
#     if not line.strip() or len(line.strip()) < 10:
#         return True
#     symbols = ['{', '}', ';', '==', 'def ', 'class ', '=', 'torch.', 'nn.', 'cv2.', 'plt.', '(', ')', '[', ']']
#     if any(sym in line for sym in symbols):
#         return True
#     alpha_ratio = sum(c.isalpha() for c in line) / max(1, len(line))
#     return alpha_ratio < 0.4

# def filter_natural_language(lines):
#     return [line for line in lines if not is_code_line(line)]

# # Summarization
# def summarize_text(text):
#     if not text.strip():
#         return "No valid natural language text found to summarize."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summaries = []
#     for chunk in chunks:
#         result = summarizer(chunk, max_length=250, min_length=60, do_sample=False)
#         summaries.append(result[0]['summary_text'])
#     return "\n\n".join(summaries)

# # Question Answering
# def answer_question(text, question):
#     if not text.strip():
#         return "No content found to answer the question."
#     context = text[:4000]  # Limit to first 4000 characters
#     result = qa_pipeline({'question': question, 'context': context})
#     return result['answer']

# # Main Execution
# # Main Execution
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Error: Missing PDF path")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     query = sys.argv[2] if len(sys.argv) > 2 else "summarize"

#     lines = extract_text_from_pdf(pdf_path)
#     filtered_text = "\n".join(filter_natural_language(lines))

#     if query.lower().strip() == "summarize" or "summary" in query.lower():
#         summary = summarize_text(filtered_text)
#         print(summary)
#     else:
#         answer = answer_question(filtered_text, query)
#         print(answer)