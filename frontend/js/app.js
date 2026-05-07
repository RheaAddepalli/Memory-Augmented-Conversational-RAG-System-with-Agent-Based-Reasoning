// ============================================================
// STATE
// ============================================================
let uploadedFileName = "";
let sessionId = generateId(); // 🧠 persistent across turns
let isProcessing = false;
let currentTraceItems = [];

document.getElementById("sessionDisplay").textContent = "SESSION " + sessionId.toUpperCase();

// ============================================================
// HELPERS
// ============================================================
function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

function formatTime(seconds) {
  seconds = parseFloat(seconds);
  if (isNaN(seconds)) return "—";
  if (seconds < 60) return seconds.toFixed(2) + "s";
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return mins + "m " + secs + "s";
}

function fillQ(text) {
  document.getElementById("questionInput").value = text;
  document.getElementById("questionInput").focus();
}

function scoreClass(val, good, warn) {
  return val >= good ? "good" : val >= warn ? "warn" : "bad";
}

// ============================================================
// AUTO-RESIZE TEXTAREA
// ============================================================
const textarea = document.getElementById("questionInput");
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
});

textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

// ============================================================
// UPLOAD
// ============================================================
const uploadArea = document.getElementById("uploadArea");
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("dragover"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) { document.getElementById("pdfUpload").files = e.dataTransfer.files; doUpload(file); }
});
document.getElementById("pdfUpload").addEventListener("change", e => {
  if (e.target.files[0]) doUpload(e.target.files[0]);
});

function doUpload(file) {
  const status = document.getElementById("uploadStatus");
  status.className = "upload-status";
  status.textContent = "⏳ Uploading...";
  status.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://127.0.0.1:5000/upload", { method: "POST", body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.status === "success") {
        uploadedFileName = data.filename;
        status.className = "upload-status success";
        status.textContent = "✅ " + uploadedFileName;
        document.getElementById("uploadIcon").textContent = "📋";
      } else {
        status.className = "upload-status error";
        status.textContent = "❌ " + (data.message || "Upload failed");
      }
    })
    .catch(() => {
      status.className = "upload-status error";
      status.textContent = "❌ Upload failed";
    });
}

// ============================================================
// CLEAR SESSION
// ============================================================
document.getElementById("clearBtn").addEventListener("click", () => {
  fetch("http://127.0.0.1:5000/clear_session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  });
  sessionId = generateId();
  document.getElementById("sessionDisplay").textContent = "SESSION " + sessionId.toUpperCase();

  // Clear chat UI
  const history = document.getElementById("chatHistory");
  history.innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="empty-title">Ask anything about<br><em>your document</em></div>
      <div class="empty-sub">Upload a PDF and ask questions. The system remembers your conversation context.</div>
      <div class="suggestions">
        <div class="suggestion-chip" onclick="fillQ('Give me a summary')">Give me a summary</div>
        <div class="suggestion-chip" onclick="fillQ('What are the key findings?')">Key findings</div>
        <div class="suggestion-chip" onclick="fillQ('List all main topics')">Main topics</div>
        <div class="suggestion-chip" onclick="fillQ('What conclusions are drawn?')">Conclusions</div>
      </div>
    </div>`;
});

// ============================================================
// SUBMIT BUTTON
// ============================================================
document.getElementById("submitBtn").addEventListener("click", askQuestion);

// ============================================================
// ASK QUESTION
// ============================================================
function askQuestion() {
  const question = textarea.value.trim();
  if (!question || isProcessing) return;
  if (!uploadedFileName) { alert("Please upload a PDF first."); return; }

  isProcessing = true;
  const requestId = generateId();
  currentTraceItems = [];

  // Hide empty state
  const emptyState = document.getElementById("emptyState");
  if (emptyState) emptyState.remove();

  // Reset textarea
  textarea.value = "";
  textarea.style.height = "auto";
  document.getElementById("submitBtn").disabled = true;

  // Add user bubble
  const history = document.getElementById("chatHistory");
  const turnEl = document.createElement("div");
  turnEl.className = "turn";
  turnEl.innerHTML = `
    <div class="turn-question">
      <div class="turn-avatar avatar-user">U</div>
      <div class="bubble-user">${escapeHtml(question)}</div>
    </div>`;
  history.appendChild(turnEl);

  // Add processing bubble
  const processingEl = document.createElement("div");
  processingEl.className = "processing-turn";
  processingEl.id = "processingBubble";
  processingEl.innerHTML = `
    <div class="turn-avatar avatar-ai">✦</div>
    <div class="thinking-bubble">
      <div class="thinking-dots">
        <span></span><span></span><span></span>
      </div>
      <div class="thinking-label" id="thinkingLabel">Processing…</div>
    </div>`;
  history.appendChild(processingEl);
  processingEl.scrollIntoView({ behavior: "smooth", block: "end" });

  // Start SSE stream for events
  const evtSource = new EventSource(`http://127.0.0.1:5000/stream?request_id=${requestId}`);
  evtSource.onmessage = function(e) {
    const data = JSON.parse(e.data);
    if (data.type === "heartbeat") return;
    if (data.type === "done") { evtSource.close(); return; }
    if (data.type === "token") return;
    if (data.type === "extract_start" || data.type === "workflow_start") {
      const label = document.getElementById("thinkingLabel");
      if (label) label.textContent = data.message.replace(/📄|⚙️/g, "").trim();
    }
    currentTraceItems.push({ type: data.type, message: data.message });
  };
  evtSource.onerror = () => evtSource.close();

  // Main fetch
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3600000);

  fetch("http://127.0.0.1:5000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      filename: uploadedFileName,
      request_id: requestId,
      session_id: sessionId  // 🧠 memory
    }),
    signal: controller.signal
  })
  .then(r => r.json())
  .then(data => {
    clearTimeout(timeout);
    evtSource.close();
    finishTurn(turnEl, processingEl, data, question);
  })
  .catch(err => {
    clearTimeout(timeout);
    evtSource.close();
    finishTurn(turnEl, processingEl, {
      answer: err.name === "AbortError" ? "Request timed out." : "Error: " + err.message,
      metrics: null
    }, question);
  });
}

// ============================================================
// FINISH TURN — render answer + trace + metrics
// ============================================================
function finishTurn(turnEl, processingEl, data, question) {
  isProcessing = false;
  document.getElementById("submitBtn").disabled = false;

  // Remove processing bubble
  processingEl.remove();

  const answer = data.answer || "No answer returned.";
  const rewritten = data.rewritten_query;

  // Build answer section
  let answerHTML = `<div class="turn-answer">
    <div class="turn-avatar avatar-ai">✦</div>
    <div class="bubble-ai">${escapeHtml(answer)}</div>
  </div>`;

  // Rewrite note
  if (rewritten && rewritten !== question) {
    answerHTML += `<div class="rewrite-note">🧠 Query rewritten: "${escapeHtml(rewritten)}"</div>`;
  }

  // Reasoning trace
  if (currentTraceItems.length > 0) {
    const traceItems = currentTraceItems.map(item => {
      const icons = {
        extract_start: "📄", workflow_start: "⚙️",
        agent_start: "🤖", agent_thought: "💭",
        agent_search: "🔍", agent_action: "⚡", agent_done: "✅"
      };
      const icon = icons[item.type] || "·";
      return `<div class="trace-item"><span class="trace-icon">${icon}</span><span class="trace-text">${escapeHtml(item.message)}</span></div>`;
    }).join("");

    answerHTML += `<div class="trace-wrap">
      <button class="trace-toggle" onclick="toggleTrace(this)">REASONING TRACE</button>
      <div class="trace-list">${traceItems}</div>
    </div>`;
  }

  // Metrics
  if (data.metrics && Object.keys(data.metrics).length > 0) {
    answerHTML += `<div class="metrics-wrap">
      <button class="metrics-toggle" onclick="toggleMetrics(this)">BENCHMARK METRICS</button>
      <div class="metrics-body">${buildMetricsHTML(data.metrics)}</div>
    </div>`;
  }

  turnEl.insertAdjacentHTML("beforeend", answerHTML);
  turnEl.scrollIntoView({ behavior: "smooth", block: "end" });
}

// ============================================================
// TOGGLE HELPERS
// ============================================================
function toggleTrace(btn) {
  btn.classList.toggle("open");
  btn.nextElementSibling.classList.toggle("open");
}

function toggleMetrics(btn) {
  btn.classList.toggle("open");
  btn.nextElementSibling.classList.toggle("open");
}

// ============================================================
// METRICS HTML
// ============================================================
function buildMetricsHTML(m) {
  let html = `<div class="metrics-grid">`;

  html += `<div class="metrics-section-title">📊 Core Performance</div>`;
  const core = [
    { label: "Type",       value: m.type === "summary" ? "Summary" : "Q&A" },
    { label: "Total Time", value: formatTime(m.response_time_sec) },
    { label: "Extraction", value: formatTime(m.extraction_time_sec) },
  ];
  if (m.type === "summary") {
    core.push({ label: "Summary Time",   value: formatTime(m.summary_time_sec) });
    core.push({ label: "Summary Words",  value: (m.summary_length_words||0).toLocaleString() });
    core.push({ label: "LLM Calls",      value: m.llm_calls || "—" });
  } else {
    core.push({ label: "QA Time",   value: formatTime(m.qa_time_sec) });
    core.push({ label: "Model",     value: m.model_used || "—" });
    core.push({ label: "LLM Calls", value: m.llm_calls ?? "—" });
  }
  core.push({ label: "Pages",      value: m.pages_processed || "—" });
  core.push({ label: "Words",      value: (m.words_processed||0).toLocaleString() });
  core.forEach(item => {
    html += `<div class="metric-card"><div class="metric-label">${item.label}</div><div class="metric-value">${item.value}</div></div>`;
  });

  html += `<div class="metrics-section-title">⚡ Latency</div>`;
  const ttft = parseFloat(m.ttft_sec || 0);
  const e2e  = parseFloat(m.e2e_latency_sec || 0);
  const tps  = parseFloat(m.tps || 0);
  [
    { label: "TTFT",        value: formatTime(ttft), cls: ttft < e2e ? "good" : "warn" },
    { label: "E2E Latency", value: formatTime(e2e),  cls: "warn" },
    { label: "TPS",         value: tps.toFixed(1),   cls: tps >= 5 ? "good" : tps >= 2 ? "warn" : "bad" },
  ].forEach(item => {
    html += `<div class="metric-card"><div class="metric-label">${item.label}</div><div class="metric-value ${item.cls}">${item.value}</div></div>`;
  });

  if (m.type !== "summary") {
    html += `<div class="metrics-section-title">🎯 RAG Quality</div>`;
    const retrieval = parseFloat(m.retrieval_score || 0);
    const conf      = parseFloat(m.confidence_score || 0);
    const recall    = parseFloat(m.recall_at_k || 0);
    [
      { label: "Retrieval",  value: retrieval.toFixed(1) + "%", cls: scoreClass(retrieval, 40, 25) },
      { label: "Confidence", value: conf.toFixed(1) + "%",      cls: scoreClass(conf, 60, 30) },
      { label: "Recall@K",   value: recall.toFixed(1) + "%",    cls: scoreClass(recall, 70, 40) },
    ].forEach(item => {
      html += `<div class="metric-card"><div class="metric-label">${item.label}</div><div class="metric-value ${item.cls}">${item.value}</div></div>`;
    });

    html += `<div class="metrics-section-title">🧠 Decision</div>`;
    const dt = m.decision_type || "accepted";
    const dtLabel = dt === "accepted" ? "✅ Accepted" : dt === "hard_reject" ? "🚫 Hard Reject" : "❌ Rejected";
    const dtCls   = dt === "accepted" ? "decision-accepted" : "decision-rejected";
    html += `<div class="metric-card" style="grid-column:1/-1">
      <div class="metric-label">Decision Type</div>
      <div style="margin-top:6px"><span class="decision-badge ${dtCls}">${dtLabel}</span></div>
    </div>`;
  }

  html += `</div>`;
  return html;
}

// ============================================================
// ESCAPE HTML
// ============================================================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}









































































// //without memory.py
//  let uploadedFileName = "";

// function formatTime(seconds) {
//   seconds = parseFloat(seconds);
//   if (isNaN(seconds)) return "—";
//   if (seconds < 60) return seconds.toFixed(2) + " sec";
//   const mins = Math.floor(seconds / 60);
//   const secs = (seconds % 60).toFixed(2);
//   return mins + " min " + secs + " sec";
// }

// function generateId() {
//   return Math.random().toString(36).substr(2, 9);
// }

// const uploadArea = document.getElementById("uploadArea");
// uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("dragover"); });
// uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
// uploadArea.addEventListener("drop", e => {
//   e.preventDefault();
//   uploadArea.classList.remove("dragover");
//   const file = e.dataTransfer.files[0];
//   if (file) { document.getElementById("pdfUpload").files = e.dataTransfer.files; doUpload(file); }
// });
// document.getElementById("pdfUpload").addEventListener("change", e => {
//   if (e.target.files[0]) doUpload(e.target.files[0]);
// });

// function doUpload(file) {
//   const status = document.getElementById("uploadStatus");
//   status.className = "upload-status";
//   status.textContent = "⏳ Uploading...";
//   status.style.display = "block";

//   const formData = new FormData();
//   formData.append("file", file);

//   fetch("http://127.0.0.1:5000/upload", { method: "POST", body: formData })
//     .then(r => r.json())
//     .then(data => {
//       if (data.status === "success") {
//         uploadedFileName = data.filename;
//         status.className = "upload-status success";
//         status.textContent = "✅ " + uploadedFileName;
//         document.querySelector('.upload-icon').textContent = "📋";
//       } else {
//         status.className = "upload-status error";
//         status.textContent = "❌ " + data.message;
//       }
//     })
//     .catch(() => {
//       status.className = "upload-status error";
//       status.textContent = "❌ Upload failed";
//     });
// }

// const EVENT_CONFIG = {
//   extract_start:  { icon: "📄", label: "EXTRACTION" },
//   workflow_start: { icon: "⚙️", label: "WORKFLOW" },
//   agent_start:    { icon: "🤖", label: "AGENT INIT" },
//   agent_thought:  { icon: "💭", label: "REASONING" },
//   agent_search:   { icon: "🔍", label: "SEARCHING" },
//   agent_action:   { icon: "⚡", label: "ACTION" },
//   agent_done:     { icon: "✅", label: "COMPLETE" },
//   stream_start:   { icon: "✍️", label: "STREAMING" },
//   heartbeat:      null,
//   done:           null,
//   token:          null,
// };

// function addTimelineEvent(type, message) {
//   const cfg = EVENT_CONFIG[type];
//   if (!cfg) return;
//   const timeline = document.getElementById("timeline");
//   const item = document.createElement("div");
//   item.className = `timeline-item timeline-line event-${type}`;
//   item.innerHTML = `
//     <div class="timeline-dot">${cfg.icon}</div>
//     <div class="timeline-content">
//       <div class="timeline-label">${cfg.label}</div>
//       <div class="timeline-message">${message}</div>
//     </div>`;
//   timeline.appendChild(item);
//   item.scrollIntoView({ behavior: "smooth", block: "nearest" });
// }

// function askQuestion() {
//   const question = document.getElementById("questionInput").value.trim();
//   if (!question) { alert("Please enter a question."); return; }
//   if (!uploadedFileName) { alert("Please upload a PDF first."); return; }

//   const requestId = generateId();
//   const summaryKeywords = ["summary", "summarize", "overview", "brief", "outline", "gist", "tldr"];
//   const isSummary = summaryKeywords.some(k => question.toLowerCase().includes(k));

//   document.getElementById("agentPanel").style.display = isSummary ? "none" : "block";
//   document.getElementById("answerPanel").style.display = "none";
//   document.getElementById("metricsPanel").style.display = "none";
//   document.getElementById("timeline").innerHTML = "";
//   document.getElementById("agentSpinner").style.display = "block";
//   document.getElementById("submitBtn").disabled = true;
//   document.getElementById("submitBtn").textContent = "Processing...";

//   let evtSource = null;

//   if (!isSummary) {
//     evtSource = new EventSource(`http://127.0.0.1:5000/stream?request_id=${requestId}`);
//     document.getElementById("answerPanel").style.display = "block";
//     document.getElementById("answerBox").textContent = "";
//     document.getElementById("answerBox").setAttribute("data-streaming", "true");

//     evtSource.onmessage = function(e) {
//       const data = JSON.parse(e.data);
//       if (data.type === "heartbeat") return;
//       if (data.type === "done") {
//         evtSource.close();
//         document.getElementById("agentSpinner").style.display = "none";
//         document.getElementById("answerBox").removeAttribute("data-streaming");
//         return;
//       }
//       if (data.type === "stream_start") {
//         document.getElementById("answerPanel").style.display = "block";
//         document.getElementById("answerBox").setAttribute("data-streaming", "true");
//         return;
//       }
//       if (data.type === "token") {
//         const box = document.getElementById("answerBox");
//         document.getElementById("answerPanel").style.display = "block";
//         box.textContent += data.message;
//         box.scrollTop = box.scrollHeight;
//         return;
//       }
//       addTimelineEvent(data.type, data.message);
//     };

//     evtSource.onerror = function() { evtSource.close(); };

//   } else {
//     document.getElementById("agentPanel").style.display = "block";
//     document.getElementById("timeline").innerHTML = "";
//     addTimelineEvent("extract_start", "📄 Extracting PDF text...");
//     addTimelineEvent("workflow_start", "⚙️ Running RAPTOR hierarchical summarization...");
//   }

//   const controller = new AbortController();
//   const timeout = setTimeout(() => controller.abort(), 3600000);

//   fetch("http://127.0.0.1:5000/ask", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ question, filename: uploadedFileName, request_id: requestId }),
//     signal: controller.signal
//   })
//   .then(r => r.json())
//   .then(data => {
//     clearTimeout(timeout);
//     if (evtSource) evtSource.close();
//     document.getElementById("agentSpinner").style.display = "none";
//     document.getElementById("submitBtn").disabled = false;
//     document.getElementById("submitBtn").textContent = "Run Agent →";

//     const box = document.getElementById("answerBox");
//     box.removeAttribute("data-streaming");
//     document.getElementById("answerPanel").style.display = "block";
//     box.textContent = data.answer || "No answer returned.";

//     if (data.metrics) showMetrics(data.metrics);
//   })
//   .catch(err => {
//     clearTimeout(timeout);
//     if (evtSource) evtSource.close();
//     document.getElementById("submitBtn").disabled = false;
//     document.getElementById("submitBtn").textContent = "Run Agent →";
//     document.getElementById("answerPanel").style.display = "block";
//     document.getElementById("answerBox").textContent =
//       err.name === "AbortError" ? "Request timed out." : "Error: " + err.message;
//   });
// }

// function scoreClass(val, goodThresh, warnThresh) {
//   if (val >= goodThresh) return "good";
//   if (val >= warnThresh) return "warn";
//   return "bad";
// }

// function showMetrics(m) {
//   const grid = document.getElementById("metricsGrid");
//   grid.innerHTML = "";

//   // ── Core Performance ──
//   grid.innerHTML += `<div class="metrics-section-title">📊 Core Performance</div>`;
//   const core = [
//     { label: "Type",       value: m.type === "summary" ? "Summary" : "Q&A", unit: "" },
//     { label: "Total Time", value: formatTime(m.response_time_sec),           unit: "" },
//     { label: "Extraction", value: formatTime(m.extraction_time_sec),         unit: "" },
//   ];
//   if (m.type === "summary") {
//     core.push({ label: "Summary Time",   value: formatTime(m.summary_time_sec),               unit: "" });
//     core.push({ label: "Summary Length", value: (m.summary_length_words||0).toLocaleString(), unit: "words" });
//     core.push({ label: "LLM Calls",      value: m.llm_calls || "—",                           unit: "calls" });
//   } else {
//     core.push({ label: "QA Time",   value: formatTime(m.qa_time_sec), unit: "" });
//     core.push({ label: "Model",     value: m.model_used || "—",       unit: "" });
//     core.push({ label: "LLM Calls", value: m.llm_calls ?? "—",        unit: "calls" });
//   }
//   core.push({ label: "Pages",      value: m.pages_processed,                            unit: "pages" });
//   core.push({ label: "Characters", value: (m.characters_processed||0).toLocaleString(), unit: "chars" });
//   core.push({ label: "Words",      value: (m.words_processed||0).toLocaleString(),      unit: "words" });
//   core.forEach(item => {
//     grid.innerHTML += `<div class="metric-item"><div class="metric-label">${item.label}</div><div class="metric-value">${item.value}<span class="metric-unit"> ${item.unit}</span></div></div>`;
//   });

//   // ── Latency Metrics ──
//   grid.innerHTML += `<div class="metrics-section-title">⚡ Latency Metrics</div>`;
//   const ttft = parseFloat(m.ttft_sec || m.response_time_sec || 0);
//   const e2e  = parseFloat(m.e2e_latency_sec || m.response_time_sec || 0);
//   const tps  = parseFloat(m.tps || 0);
//   [
//     { label: "TTFT", sub: "Time To First Token", value: formatTime(ttft),
//       note: ttft < e2e ? "✅ Streaming active!" : "= E2E (no streaming yet)",
//       cls: ttft < e2e ? "good" : "warn" },
//     { label: "E2E Latency", sub: "End To End", value: formatTime(e2e),
//       note: "total response time", cls: "warn" },
//     { label: "TPS", sub: "Tokens/Second", value: tps.toFixed(1),
//       note: tps >= 5 ? "✅ Fast" : tps >= 2 ? "⚠️ Medium" : "❌ Slow",
//       cls: tps >= 5 ? "good" : tps >= 2 ? "warn" : "bad" },
//   ].forEach(item => {
//     grid.innerHTML += `<div class="metric-item ${item.cls}"><div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div><div class="metric-value">${item.value}</div><div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div></div>`;
//   });

//   // ── RAG Quality Metrics ──
//   if (m.type !== "summary") {
//     grid.innerHTML += `<div class="metrics-section-title">🎯 RAG Quality Metrics</div>`;
//     const retrieval = parseFloat(m.retrieval_score || 0);
//     const conf      = parseFloat(m.confidence_score || 0);
//     const recallAtK = parseFloat(m.recall_at_k || 0);

//     [
//       {
//         label: "Retrieval Score", sub: "Chunk Relevance",
//         value: retrieval.toFixed(1) + "%",
//         note:  retrieval >= 40 ? "✅ Good" : "⚠️ Low",
//         cls:   scoreClass(retrieval, 40, 25)
//       },
//       {
//         label: "Confidence", sub: "Retrieval + Verified",
//         value: conf.toFixed(1) + "%",
//         note:  conf >= 60 ? "✅ High" : conf >= 30 ? "⚠️ Medium" : "❌ Low",
//         cls:   scoreClass(conf, 60, 30)
//       },
//       {
//         label: "Recall@K", sub: "Oracle Chunk Hit",
//         value: recallAtK.toFixed(1) + "%",
//         note:  recallAtK === 100  ? "✅ Oracle retrieved" :
//                recallAtK >= 70   ? "✅ Very close" :
//                recallAtK >= 40   ? "⚠️ Partial" : "❌ Missed",
//         cls:   scoreClass(recallAtK, 70, 40)
//       },
//     ].forEach(item => {
//       grid.innerHTML += `<div class="metric-item ${item.cls}">
//         <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
//         <div class="metric-value">${item.value}</div>
//         <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
//       </div>`;
//     });

//     // ── Decision Intelligence (NEW) ──
//     grid.innerHTML += `<div class="metrics-section-title">🧠 Decision Intelligence</div>`;

//     // Decision Type badge
//     const dt = m.decision_type || "accepted";
//     const dtLabel = dt === "accepted"                  ? "✅ Accepted"
//                   : dt === "rejected_verification"     ? "❌ Rejected · Verification"
//                   : dt === "rejected_low_retrieval"    ? "🚫 Rejected · Low Retrieval"
//                   : dt === "hard_reject"               ? "🚫 Hard Reject"
//                   : dt;
//     const dtCls = dt === "accepted" ? "decision-accepted" : "decision-rejected";
//     grid.innerHTML += `<div class="metric-item" style="grid-column: 1/-1; background: var(--surface);">
//       <div class="metric-label">Decision Type</div>
//       <div style="margin-top:6px;"><span class="decision-badge ${dtCls}">${dtLabel}</span></div>
//     </div>`;

//     // Verification Mode + Keyword Score + Verified
//     const vm       = m.verification_mode || "none";
//     const kwScore  = parseFloat(m.keyword_score || 0);
//     const verified = m.verified === true;

//     [
//       {
//         label: "Verification Mode", sub: "Strictness",
//         value: vm === "strict" ? "STRICT" : vm === "normal" ? "NORMAL" : "NONE",
//         note:  vm === "strict" ? "⚠️ Weak retrieval" : vm === "normal" ? "✅ Standard" : "— Not applied",
//         cls:   vm === "strict" ? "warn" : vm === "normal" ? "good" : ""
//       },
//       {
//         label: "Keyword Score", sub: "Answer Grounding",
//         value: kwScore.toFixed(1) + "%",
//         note:  kwScore >= 80 ? "✅ Strong" : kwScore >= 60 ? "✅ Good" : kwScore >= 30 ? "⚠️ Weak" : "❌ Low",
//         cls:   scoreClass(kwScore, 60, 30)
//       },
//       {
//         label: "Verified", sub: "Answer Confirmed",
//         value: verified ? "YES" : "NO",
//         note:  verified ? "✅ Answer verified" : "❌ Not verified",
//         cls:   verified ? "good" : "bad"
//       },
//     ].forEach(item => {
//       grid.innerHTML += `<div class="metric-item ${item.cls}">
//         <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
//         <div class="metric-value">${item.value}</div>
//         <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
//       </div>`;
//     });

//     // ── Debug Metrics (display only) ──
//     grid.innerHTML += `<div class="metrics-section-title">🔬 Debug Metrics · For Analysis Only</div>`;
//     const precision = parseFloat(m.context_precision || 0);
//     const grounding = parseFloat(m.answer_grounding  || 0);
//     [
//       {
//         label: "Context Precision", sub: "Relevant/Retrieved",
//         value: precision.toFixed(1) + "%",
//         note:  "Debug only — not used in decisions",
//         cls:   ""
//       },
//       {
//         label: "Answer Grounding", sub: "Word Overlap",
//         value: grounding.toFixed(1) + "%",
//         note:  "Debug only — not used in decisions",
//         cls:   ""
//       },
//     ].forEach(item => {
//       grid.innerHTML += `<div class="metric-item" style="opacity:0.6;">
//         <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
//         <div class="metric-value" style="font-size:16px;">${item.value}</div>
//         <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
//       </div>`;
//     });
//   }

//   document.getElementById("metricsPanel").style.display = "block";
// }

// document.getElementById("questionInput").addEventListener("keydown", e => {
//   if (e.key === "Enter") askQuestion();
// });
// document.getElementById("submitBtn").addEventListener("click", askQuestion);