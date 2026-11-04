/* ── RevealAI Frontend ──────────────────────────────────────────── */

(function () {
  "use strict";

  // ── DOM refs ────────────────────────────────────────────────────
  const $  = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const tabs         = $$(".tab");
  const panelUpload  = $("#panelUpload");
  const panelLive    = $("#panelLive");
  const dropZone     = $("#dropZone");
  const fileInput    = $("#fileInput");
  const modelSelect  = $("#modelSelect");
  const btnAnalyze   = $("#btnAnalyze");
  const progressWrap = $("#progressWrap");
  const progressFill = $("#progressFill");
  const progressLabel= $("#progressLabel");
  const resultCard   = $("#resultCard");

  const btnLive         = $("#btnLive");
  const liveModelSelect = $("#liveModelSelect");
  const pulseRing       = $("#pulseRing");
  const liveVerdict     = $("#liveVerdict");
  const liveProb        = $("#liveProb");
  const liveLog         = $("#liveLog");

  let selectedFile = null;
  let wsConnection = null;
  let audioCtx     = null;
  let micStream    = null;
  let isRecording  = false;

  // ── Init ────────────────────────────────────────────────────────
  fetchStatus();

  // ── Tabs ────────────────────────────────────────────────────────
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const which = tab.dataset.tab;
      panelUpload.classList.toggle("hidden", which !== "upload");
      panelLive.classList.toggle("hidden", which !== "live");
    });
  });

  // ── File selection ──────────────────────────────────────────────
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
  });

  function pickFile(file) {
    if (!file) return;
    selectedFile = file;
    $(".upload-text").textContent = file.name;
    $(".upload-hint").textContent = `${(file.size / 1024).toFixed(1)} KB`;
    btnAnalyze.disabled = false;
    resultCard.classList.add("hidden");
  }

  // ── Upload & analyze ────────────────────────────────────────────
  btnAnalyze.addEventListener("click", analyzeFile);

  async function analyzeFile() {
    if (!selectedFile) return;
    btnAnalyze.disabled = true;
    resultCard.classList.add("hidden");
    progressWrap.classList.remove("hidden");
    progressFill.style.width = "20%";
    progressLabel.textContent = "Uploading…";

    const fd = new FormData();
    fd.append("audio", selectedFile);
    fd.append("model", modelSelect.value);

    try {
      progressFill.style.width = "50%";
      progressLabel.textContent = "Analyzing…";

      const res  = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await res.json();

      progressFill.style.width = "100%";
      progressLabel.textContent = "Done";

      if (data.error) {
        progressLabel.textContent = "Error: " + data.error;
        btnAnalyze.disabled = false;
        return;
      }

      showResult(data);
    } catch (err) {
      progressLabel.textContent = "Network error";
    }
    btnAnalyze.disabled = false;
  }

  function showResult(data) {
    resultCard.classList.remove("hidden");

    const badge = $("#verdictBadge");
    badge.textContent = data.overall_verdict;
    badge.className = "verdict-badge " + data.overall_verdict.toLowerCase();

    $("#verdictFile").textContent = data.filename;
    const dur = data.windows.length
      ? data.windows[data.windows.length - 1].time || data.windows[0].duration_sec
      : data.windows[0]?.duration_sec || "?";
    $("#verdictDuration").textContent = typeof dur === "number" ? dur.toFixed(1) + "s" : dur + "s";

    $("#statSpoof").textContent = (data.overall_spoof_prob * 100).toFixed(1) + "%";
    $("#statLatency").textContent = data.avg_latency_ms.toFixed(1) + " ms";
    $("#statWindows").textContent = data.windows.length;
    const avgRTF = data.windows.reduce((s, w) => s + w.rtf, 0) / data.windows.length;
    $("#statRTF").textContent = avgRTF.toFixed(4);

    // color the spoof stat
    $("#statSpoof").style.color = data.overall_spoof_prob >= 0.5 ? "var(--red)" : "var(--green)";

    drawChart(data.windows);
  }

  // ── Simple canvas chart ─────────────────────────────────────────
  function drawChart(windows) {
    const canvas = $("#chartCanvas");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    if (!windows.length) return;
    const probs = windows.map((r) => r.spoof_prob);
    const n = probs.length;
    const padX = 0, padY = 16;
    const plotW = w - padX * 2;
    const plotH = h - padY * 2;

    // threshold line
    ctx.strokeStyle = "rgba(245, 158, 11, .35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    const threshY = padY + plotH * (1 - 0.5);
    ctx.beginPath();
    ctx.moveTo(padX, threshY);
    ctx.lineTo(padX + plotW, threshY);
    ctx.stroke();
    ctx.setLineDash([]);

    // gradient fill
    const grad = ctx.createLinearGradient(0, padY, 0, padY + plotH);
    grad.addColorStop(0, "rgba(239,68,68,.18)");
    grad.addColorStop(0.5, "rgba(239,68,68,.04)");
    grad.addColorStop(1, "rgba(34,197,94,.08)");

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padX + (i / Math.max(n - 1, 1)) * plotW;
      const y = padY + plotH * (1 - probs[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    const lastX = padX + plotW;
    ctx.lineTo(lastX, padY + plotH);
    ctx.lineTo(padX, padY + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // line
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padX + (i / Math.max(n - 1, 1)) * plotW;
      const y = padY + plotH * (1 - probs[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "var(--accent)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // dots
    for (let i = 0; i < n; i++) {
      const x = padX + (i / Math.max(n - 1, 1)) * plotW;
      const y = padY + plotH * (1 - probs[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = probs[i] >= 0.5 ? "#ef4444" : "#22c55e";
      ctx.fill();
    }
  }

  // ── LIVE MICROPHONE ─────────────────────────────────────────────
  const liveData = [];
  const MAX_LIVE_POINTS = 120;

  btnLive.addEventListener("click", toggleLive);

  async function toggleLive() {
    if (isRecording) {
      stopLive();
      return;
    }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    } catch (e) {
      logLive("Microphone access denied: " + e.message);
      return;
    }

    isRecording = true;
    btnLive.textContent = "Stop";
    btnLive.classList.add("recording");
    pulseRing.classList.add("listening");
    liveVerdict.textContent = "LISTENING";
    liveVerdict.className = "live-verdict";
    liveProb.textContent = "waiting for audio…";
    liveData.length = 0;

    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(micStream);
    const processor = audioCtx.createScriptProcessor(8000, 1, 1);

    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    wsConnection = new WebSocket(wsProto + "//" + location.host + "/ws/live");
    wsConnection.binaryType = "arraybuffer";

    wsConnection.onopen = () => {
      wsConnection.send(JSON.stringify({ type: "config", model: liveModelSelect.value }));
      logLive("Connected — streaming audio to server");
    };

    wsConnection.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === "silence") {
        logLive(`[${msg.time.toFixed(1)}s] silence rms=${msg.rms.toFixed(4)}`);
        return;
      }
      if (msg.type === "result") {
        handleLiveResult(msg);
      }
    };

    wsConnection.onerror = () => logLive("WebSocket error");
    wsConnection.onclose = () => logLive("Disconnected");

    processor.onaudioprocess = (e) => {
      if (!isRecording || !wsConnection || wsConnection.readyState !== 1) return;
      const pcm = e.inputBuffer.getChannelData(0);
      wsConnection.send(new Float32Array(pcm).buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
  }

  function stopLive() {
    isRecording = false;
    btnLive.textContent = "Start Listening";
    btnLive.classList.remove("recording");
    pulseRing.className = "pulse-ring";
    liveVerdict.textContent = "STOPPED";
    liveVerdict.className = "live-verdict";

    if (wsConnection) { wsConnection.close(); wsConnection = null; }
    if (audioCtx)     { audioCtx.close(); audioCtx = null; }
    if (micStream)    { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  }

  function handleLiveResult(r) {
    const isSpoof = r.smoothed >= 0.5;
    const cls = isSpoof ? "spoof" : "bonafide";

    pulseRing.className = "pulse-ring " + cls;
    liveVerdict.textContent = isSpoof ? "SPOOF DETECTED" : "BONAFIDE";
    liveVerdict.className = "live-verdict " + cls;
    liveProb.textContent = `Spoof: ${(r.smoothed * 100).toFixed(1)}%  |  Raw: ${(r.spoof_prob * 100).toFixed(1)}%`;

    $("#liveLat").textContent  = r.latency_ms.toFixed(1) + " ms";
    $("#liveRTF").textContent  = r.rtf.toFixed(4);
    $("#liveRMS").textContent  = r.rms.toFixed(4);
    $("#liveTime").textContent = r.time.toFixed(1) + "s";

    liveData.push(r.smoothed);
    if (liveData.length > MAX_LIVE_POINTS) liveData.shift();
    drawLiveChart();

    logLive(`[${r.time.toFixed(1)}s] ${isSpoof ? "SPOOF" : "OK"}  prob=${r.spoof_prob.toFixed(3)}  smooth=${r.smoothed.toFixed(3)}  lat=${r.latency_ms.toFixed(0)}ms`);
  }

  function drawLiveChart() {
    const canvas = $("#liveChart");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const n = liveData.length;
    if (!n) return;
    const padY = 8;
    const plotH = h - padY * 2;

    // threshold
    ctx.strokeStyle = "rgba(245,158,11,.3)";
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    const thY = padY + plotH * 0.5;
    ctx.moveTo(0, thY);
    ctx.lineTo(w, thY);
    ctx.stroke();
    ctx.setLineDash([]);

    // line
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / Math.max(n - 1, 1)) * w;
      const y = padY + plotH * (1 - liveData[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function logLive(msg) {
    const el = liveLog;
    const line = document.createElement("div");
    line.textContent = msg;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
    if (el.childElementCount > 200) el.removeChild(el.firstChild);
  }

  // ── Status fetch ────────────────────────────────────────────────
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      $("#deviceBadge").textContent = data.gpu_name || data.device;
      $("#modelBadge").textContent = data.active_model || "none";
    } catch (e) {
      $("#deviceBadge").textContent = "offline";
    }
  }
})();
