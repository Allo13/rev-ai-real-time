/**
 * RevealAI Content Script
 *
 * Injects a floating overlay on Google Meet / Zoom pages showing
 * real-time deepfake detection verdict, spoof probability, and latency.
 */

(function () {
  "use strict";

  let overlay = null;
  let historyData = [];
  const MAX_HISTORY = 60;

  function createOverlay() {
    if (overlay) return;

    overlay = document.createElement("div");
    overlay.id = "revealai-overlay";
    overlay.innerHTML = `
      <div class="revealai-header">
        <div class="revealai-logo">
          <svg width="18" height="18" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="13" stroke="#3b82f6" stroke-width="2"/>
            <path d="M8 14 L12 18 L20 10" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <span>RevealAI</span>
        </div>
        <button id="revealai-minimize" title="Minimize">—</button>
      </div>
      <div class="revealai-body" id="revealai-body">
        <div class="revealai-indicator" id="revealai-indicator">
          <div class="revealai-dot" id="revealai-dot"></div>
        </div>
        <div class="revealai-verdict" id="revealai-verdict">WAITING</div>
        <div class="revealai-prob" id="revealai-prob">—</div>
        <div class="revealai-stats">
          <div class="revealai-stat">
            <span class="revealai-stat-label">Latency</span>
            <span class="revealai-stat-value" id="revealai-lat">—</span>
          </div>
          <div class="revealai-stat">
            <span class="revealai-stat-label">RTF</span>
            <span class="revealai-stat-value" id="revealai-rtf">—</span>
          </div>
          <div class="revealai-stat">
            <span class="revealai-stat-label">Time</span>
            <span class="revealai-stat-value" id="revealai-time">—</span>
          </div>
        </div>
        <canvas id="revealai-chart" width="240" height="60"></canvas>
      </div>
    `;

    // Make it draggable
    let isDragging = false, offsetX, offsetY;
    overlay.querySelector(".revealai-header").addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON") return;
      isDragging = true;
      offsetX = e.clientX - overlay.getBoundingClientRect().left;
      offsetY = e.clientY - overlay.getBoundingClientRect().top;
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      overlay.style.left = (e.clientX - offsetX) + "px";
      overlay.style.top = (e.clientY - offsetY) + "px";
      overlay.style.right = "auto";
    });
    document.addEventListener("mouseup", () => { isDragging = false; });

    // Minimize toggle
    overlay.querySelector("#revealai-minimize").addEventListener("click", () => {
      const body = overlay.querySelector("#revealai-body");
      body.style.display = body.style.display === "none" ? "block" : "none";
    });

    document.body.appendChild(overlay);
  }

  function updateOverlay(data) {
    createOverlay();

    const isSpoof = data.smoothed >= 0.5;
    const verdict = isSpoof ? "SPOOF DETECTED" : "BONAFIDE";
    const cls = isSpoof ? "spoof" : "bonafide";

    const dot = document.getElementById("revealai-dot");
    const indicator = document.getElementById("revealai-indicator");
    const verdictEl = document.getElementById("revealai-verdict");
    const probEl = document.getElementById("revealai-prob");

    dot.className = "revealai-dot " + cls;
    indicator.className = "revealai-indicator " + cls;
    verdictEl.textContent = verdict;
    verdictEl.className = "revealai-verdict " + cls;
    probEl.textContent = (data.smoothed * 100).toFixed(1) + "% spoof probability";

    document.getElementById("revealai-lat").textContent = data.latency_ms.toFixed(0) + "ms";
    document.getElementById("revealai-rtf").textContent = data.rtf.toFixed(3);
    document.getElementById("revealai-time").textContent = data.time.toFixed(0) + "s";

    // Update chart
    historyData.push(data.smoothed);
    if (historyData.length > MAX_HISTORY) historyData.shift();
    drawMiniChart();
  }

  function handleSilence(data) {
    createOverlay();
    document.getElementById("revealai-verdict").textContent = "SILENCE";
    document.getElementById("revealai-verdict").className = "revealai-verdict";
  }

  function drawMiniChart() {
    const canvas = document.getElementById("revealai-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!historyData.length) return;

    // Threshold line
    ctx.strokeStyle = "rgba(245,158,11,0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data line
    ctx.beginPath();
    const n = historyData.length;
    for (let i = 0; i < n; i++) {
      const x = (i / Math.max(n - 1, 1)) * w;
      const y = h - historyData[i] * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Listen for messages from background
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "detectionResult") {
      if (msg.type === "silence") {
        handleSilence(msg);
      } else if (msg.type === "result") {
        updateOverlay(msg);
      }
    }
    if (msg.action === "detectionStopped") {
      if (overlay) {
        overlay.remove();
        overlay = null;
        historyData = [];
      }
    }
  });
})();
