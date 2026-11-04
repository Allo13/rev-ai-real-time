/**
 * RevealAI Popup Script
 *
 * Controls start/stop of tab audio capture and shows live results.
 */

const BACKEND_URL = "http://127.0.0.1:5000";

const statusDot   = document.getElementById("statusDot");
const statusText  = document.getElementById("statusText");
const platformTag = document.getElementById("platformTag");
const btnToggle   = document.getElementById("btnToggle");
const resultSection = document.getElementById("resultSection");

let isCapturing = false;

// Check backend and tab on popup open
init();

async function init() {
  // Check backend health
  let backendOk = false;
  try {
    const res = await fetch(BACKEND_URL + "/api/status", { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      backendOk = true;
      const data = await res.json();
      statusText.textContent = `Backend: ${data.device} | Model: ${data.active_model || "loading"}`;
    }
  } catch (e) {
    statusText.textContent = "Backend offline — start app.py first";
    statusDot.classList.add("error");
    return;
  }

  statusDot.classList.add("connected");

  // Detect platform from current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url) {
    if (tab.url.includes("meet.google.com")) {
      platformTag.textContent = "GOOGLE MEET";
      platformTag.className = "platform-tag meet";
      platformTag.style.display = "";
    } else if (tab.url.includes("zoom.us")) {
      platformTag.textContent = "ZOOM";
      platformTag.className = "platform-tag zoom";
      platformTag.style.display = "";
    } else {
      platformTag.textContent = "TAB AUDIO";
      platformTag.className = "platform-tag other";
      platformTag.style.display = "";
    }
  }

  // Check if already capturing
  chrome.runtime.sendMessage({ action: "getStatus" }, (res) => {
    if (res && res.isCapturing) {
      isCapturing = true;
      btnToggle.textContent = "Stop Detection";
      btnToggle.className = "btn btn-stop";
      resultSection.classList.add("visible");
    }
    btnToggle.disabled = !backendOk;
  });
}

btnToggle.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (isCapturing) {
    chrome.runtime.sendMessage({ action: "stopCapture" }, () => {
      isCapturing = false;
      btnToggle.textContent = "Start Detection";
      btnToggle.className = "btn btn-start";
      resultSection.classList.remove("visible");
    });
  } else {
    chrome.runtime.sendMessage({ action: "startCapture", tabId: tab.id }, (res) => {
      if (res && res.ok) {
        isCapturing = true;
        btnToggle.textContent = "Stop Detection";
        btnToggle.className = "btn btn-stop";
        resultSection.classList.add("visible");
      } else {
        statusText.textContent = "Capture failed: " + (res?.error || "unknown");
      }
    });
  }
});

// Listen for live detection results
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.action !== "detectionResult") return;

  resultSection.classList.add("visible");

  if (msg.type === "silence") {
    document.getElementById("resultVerdict").textContent = "SILENCE";
    document.getElementById("resultVerdict").className = "result-verdict";
    return;
  }

  if (msg.type === "result") {
    const isSpoof = msg.smoothed >= 0.5;
    const verdict = isSpoof ? "SPOOF" : "BONAFIDE";
    const cls = isSpoof ? "spoof" : "bonafide";

    document.getElementById("resultVerdict").textContent = verdict;
    document.getElementById("resultVerdict").className = "result-verdict " + cls;
    document.getElementById("rSpoof").textContent = (msg.smoothed * 100).toFixed(1) + "%";
    document.getElementById("rLat").textContent = msg.latency_ms.toFixed(0) + "ms";
    document.getElementById("rRTF").textContent = msg.rtf.toFixed(3);
    document.getElementById("rTime").textContent = msg.time.toFixed(0) + "s";
  }
});
