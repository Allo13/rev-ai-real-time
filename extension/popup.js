/**
 * RevealAI Popup Script
 *
 * Controls start/stop of tab audio capture and shows live results.
 * The popup gets the stream ID directly (user gesture context) and
 * passes it to the background → offscreen pipeline.
 */

const BACKEND_URL = "http://127.0.0.1:5000";

const statusDot   = document.getElementById("statusDot");
const statusText  = document.getElementById("statusText");
const platformTag = document.getElementById("platformTag");
const btnToggle   = document.getElementById("btnToggle");
const resultSection = document.getElementById("resultSection");

let isCapturing = false;

init();

async function init() {
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
  if (isCapturing) {
    chrome.runtime.sendMessage({ action: "stopCapture" }, () => {
      isCapturing = false;
      btnToggle.textContent = "Start Detection";
      btnToggle.className = "btn btn-start";
      resultSection.classList.remove("visible");
    });
    return;
  }

  // Get stream ID directly from popup (preserves user gesture context)
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  try {
    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tab.id
    });

    // Pass stream ID to background to create offscreen and start processing
    chrome.runtime.sendMessage(
      { action: "startWithStream", tabId: tab.id, streamId },
      (res) => {
        if (res && res.ok) {
          isCapturing = true;
          btnToggle.textContent = "Stop Detection";
          btnToggle.className = "btn btn-stop";
          resultSection.classList.add("visible");
        } else {
          statusText.textContent = "Failed: " + (res?.error || "unknown");
        }
      }
    );
  } catch (e) {
    statusText.textContent = "Capture failed: " + e.message;
    statusText.style.fontSize = "10px";
  }
});

// Listen for live detection results
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.action !== "detectionResult") return;

  resultSection.classList.add("visible");

  if (msg.type === "info") {
    document.getElementById("resultVerdict").textContent = "LOADING...";
    document.getElementById("resultVerdict").className = "result-verdict";
    statusText.textContent = msg.message || "Loading...";
    statusText.style.fontSize = "10px";
    return;
  }

  if (msg.type === "error") {
    document.getElementById("resultVerdict").textContent = "ERROR";
    document.getElementById("resultVerdict").className = "result-verdict";
    statusText.textContent = msg.error || "Unknown error";
    statusText.style.fontSize = "10px";
    return;
  }

  if (msg.type === "silence") {
    document.getElementById("resultVerdict").textContent = "LISTENING...";
    document.getElementById("resultVerdict").className = "result-verdict";
    document.getElementById("rTime").textContent = (msg.time || 0).toFixed(0) + "s";
    return;
  }

  if (msg.type === "analyzing") {
    document.getElementById("resultVerdict").textContent = "ANALYZING... " + (msg.progress || 0) + "%";
    document.getElementById("resultVerdict").className = "result-verdict";
    document.getElementById("rTime").textContent = (msg.time || 0).toFixed(0) + "s";
    document.getElementById("rSpoof").textContent = "—";
    document.getElementById("rLat").textContent = "—";
    document.getElementById("rRTF").textContent = "—";
    return;
  }

  if (msg.type === "result") {
    const isSpoof = msg.spoof_prob >= 0.5;
    const verdict = isSpoof ? "SPOOF" : "BONAFIDE";
    const cls = isSpoof ? "spoof" : "bonafide";

    document.getElementById("resultVerdict").textContent = verdict;
    document.getElementById("resultVerdict").className = "result-verdict " + cls;
    document.getElementById("rSpoof").textContent = (msg.spoof_prob * 100).toFixed(1) + "%";
    document.getElementById("rLat").textContent = msg.latency_ms.toFixed(0) + "ms";
    document.getElementById("rRTF").textContent = msg.rtf.toFixed(3);
    document.getElementById("rTime").textContent = msg.time.toFixed(0) + "s";
  }
});
