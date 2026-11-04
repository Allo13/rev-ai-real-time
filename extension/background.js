/**
 * RevealAI Background Service Worker
 *
 * Manages tab audio capture via the Offscreen API and relays
 * audio data between the offscreen document and the content script overlay.
 */

let offscreenCreated = false;
let activeTabId = null;
let isCapturing = false;

// Listen for messages from popup and content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "startCapture") {
    startCapture(msg.tabId).then(() => sendResponse({ ok: true })).catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }

  if (msg.action === "stopCapture") {
    stopCapture().then(() => sendResponse({ ok: true }));
    return true;
  }

  if (msg.action === "getStatus") {
    sendResponse({ isCapturing, activeTabId });
    return false;
  }

  // Forward results from offscreen doc to the content script on the active tab
  if (msg.action === "detectionResult" && activeTabId) {
    chrome.tabs.sendMessage(activeTabId, msg).catch(() => {});
    return false;
  }

  // Forward results to popup as well
  if (msg.action === "detectionResult") {
    // Popup listens via runtime.onMessage, it will pick this up
    return false;
  }
});

async function startCapture(tabId) {
  activeTabId = tabId || (await getActiveTab()).id;

  // Get a MediaStream from the tab's audio
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: activeTabId });

  // Create offscreen document if not already present
  if (!offscreenCreated) {
    await chrome.offscreen.createDocument({
      url: "offscreen/offscreen.html",
      reasons: ["USER_MEDIA"],
      justification: "Capture tab audio for deepfake detection processing"
    });
    offscreenCreated = true;
  }

  // Tell offscreen document to start processing with the stream ID
  chrome.runtime.sendMessage({ action: "offscreen_start", streamId, tabId: activeTabId });
  isCapturing = true;

  // Update badge
  chrome.action.setBadgeText({ text: "ON" });
  chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
}

async function stopCapture() {
  chrome.runtime.sendMessage({ action: "offscreen_stop" });
  isCapturing = false;
  activeTabId = null;

  chrome.action.setBadgeText({ text: "" });

  // Notify content script to hide overlay
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { action: "detectionStopped" }).catch(() => {});
    }
  });
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}
