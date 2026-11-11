/**
 * RevealAI Background Service Worker
 *
 * Manages offscreen document lifecycle and relays detection results.
 * The popup handles tabCapture.getMediaStreamId (needs user gesture),
 * then sends the stream ID here to create the offscreen doc.
 */

let offscreenCreated = false;
let activeTabId = null;
let isCapturing = false;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "startWithStream") {
    startWithStream(msg.tabId, msg.streamId)
      .then(() => sendResponse({ ok: true }))
      .catch(e => sendResponse({ ok: false, error: e.message }));
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

  // Forward results from offscreen doc to the content script and popup
  if (msg.action === "detectionResult" && activeTabId) {
    chrome.tabs.sendMessage(activeTabId, msg).catch(() => {});
    // Re-broadcast so popup can pick it up (sendMessage goes to all extension pages except sender)
    return false;
  }

  return false;
});

async function startWithStream(tabId, streamId) {
  // Stop any existing capture
  if (isCapturing || offscreenCreated) {
    await stopCapture();
    await new Promise(r => setTimeout(r, 300));
  }

  activeTabId = tabId;

  // Create offscreen document
  try {
    await chrome.offscreen.createDocument({
      url: "offscreen/offscreen.html",
      reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
      justification: "Capture tab audio for deepfake detection and play it back"
    });
    offscreenCreated = true;
  } catch (e) {
    if (e.message.includes("single offscreen")) {
      offscreenCreated = true;
    } else {
      throw e;
    }
  }

  // Wait for offscreen document to be ready
  for (let i = 0; i < 20; i++) {
    try {
      const resp = await chrome.runtime.sendMessage({ action: "offscreen_ping" });
      if (resp && resp.ready) break;
    } catch (e) {}
    await new Promise(r => setTimeout(r, 100));
  }

  // Tell offscreen document to start processing with the stream ID
  chrome.runtime.sendMessage({
    action: "offscreen_start",
    streamId,
    tabId: activeTabId
  });

  isCapturing = true;
  chrome.action.setBadgeText({ text: "ON" });
  chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
}

async function stopCapture() {
  try {
    chrome.runtime.sendMessage({ action: "offscreen_stop" });
  } catch (e) {}

  if (offscreenCreated) {
    try {
      await chrome.offscreen.closeDocument();
    } catch (e) {}
    offscreenCreated = false;
  }

  isCapturing = false;
  const prevTab = activeTabId;
  activeTabId = null;

  chrome.action.setBadgeText({ text: "" });

  if (prevTab) {
    try {
      await chrome.tabs.sendMessage(prevTab, { action: "detectionStopped" });
    } catch (e) {}
  }
}
