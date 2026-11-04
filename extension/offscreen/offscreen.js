/**
 * RevealAI Offscreen Document
 *
 * Captures tab audio via getMediaStreamId, resamples to 16 kHz,
 * and streams PCM float32 chunks to the Flask backend via WebSocket.
 * Detection results are forwarded back to the background service worker.
 */

const BACKEND_WS = "ws://127.0.0.1:5000/ws/live";
const TARGET_SR = 16000;
const CHUNK_SIZE = 8000; // 0.5s at 16 kHz

let ws = null;
let audioCtx = null;
let mediaStream = null;
let processor = null;
let isRunning = false;

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.action === "offscreen_start") {
    startProcessing(msg.streamId);
  }
  if (msg.action === "offscreen_stop") {
    stopProcessing();
  }
});

async function startProcessing(streamId) {
  if (isRunning) stopProcessing();

  try {
    // Get the actual MediaStream from the stream ID
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId
        }
      }
    });
  } catch (e) {
    console.error("[RevealAI] Failed to get tab media stream:", e);
    return;
  }

  // Connect to Flask WebSocket backend
  ws = new WebSocket(BACKEND_WS);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("[RevealAI] WebSocket connected to backend");
    ws.send(JSON.stringify({ type: "config", model: "run_b_8" }));
  };

  ws.onmessage = (evt) => {
    try {
      const result = JSON.parse(evt.data);
      // Forward detection result to background -> content script
      chrome.runtime.sendMessage({
        action: "detectionResult",
        ...result
      });
    } catch (e) {}
  };

  ws.onerror = (e) => console.error("[RevealAI] WebSocket error:", e);
  ws.onclose = () => console.log("[RevealAI] WebSocket disconnected");

  // Set up audio processing at 16 kHz
  audioCtx = new AudioContext({ sampleRate: TARGET_SR });
  const source = audioCtx.createMediaStreamSource(mediaStream);

  processor = audioCtx.createScriptProcessor(CHUNK_SIZE, 1, 1);
  processor.onaudioprocess = (e) => {
    if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = e.inputBuffer.getChannelData(0);
    ws.send(new Float32Array(pcm).buffer);
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);
  isRunning = true;

  console.log("[RevealAI] Audio capture started");
}

function stopProcessing() {
  isRunning = false;

  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }

  console.log("[RevealAI] Audio capture stopped");
}
