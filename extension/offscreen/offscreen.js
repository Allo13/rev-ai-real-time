/**
 * RevealAI Offscreen Document
 *
 * Captures tab audio via getMediaStreamId, resamples to 16 kHz,
 * and streams PCM float32 chunks to the Flask backend via WebSocket.
 * Detection results are forwarded back to the background service worker.
 */

const BACKEND_WS = "ws://127.0.0.1:5000/ws/live";
const TARGET_SR = 16000;
const CHUNK_SIZE = 8192; // ~0.5s at 16 kHz (must be power of 2)

let ws = null;
let audioCtx = null;
let mediaStream = null;
let processor = null;
let isRunning = false;
let playbackAudio = null;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "offscreen_ping") {
    sendResponse({ ready: true });
    return;
  }
  if (msg.action === "offscreen_start") {
    startProcessing(msg.streamId);
  }
  if (msg.action === "offscreen_stop") {
    stopProcessing();
  }
});

function sendStatus(type, detail) {
  chrome.runtime.sendMessage({
    action: "detectionResult",
    type: type,
    error: typeof detail === "string" ? detail : undefined,
    message: typeof detail === "string" ? detail : undefined,
    ...( typeof detail === "object" ? detail : {})
  }).catch(() => {});
}

async function connectWebSocket() {
  return new Promise((resolve, reject) => {
    console.log("[RevealAI] Connecting to", BACKEND_WS);
    const socket = new WebSocket(BACKEND_WS);
    socket.binaryType = "arraybuffer";

    const timeout = setTimeout(() => {
      socket.close();
      reject(new Error("WebSocket connection timeout (120s)"));
    }, 120000);

    socket.onopen = () => {
      clearTimeout(timeout);
      console.log("[RevealAI] WebSocket connected!");
      socket.send(JSON.stringify({ type: "config", model: "v2_fast" }));
      resolve(socket);
    };

    socket.onerror = (e) => {
      clearTimeout(timeout);
      console.error("[RevealAI] WebSocket error:", e);
      reject(new Error("WebSocket connection failed"));
    };
  });
}

async function startProcessing(streamId) {
  if (isRunning) stopProcessing();

  console.log("[RevealAI] startProcessing called with streamId:", streamId ? streamId.substring(0, 20) + "..." : "null");

  // Step 1: Get tab audio stream
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId
        }
      },
      video: false
    });
    const tracks = mediaStream.getAudioTracks();
    console.log("[RevealAI] Got tab audio stream, tracks:", tracks.length,
      "settings:", tracks.length > 0 ? JSON.stringify(tracks[0].getSettings()) : "none");
    if (tracks.length === 0) {
      throw new Error("No audio tracks in captured stream");
    }
  } catch (e) {
    console.error("[RevealAI] Failed to get tab audio:", e);
    sendStatus("error", "Tab audio capture failed: " + e.message);
    return;
  }

  // Step 1b: Keep tab audio audible by playing the captured stream back via an Audio element
  try {
    playbackAudio = new Audio();
    playbackAudio.srcObject = mediaStream;
    await playbackAudio.play();
    console.log("[RevealAI] Audio playback started (tab audio remains audible)");
  } catch (e) {
    console.warn("[RevealAI] Audio playback failed (tab audio may be muted):", e.message);
  }

  // Step 2: Connect WebSocket
  try {
    ws = await connectWebSocket();
  } catch (e) {
    console.error("[RevealAI] WebSocket failed:", e);
    try {
      const res = await fetch("http://127.0.0.1:5000/api/status");
      sendStatus("error", "HTTP OK but WebSocket failed. Check backend console.");
    } catch (fetchErr) {
      sendStatus("error", "Backend not reachable. Run: venv\\Scripts\\python app.py");
    }
    stopProcessing();
    return;
  }

  // Step 3: Wire up WebSocket message handler
  ws.onmessage = (evt) => {
    try {
      const result = JSON.parse(evt.data);
      console.log("[RevealAI] Result:", result.type, result.verdict || "");
      chrome.runtime.sendMessage({ action: "detectionResult", ...result }).catch(() => {});
    } catch (e) {
      console.error("[RevealAI] Failed to parse WS message:", e);
    }
  };

  ws.onclose = (e) => {
    console.log("[RevealAI] WebSocket closed:", e.code, e.reason);
    if (isRunning) {
      sendStatus("error", "Backend disconnected (code " + e.code + ")");
    }
  };

  ws.onerror = (e) => {
    console.error("[RevealAI] WebSocket error:", e);
  };

  // Step 4: Set up audio processing
  try {
    // Create AudioContext at the target sample rate for analysis
    audioCtx = new AudioContext({ sampleRate: TARGET_SR });

    // AudioContext may start suspended in offscreen documents — must resume explicitly
    if (audioCtx.state === "suspended") {
      console.log("[RevealAI] AudioContext is suspended, resuming...");
      await audioCtx.resume();
    }
    console.log("[RevealAI] AudioContext state:", audioCtx.state, "sampleRate:", audioCtx.sampleRate);

    const source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(CHUNK_SIZE, 1, 1);

    let chunksSent = 0;
    processor.onaudioprocess = (e) => {
      if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
      const pcm = e.inputBuffer.getChannelData(0);
      const data = new Float32Array(pcm);
      ws.send(data.buffer);
      chunksSent++;
      if (chunksSent <= 5 || chunksSent % 20 === 0) {
        const rms = Math.sqrt(pcm.reduce((s, v) => s + v * v, 0) / pcm.length);
        console.log("[RevealAI] Sent chunk #" + chunksSent + " RMS:" + rms.toFixed(5) + " samples:" + data.length);
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
    isRunning = true;
    console.log("[RevealAI] Audio pipeline started! State:", audioCtx.state);
  } catch (e) {
    console.error("[RevealAI] Audio pipeline setup failed:", e);
    sendStatus("error", "Audio processing failed: " + e.message);
    stopProcessing();
  }
}

function stopProcessing() {
  isRunning = false;
  if (processor) { try { processor.disconnect(); } catch(e) {} processor = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  if (playbackAudio) { playbackAudio.pause(); playbackAudio.srcObject = null; playbackAudio = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (ws) { try { ws.close(); } catch(e) {} ws = null; }
  console.log("[RevealAI] Stopped");
}
