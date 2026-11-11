"""
Flask backend for Reveal AI — Real-Time Audio Deepfake Detection.

Endpoints:
  POST /signup         → register new user
  POST /token          → login (returns access_token)
  GET  /api/status     → backend health (for Chrome extension)
  GET  /models         → model registry
  POST /detect         → file upload deepfake detection
  GET  /test-samples   → list available test samples
  GET  /test-samples/<name> → download a test sample
  WS   /ws/stream      → live mic streaming (frontend)
  WS   /ws/live        → live tab audio streaming (Chrome extension)
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
import time
import json
import hashlib
import tempfile
import threading
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, request, jsonify, send_file
from flask_sock import Sock
from flask_cors import CORS

from models.truncated_xlsr import TruncatedXLSR, load_xlsr_300m
from models.fake_mamba import FakeMambaClassifier, FakeMambaModel
from dataset.asvspoof_loader import TARGET_SR

RUN_CONFIGS = {
    # Legacy XLSR+Mamba models
    "baseline":  {"num_layers": 24, "checkpoint": "./checkpoints/baseline/best_model.pt", "arch": "xlsr_mamba"},
    "run_a_12":  {"num_layers": 12, "checkpoint": "./checkpoints/run_a_12/best_model.pt", "arch": "xlsr_mamba"},
    "run_b_8":   {"num_layers": 8,  "checkpoint": "./checkpoints/run_b_8/best_model.pt", "arch": "xlsr_mamba"},
    # v2: wav2vec2-base + AASIST (fast, real-time)
    "v2_fast":   {"num_layers": 12, "checkpoint": "./checkpoints/v2_fast/best_model.pt", "arch": "w2v2_aasist"},
    # v2: XLSR-300M + AASIST (accurate)
    "v2_accurate": {"num_layers": 12, "checkpoint": "./checkpoints/v2_accurate/best_model.pt", "arch": "xlsr_aasist"},
}

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def _load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"admin": "revealai2026", "demo": "demo"}

def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def _make_token(username):
    return hashlib.sha256(f"revealai-{username}-token".encode()).hexdigest()

app = Flask(__name__)
CORS(app)
sock = Sock(app)

_model = None
_device = None
_active_run = None
_model_lock = threading.Lock()


def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_mem >= 2.0:
                _device = torch.device("cuda")
            else:
                print(f"[Reveal AI] GPU has only {free_mem:.1f}GB free, using CPU")
                _device = torch.device("cpu")
        else:
            _device = torch.device("cpu")
    return _device


def load_model(run_name="v2_fast", xlsr_path=None):
    global _model, _active_run
    with _model_lock:
        if _model is not None and _active_run == run_name:
            return _model

        import gc
        device = get_device()
        cfg = RUN_CONFIGS[run_name]
        num_layers = cfg["num_layers"]
        ckpt_path = cfg["checkpoint"]
        arch = cfg.get("arch", "xlsr_mamba")

        state = None
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            num_layers = state.get("num_layers", num_layers)

        if arch == "xlsr_mamba":
            base_xlsr = load_xlsr_300m(checkpoint_path=xlsr_path)
            front_end = TruncatedXLSR(base_xlsr_model=base_xlsr, num_layers=num_layers)
            del base_xlsr; gc.collect()
            back_end = FakeMambaClassifier(input_dim=1024, hidden_dim=144, num_mamba_layers=2, num_classes=2)
            model = FakeMambaModel(front_end=front_end, back_end=back_end)

        elif arch == "w2v2_aasist":
            from models.ssl_frontend import load_wav2vec2_base, TruncatedSSL
            from models.aasist_backend import AASISTBackend, SSLAntiSpoofModel
            base = load_wav2vec2_base()
            front_end = TruncatedSSL(base, num_layers=num_layers)
            del base; gc.collect()
            back_end = AASISTBackend(input_dim=768, hidden_dim=160, num_heads=4, num_classes=2)
            model = SSLAntiSpoofModel(front_end, back_end)

        elif arch == "xlsr_aasist":
            from models.ssl_frontend import load_xlsr_large, TruncatedSSL
            from models.aasist_backend import AASISTBackend, SSLAntiSpoofModel
            base = load_xlsr_large()
            front_end = TruncatedSSL(base, num_layers=num_layers)
            del base; gc.collect()
            back_end = AASISTBackend(input_dim=1024, hidden_dim=256, num_heads=4, num_classes=2)
            model = SSLAntiSpoofModel(front_end, back_end)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        if state is not None:
            model.load_state_dict(state["model_state_dict"])
            del state; gc.collect()

        model = model.to(device).eval()
        gc.collect()
        with torch.no_grad():
            model(torch.zeros(1, TARGET_SR, device=device))

        _model = model
        _active_run = run_name
        return _model


def score_audio(waveform: np.ndarray, model: nn.Module, device: torch.device):
    x = waveform / (np.max(np.abs(waveform)) + 1e-8)
    source = torch.from_numpy(x).float().unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(source)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=1)[0]
    spoof_prob = float(probs[1].item())
    bonafide_prob = float(probs[0].item())

    return {
        "spoof_prob": spoof_prob,
        "bonafide_prob": bonafide_prob,
        "latency": latency,
    }


# ── Auth ─────────────────────────────────────────────────────────────────────

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if not username or not password:
        return jsonify({"detail": "Username and password are required"}), 400
    if len(username) < 3:
        return jsonify({"detail": "Username must be at least 3 characters"}), 400
    if len(password) < 4:
        return jsonify({"detail": "Password must be at least 4 characters"}), 400

    users = _load_users()
    if username in users:
        return jsonify({"detail": "Username already taken"}), 409

    users[username] = password
    _save_users(users)
    token = _make_token(username)
    return jsonify({"access_token": token, "token_type": "bearer", "username": username})


@app.route("/token", methods=["POST"])
def token():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    users = _load_users()
    if users.get(username) == password:
        return jsonify({"access_token": _make_token(username), "token_type": "bearer"})
    return jsonify({"detail": "Invalid credentials"}), 401


# ── Status (for Chrome extension) ───────────────────────────────────────────

@app.route("/api/status")
def api_status():
    device_str = str(_device) if _device else ("cuda" if torch.cuda.is_available() else "cpu")
    return jsonify({
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_model": _active_run,
        "models": {k: {"layers": v["num_layers"], "trained": os.path.exists(v["checkpoint"])} for k, v in RUN_CONFIGS.items()},
    })


# ── Models registry ──────────────────────────────────────────────────────────

@app.route("/models")
def models():
    registry = {}
    registry["xlsr_mamba"] = {
        "available": any(os.path.exists(v["checkpoint"]) for v in RUN_CONFIGS.values())
    }
    return jsonify(registry)


# ── Detection ────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"detail": "No audio file provided"}), 400

    f = request.files["file"]

    suffix = os.path.splitext(f.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    f.save(tmp_path)
    tmp.close()

    try:
        import librosa
        waveform, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        return jsonify({"detail": f"Could not read audio: {e}"}), 422
    finally:
        os.unlink(tmp_path)

    waveform = waveform.astype(np.float32)
    audio_hash = hashlib.sha256(waveform.tobytes()).hexdigest()

    model = load_model("v2_fast")
    device = get_device()

    total_samples = len(waveform)
    window_samples = TARGET_SR * 4

    if total_samples < TARGET_SR:
        waveform = np.pad(waveform, (0, TARGET_SR - total_samples))

    if total_samples <= window_samples:
        r = score_audio(waveform, model, device)
        spoof_prob = r["spoof_prob"]
        latency = r["latency"]
    else:
        hop = TARGET_SR // 2
        results = []
        for start in range(0, total_samples - window_samples + 1, hop):
            chunk = waveform[start:start + window_samples]
            results.append(score_audio(chunk, model, device))
        spoof_prob = float(np.mean([r["spoof_prob"] for r in results]))
        latency = float(np.mean([r["latency"] for r in results]))

    label = "fake" if spoof_prob >= 0.5 else "real"
    confidence = spoof_prob if label == "fake" else 1.0 - spoof_prob

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "model": "XLSR-300M + Fake-Mamba (Reveal AI)",
        "latency_ms": round(latency * 1000, 1),
        "audio_hash": audio_hash,
    })


# ── Test samples ─────────────────────────────────────────────────────────────

@app.route("/test-samples")
def test_samples():
    """List .flac files from the ASVspoof eval set (not used in training)."""
    eval_proto = "./dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    eval_dir = "./dataset/LA/LA/ASVspoof2019_LA_eval/flac"

    samples = []
    if os.path.exists(eval_proto) and os.path.isdir(eval_dir):
        with open(eval_proto) as f:
            lines = f.readlines()
        bonafide = [l for l in lines if "bonafide" in l][:10]
        spoof = [l for l in lines if "spoof" in l][:10]
        for line in bonafide + spoof:
            parts = line.strip().split()
            utt_id = parts[1]
            label = parts[-1]
            filepath = os.path.join(eval_dir, utt_id + ".flac")
            if os.path.exists(filepath):
                samples.append({
                    "id": utt_id,
                    "label": "real" if label == "bonafide" else "fake",
                    "attack": parts[3] if len(parts) > 3 else "-",
                })
    return jsonify(samples)


@app.route("/test-samples/<name>")
def get_test_sample(name):
    safe_name = os.path.basename(name)
    filepath = os.path.join("./dataset/LA/LA/ASVspoof2019_LA_eval/flac", safe_name + ".flac")
    if not os.path.exists(filepath):
        return jsonify({"detail": "Sample not found"}), 404
    return send_file(filepath, mimetype="audio/flac", as_attachment=True, download_name=safe_name + ".flac")


# ── Synthesis engines stub (returns empty list so Generate tab loads) ────────

@app.route("/synthesis/engines")
def synthesis_engines():
    return jsonify([])

@app.route("/synthesize", methods=["POST"])
def synthesize_stub():
    return jsonify({"detail": "Speech synthesis not available — this is a detection-only build"}), 501

@app.route("/watermark/verify", methods=["POST"])
def watermark_verify():
    return jsonify({"detail": "Watermark verification not available — this is a detection-only build"}), 501

@app.route("/forensic/report", methods=["POST"])
def forensic_report():
    return jsonify({"detail": "Forensic reports not available in this build"}), 501


# ── Live WebSocket streaming (frontend Live tab) ────────────────────────────

@sock.route("/ws/stream")
def ws_stream(ws):
    authenticated = False
    model_obj = load_model("v2_fast")
    device = get_device()

    buffer = np.zeros(0, dtype=np.float32)
    window_id = 0
    min_samples = TARGET_SR * 3
    window_samples = TARGET_SR * 4

    while True:
        try:
            data = ws.receive(timeout=15)
        except Exception:
            break
        if data is None:
            break

        if not authenticated:
            try:
                msg = json.loads(data)
                token = msg.get("token", "")
                if token and len(token) > 0:
                    authenticated = True
                    ws.send(json.dumps({"type": "auth_ok"}))
                else:
                    ws.send(json.dumps({"type": "error", "detail": "bad token"}))
                    break
            except (json.JSONDecodeError, TypeError):
                break
            continue

        if isinstance(data, bytes):
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            continue

        buffer = np.concatenate([buffer, samples])

        if len(buffer) < min_samples:
            continue

        window = buffer[-window_samples:] if len(buffer) >= window_samples else buffer
        r = score_audio(window, model_obj, device)
        window_id += 1

        sp = r["spoof_prob"]
        label = "fake" if sp >= 0.5 else "real"
        confidence = sp if label == "fake" else 1.0 - sp
        is_final = len(buffer) >= TARGET_SR * 60

        ws.send(json.dumps({
            "timestamp_ms": int(time.time() * 1000),
            "window_id": window_id,
            "label": label,
            "confidence": round(confidence, 4),
            "model": "xlsr_mamba",
            "final": is_final,
        }))

        if is_final:
            break

        min_samples = len(buffer) + TARGET_SR * 2

    ws.close()


# ── Live WebSocket for Chrome extension (float32 PCM, no auth) ──────────────

@sock.route("/ws/live")
def ws_live(ws):
    print("[ws/live] New connection")
    model_obj = load_model("v2_fast")
    device = get_device()
    print("[ws/live] Model ready, listening for audio...")

    buffer = np.zeros(0, dtype=np.float32)
    stream_time = 0.0
    chunks_received = 0
    window_samples = TARGET_SR * 4
    VERDICT_PERIOD = 10.0
    scores_window = []
    last_verdict_time = 0.0

    while True:
        try:
            data = ws.receive(timeout=30)
        except Exception as e:
            print(f"[ws/live] Receive error after {chunks_received} chunks: {e}")
            break
        if data is None:
            print(f"[ws/live] Connection closed by client after {chunks_received} chunks")
            break

        if isinstance(data, str):
            print(f"[ws/live] Got text message: {data[:100]}")
            try:
                msg = json.loads(data)
                if msg.get("type") == "config":
                    continue
            except json.JSONDecodeError:
                continue
            continue

        chunk = np.frombuffer(data, dtype=np.float32)
        chunks_received += 1
        buffer = np.concatenate([buffer, chunk])
        stream_time += len(chunk) / TARGET_SR

        if chunks_received <= 3:
            rms_chunk = float(np.sqrt(np.mean(chunk ** 2)))
            print(f"[ws/live] Chunk #{chunks_received}: {len(chunk)} samples, RMS={rms_chunk:.5f}, buffer={len(buffer)}")

        if len(buffer) < TARGET_SR:
            continue

        window = buffer[-window_samples:] if len(buffer) >= window_samples else buffer

        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < 0.001:
            ws.send(json.dumps({"type": "silence", "time": round(stream_time, 2), "rms": round(rms, 5)}))
            if chunks_received <= 10:
                print(f"[ws/live] Silence (RMS={rms:.5f})")
            continue

        result = score_audio(window, model_obj, device)
        sp = result["spoof_prob"]
        scores_window.append(sp)

        time_since_verdict = stream_time - last_verdict_time
        if time_since_verdict < VERDICT_PERIOD:
            progress_pct = min(100, int(time_since_verdict / VERDICT_PERIOD * 100))
            ws.send(json.dumps({
                "type": "analyzing",
                "progress": progress_pct,
                "time": round(stream_time, 2),
                "samples_collected": len(scores_window),
            }))
            continue

        avg_spoof = float(np.mean(scores_window))
        max_spoof = float(np.max(scores_window))
        median_spoof = float(np.median(scores_window))
        verdict = "SPOOF" if avg_spoof >= 0.5 else "BONAFIDE"

        verdict_data = {
            "type": "result",
            "spoof_prob": round(avg_spoof, 4),
            "bonafide_prob": round(1.0 - avg_spoof, 4),
            "verdict": verdict,
            "latency_ms": round(result["latency"] * 1000, 2),
            "rtf": round(result["latency"] / (len(window) / TARGET_SR), 6),
            "time": round(stream_time, 2),
            "max_spoof": round(max_spoof, 4),
            "median_spoof": round(median_spoof, 4),
            "num_samples": len(scores_window),
            "rms": round(rms, 5),
        }
        ws.send(json.dumps(verdict_data))
        print(f"[ws/live] VERDICT after {len(scores_window)} samples → {verdict} (avg={avg_spoof:.3f}, max={max_spoof:.3f}, median={median_spoof:.3f})")

        scores_window = []
        last_verdict_time = stream_time

    print(f"[ws/live] Session ended. Total chunks: {chunks_received}, stream time: {stream_time:.1f}s")
    ws.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default="run_b_8", choices=list(RUN_CONFIGS))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"[Reveal AI] Pre-loading model '{args.model}'...")
    try:
        load_model(args.model)
        print(f"[Reveal AI] Model loaded successfully!")
    except Exception as e:
        print(f"[Reveal AI] Model pre-load failed ({e}), will retry on first request")
    print(f"[Reveal AI] Server starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
