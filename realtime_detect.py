import os
import sys
import time
import csv
import queue
import argparse
import threading
from collections import deque
from typing import Optional, Iterator

import numpy as np
import torch
import torch.nn as nn

from models.truncated_xlsr import TruncatedXLSR, load_xlsr_300m
from models.fake_mamba import FakeMambaClassifier, FakeMambaModel
from dataset.asvspoof_loader import TARGET_SR

# Default checkpoint locations per ablation run (as written by train_ablation.py)
RUN_CONFIGS = {
    "baseline": {"num_layers": 24, "checkpoint": "./checkpoints/baseline/best_model.pt"},
    "run_a_12": {"num_layers": 12, "checkpoint": "./checkpoints/run_a_12/best_model.pt"},
    "run_b_8":  {"num_layers": 8,  "checkpoint": "./checkpoints/run_b_8/best_model.pt"},
}


def build_model(run_name: str, checkpoint_path: Optional[str], xlsr_path: Optional[str], device: torch.device) -> nn.Module:
    """
    Rebuilds the TruncatedXLSR + FakeMamba model and loads trained weights if available.
    """
    cfg = RUN_CONFIGS[run_name]
    num_layers = cfg["num_layers"]
    checkpoint_path = checkpoint_path or cfg["checkpoint"]

    state = None
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        num_layers = state.get("num_layers", num_layers)
    else:
        print(f"[realtime] Warning: checkpoint '{checkpoint_path}' not found. "
              "Running with UNTRAINED weights - predictions are meaningless.")

    base_xlsr = load_xlsr_300m(checkpoint_path=xlsr_path)
    front_end = TruncatedXLSR(base_xlsr_model=base_xlsr, num_layers=num_layers)
    back_end = FakeMambaClassifier(input_dim=1024, hidden_dim=144, num_mamba_layers=2, num_classes=2)
    model = FakeMambaModel(front_end=front_end, back_end=back_end)

    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        print(f"[realtime] Loaded '{state.get('run_name', run_name)}' checkpoint "
              f"(epoch {state.get('epoch')}, val EER {state.get('val_eer', float('nan')):.2f}%)")

    return model.to(device).eval()


def file_chunk_stream(path: str, chunk_samples: int, simulate_realtime: bool) -> Iterator[np.ndarray]:
    """
    Yields a 16 kHz mono audio file in fixed-size chunks, optionally paced at real-time speed.
    """
    import librosa
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    chunk_sec = chunk_samples / TARGET_SR
    for start in range(0, len(audio), chunk_samples):
        t0 = time.perf_counter()
        yield audio[start:start + chunk_samples].astype(np.float32)
        if simulate_realtime:
            time.sleep(max(0.0, chunk_sec - (time.perf_counter() - t0)))


def mic_chunk_stream(chunk_samples: int, device_index: Optional[int], stop_event: threading.Event) -> Iterator[np.ndarray]:
    """
    Yields microphone audio in fixed-size chunks captured at 16 kHz mono via sounddevice.
    """
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("[realtime] Microphone input requires 'sounddevice' (pip install sounddevice).")

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[realtime] Audio status: {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                        blocksize=chunk_samples, device=device_index, callback=callback):
        print("[realtime] Listening on microphone... press Ctrl+C to stop.")
        while not stop_event.is_set():
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                continue


class LiveAnalyzer:
    """
    Sliding-window live spoof detector.

    Keeps a rolling buffer of the last `window_sec` seconds, scores it every `hop_sec`,
    and smooths the spoof probability with an exponential moving average.
    """
    def __init__(self, model: nn.Module, device: torch.device, window_sec: float = 4.0,
                 threshold: float = 0.5, ema_alpha: float = 0.6, silence_rms: float = 0.005):
        self.model = model
        self.device = device
        self.window_samples = int(window_sec * TARGET_SR)
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.silence_rms = silence_rms
        self.buffer = np.zeros(0, dtype=np.float32)
        self.smoothed = None
        self.history = []

    def push(self, chunk: np.ndarray, stream_time: float) -> Optional[dict]:
        self.buffer = np.concatenate([self.buffer, chunk])[-self.window_samples:]
        window = self.buffer
        # Need at least 1s of audio before scoring
        if len(window) < TARGET_SR:
            return None

        rms = float(np.sqrt(np.mean(window[-len(chunk):] ** 2)))
        if rms < self.silence_rms:
            result = {"time": stream_time, "rms": rms, "silent": True,
                      "spoof_prob": None, "smoothed": self.smoothed, "latency_ms": 0.0, "rtf": 0.0}
            self.history.append(result)
            return result

        # Peak-normalise like offline loading so scores are comparable
        x = window / (np.max(np.abs(window)) + 1e-8)
        source = torch.from_numpy(x).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(source)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        # Class 0 = bonafide, class 1 = spoof
        spoof_prob = float(torch.softmax(logits, dim=1)[0, 1].item())
        self.smoothed = spoof_prob if self.smoothed is None else (
            self.ema_alpha * spoof_prob + (1 - self.ema_alpha) * self.smoothed)

        result = {"time": stream_time, "rms": rms, "silent": False,
                  "spoof_prob": spoof_prob, "smoothed": self.smoothed,
                  "latency_ms": latency * 1000.0,
                  "rtf": latency / (len(window) / TARGET_SR)}
        self.history.append(result)
        return result

    def verdict(self, prob: Optional[float]) -> str:
        if prob is None:
            return "  ...  "
        return "SPOOF  " if prob >= self.threshold else "BONAFIDE"

    def summary(self) -> dict:
        scored = [h for h in self.history if not h["silent"]]
        if not scored:
            return {"windows": 0}
        probs = np.array([h["spoof_prob"] for h in scored])
        lat = np.array([h["latency_ms"] for h in scored])
        return {
            "windows": len(scored),
            "silent_windows": len(self.history) - len(scored),
            "mean_spoof_prob": float(probs.mean()),
            "spoof_window_ratio": float((probs >= self.threshold).mean()),
            "final_verdict": self.verdict(float(probs.mean())).strip(),
            "avg_latency_ms": float(lat.mean()),
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "avg_rtf": float(np.mean([h["rtf"] for h in scored])),
        }


def print_result(analyzer: LiveAnalyzer, r: dict):
    if r["silent"]:
        print(f"[{r['time']:7.2f}s] (silence)  rms={r['rms']:.4f}")
        return
    bar_len = 30
    filled = int(round(r["smoothed"] * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"[{r['time']:7.2f}s] {analyzer.verdict(r['smoothed'])} |{bar}| "
          f"spoof={r['spoof_prob']:.3f} smooth={r['smoothed']:.3f} "
          f"lat={r['latency_ms']:.1f}ms RTF={r['rtf']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Live (real-time) audio deepfake detection with Truncated XLSR + Fake-Mamba")
    parser.add_argument("--source", choices=["mic", "file"], default="mic", help="Audio input source")
    parser.add_argument("--file", type=str, default=None, help="Audio file to stream when --source file")
    parser.add_argument("--no_pacing", action="store_true", help="Stream file as fast as possible instead of real-time speed")
    parser.add_argument("--run", choices=list(RUN_CONFIGS), default="run_b_8", help="Ablation model to load")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override path to best_model.pt")
    parser.add_argument("--xlsr_path", type=str, default=None, help="Path to Fairseq XLSR-300M weights")
    parser.add_argument("--window", type=float, default=4.0, help="Analysis window length (seconds)")
    parser.add_argument("--hop", type=float, default=0.5, help="Seconds between successive analyses")
    parser.add_argument("--threshold", type=float, default=0.5, help="Spoof probability decision threshold")
    parser.add_argument("--ema", type=float, default=0.6, help="EMA smoothing factor (1.0 = no smoothing)")
    parser.add_argument("--silence_rms", type=float, default=0.005, help="RMS below which a hop is treated as silence")
    parser.add_argument("--mic_device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--log_csv", type=str, default=None, help="Write per-window results to a CSV file")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu' or 'auto'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"[realtime] Compute device: {device}")

    model = build_model(args.run, args.checkpoint, args.xlsr_path, device)
    analyzer = LiveAnalyzer(model, device, window_sec=args.window, threshold=args.threshold,
                            ema_alpha=args.ema, silence_rms=args.silence_rms)

    # Warm-up so the first live window isn't dominated by CUDA/kernel init
    with torch.no_grad():
        model(torch.zeros(1, analyzer.window_samples, device=device))

    hop_samples = int(args.hop * TARGET_SR)
    stop_event = threading.Event()
    if args.source == "file":
        if not args.file or not os.path.exists(args.file):
            sys.exit("[realtime] --source file requires an existing --file path.")
        stream = file_chunk_stream(args.file, hop_samples, simulate_realtime=not args.no_pacing)
    else:
        stream = mic_chunk_stream(hop_samples, args.mic_device, stop_event)

    stream_time = 0.0
    try:
        for chunk in stream:
            stream_time += len(chunk) / TARGET_SR
            r = analyzer.push(chunk, stream_time)
            if r is not None:
                print_result(analyzer, r)
            if args.duration and stream_time >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[realtime] Stopped by user.")
    finally:
        stop_event.set()

    if args.log_csv and analyzer.history:
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(analyzer.history[0].keys()))
            writer.writeheader()
            writer.writerows(analyzer.history)
        print(f"[realtime] Wrote per-window log to {args.log_csv}")

    s = analyzer.summary()
    print("\n" + "=" * 55)
    print("            LIVE ANALYSIS SESSION SUMMARY            ")
    print("=" * 55)
    for k, v in s.items():
        print(f" {k:<20}: {v:.4f}" if isinstance(v, float) else f" {k:<20}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
