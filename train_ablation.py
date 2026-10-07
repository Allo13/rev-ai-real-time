import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from dataset.asvspoof_loader import ASVspoofDataset
from models.truncated_xlsr import TruncatedXLSR, load_xlsr_300m
from models.fake_mamba import FakeMambaClassifier, FakeMambaModel
from utils.profiler import benchmark_rtf_and_gflops


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) given bonafide and spoof prediction probability scores.
    
    Args:
        bonafide_scores (np.ndarray): Scores assigned to genuine audio clips
        spoof_scores (np.ndarray): Scores assigned to spoofed audio clips
    Returns:
        float: Equal Error Rate percentage (%)
    """
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer * 100.0)


def train_single_ablation_run(
    run_name: str,
    num_layers: int,
    checkpoint_dir: str,
    base_xlsr_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Executes a single ablation training run (e.g. Baseline, Run A, or Run B).
    
    Args:
        run_name (str): Identifier for the ablation experiment
        num_layers (int): Number of XLSR encoder layers to retain
        checkpoint_dir (str): Isolated output directory for checkpoints & logs
        base_xlsr_model (nn.Module): Base Fairseq XLSR-300M model instance
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        epochs (int): Training epoch count
        lr (float): Learning rate
        device (torch.device): CUDA/CPU device
        
    Returns:
        Dict[str, Any]: Summary metrics for the run
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print("\n" + "="*70)
    print(f" STARTING ABLATION EXPERIMENT: {run_name.upper()} ({num_layers} XLSR Layers)")
    print(f" Isolated Checkpoint Directory: {checkpoint_dir}")
    print("="*70)

    # 1. Instantiate Truncated Front-End and Fake-Mamba Classifier Back-End
    front_end = TruncatedXLSR(base_xlsr_model=base_xlsr_model, num_layers=num_layers)
    back_end = FakeMambaClassifier(input_dim=1024, hidden_dim=144, num_mamba_layers=2, num_classes=2)
    model = FakeMambaModel(front_end=front_end, back_end=back_end).to(device)

    # 2. Benchmark RTF and GFLOPs before training loop
    print(f"\n--- Profiling Real-Time Performance ({run_name}) ---")
    profile_stats = benchmark_rtf_and_gflops(model, sample_len=64000, device=device)

    # 3. Setup Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_eer = 100.0
    best_val_acc = 0.0

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()

        for step, (waveforms, labels, _) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        bonafide_scores = []
        spoof_scores = []

        with torch.no_grad():
            for waveforms, labels, _ in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                logits = model(waveforms)
                loss = criterion(logits, labels)

                val_loss += loss.item() * waveforms.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Store bonafide probability scores (class 0 = bonafide)
                for prob, label in zip(probs[:, 0].cpu().numpy(), labels.cpu().numpy()):
                    if label == 0:
                        bonafide_scores.append(prob)
                    else:
                        spoof_scores.append(prob)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        try:
            epoch_eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
        except Exception:
            epoch_eer = 100.0 - epoch_val_acc

        epoch_time = time.time() - start_time

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", epoch_val_acc, epoch)
        writer.add_scalar("EER/Validation", epoch_eer, epoch)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, EER: {epoch_eer:.2f}%")

        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_eer = epoch_eer

            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'run_name': run_name,
                'num_layers': num_layers,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_eer': best_eer,
                'layer_weights': front_end.layer_weights.data.cpu().numpy().tolist()
            }, checkpoint_path)
            print(f"   --> Saved new best checkpoint to {checkpoint_path}")

    writer.close()

    run_summary = {
        "run_name": run_name,
        "num_layers": num_layers,
        "checkpoint_dir": checkpoint_dir,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_eer": best_eer,
        "gflops": profile_stats["gflops"],
        "macs_str": profile_stats["macs_str"],
        "params_str": profile_stats["params_str"],
        "avg_latency_ms": profile_stats["avg_latency_ms"],
        "rtf": profile_stats["rtf"]
    }
    
    # Save individual run summary JSON
    summary_json_path = os.path.join(checkpoint_dir, "run_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(run_summary, f, indent=4)

    return run_summary


def run_all_ablations(
    protocol_file: str = None, 
    audio_dir: str = None, 
    val_protocol_file: str = None,
    val_audio_dir: str = None,
    xlsr_path: str = None,
    epochs: int = 3, 
    batch_size: int = 16, 
    lr: float = 1e-4, 
    run_mode: str = "all",
    device_str: str = "auto"
):
    """
    Sequentially executes the three ablation experiments:
    1. Baseline: Full 24-layer XLSR extraction -> ./checkpoints/baseline/
    2. Run A: Truncated to 12 layers         -> ./checkpoints/run_a_12/
    3. Run B: Truncated to 8 layers          -> ./checkpoints/run_b_8/
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using Compute Device: {device}")

    # Auto-resolve paths if not provided
    if protocol_file is None:
        default_train_proto = "./dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        if os.path.exists(default_train_proto):
            protocol_file = default_train_proto
            print(f"Auto-resolved train protocol file: {protocol_file}")

    if audio_dir is None:
        default_train_audio = "./dataset/LA/LA/ASVspoof2019_LA_train/flac"
        if os.path.exists(default_train_audio):
            audio_dir = default_train_audio
            print(f"Auto-resolved train audio directory: {audio_dir}")

    if val_protocol_file is None:
        default_val_proto = "./dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        if os.path.exists(default_val_proto):
            val_protocol_file = default_val_proto
            print(f"Auto-resolved validation protocol file: {val_protocol_file}")
        else:
            val_protocol_file = protocol_file

    if val_audio_dir is None:
        default_val_audio = "./dataset/LA/LA/ASVspoof2019_LA_dev/flac"
        if os.path.exists(default_val_audio):
            val_audio_dir = default_val_audio
            print(f"Auto-resolved validation audio directory: {val_audio_dir}")
        else:
            val_audio_dir = audio_dir

    if xlsr_path is None:
        default_xlsr = "./weights/xlsr_53_56k.pt"
        if os.path.exists(default_xlsr):
            xlsr_path = default_xlsr
            print(f"Auto-resolved XLSR checkpoint path: {xlsr_path}")

    # Load Base XLSR-300M model once (or architecture fallback)
    print("\nLoading Base Fairseq XLSR-300M Model...")
    base_xlsr = load_xlsr_300m(checkpoint_path=xlsr_path)

    # Prepare DataLoaders
    train_dataset = ASVspoofDataset(
        protocol_file=protocol_file, 
        audio_dir=audio_dir, 
        is_train=True, 
        rawboost_algo=4,
        num_synth_samples=64
    )
    val_dataset = ASVspoofDataset(
        protocol_file=val_protocol_file, 
        audio_dir=val_audio_dir, 
        is_train=False, 
        rawboost_algo=0,
        num_synth_samples=32
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the three ablation configurations with strict output routing
    ablation_configs = [
        {"run_name": "baseline", "num_layers": 24, "checkpoint_dir": "./checkpoints/baseline/"},
        {"run_name": "run_a_12", "num_layers": 12, "checkpoint_dir": "./checkpoints/run_a_12/"},
        {"run_name": "run_b_8",  "num_layers": 8,  "checkpoint_dir": "./checkpoints/run_b_8/"},
    ]

    if run_mode != "all":
        ablation_configs = [cfg for cfg in ablation_configs if cfg["run_name"] == run_mode]

    all_summaries = []

    for cfg in ablation_configs:
        summary = train_single_ablation_run(
            run_name=cfg["run_name"],
            num_layers=cfg["num_layers"],
            checkpoint_dir=cfg["checkpoint_dir"],
            base_xlsr_model=base_xlsr,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )
        all_summaries.append(summary)

    # Print Final Comparative Summary Table across Ablation Runs
    print("\n" + "="*85)
    print("                     FINAL ABLATION STUDY COMPARATIVE REPORT                    ")
    print("="*85)
    header = f"{'Run Name':<12} | {'Layers':<6} | {'Params':<10} | {'GFLOPs':<9} | {'RTF':<10} | {'Latency(ms)':<12} | {'Val Acc(%)':<10} | {'Val EER(%)':<10}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(f"{s['run_name']:<12} | {s['num_layers']:<6} | {s['params_str']:<10} | {s['gflops']:<9.2f} | {s['rtf']:<10.6f} | {s['avg_latency_ms']:<12.2f} | {s['best_val_acc']:<10.2f} | {s['best_val_eer']:<10.2f}")
    print("="*85 + "\n")

    # Save overall summary to JSON file
    summary_file = "./checkpoints/ablation_study_summary.json"
    os.makedirs("./checkpoints", exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=4)
    print(f"Saved complete comparative ablation report to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Training Runner for Truncated XLSR + Fake-Mamba Audio Deepfake Detection")
    parser.add_argument("--protocol_file", type=str, default=None, help="Path to ASVspoof train protocol file")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing ASVspoof train .flac audio files")
    parser.add_argument("--val_protocol_file", type=str, default=None, help="Path to ASVspoof validation protocol file")
    parser.add_argument("--val_audio_dir", type=str, default=None, help="Directory containing ASVspoof validation .flac audio files")
    parser.add_argument("--xlsr_path", type=str, default=None, help="Path to Fairseq XLSR-300M .pt checkpoint file (e.g. ./weights/xlsr_53_56k.pt)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per ablation run")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--run_mode", type=str, default="all", choices=["all", "baseline", "run_a_12", "run_b_8"], help="Ablation run target")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()
    run_all_ablations(
        protocol_file=args.protocol_file,
        audio_dir=args.audio_dir,
        val_protocol_file=args.val_protocol_file,
        val_audio_dir=args.val_audio_dir,
        xlsr_path=args.xlsr_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_mode=args.run_mode,
        device_str=args.device
    )

