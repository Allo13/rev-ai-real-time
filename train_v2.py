"""
RevealAI v2 Training Script

Two architecture configurations:
  Fast:     wav2vec2-base (95M, 768-dim) + AASIST → real-time, good accuracy
  Accurate: XLSR-300M (315M, 1024-dim)  + AASIST → slower, higher accuracy

Trains on ASVspoof5 + MLAAD + In-The-Wild with codec augmentation.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
import json
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from dataset.asvspoof_loader import TARGET_SR


def compute_eer(bonafide_scores, spoof_scores):
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer * 100.0)


def build_model(arch, device):
    """Build model for a given architecture name."""
    if arch == "fast":
        from models.ssl_frontend import load_wav2vec2_base, TruncatedSSL
        from models.aasist_backend import AASISTBackend, SSLAntiSpoofModel

        print("[v2] Loading wav2vec2-base (95M params)...")
        base = load_wav2vec2_base()
        frontend = TruncatedSSL(base, num_layers=12)
        del base; gc.collect()
        feat_dim = 768
        backend = AASISTBackend(input_dim=feat_dim, hidden_dim=160, num_heads=4, num_classes=2)
        model = SSLAntiSpoofModel(frontend, backend)

    elif arch == "accurate":
        from models.ssl_frontend import load_xlsr_large, TruncatedSSL
        from models.aasist_backend import AASISTBackend, SSLAntiSpoofModel

        print("[v2] Loading XLSR-300M (315M params)...")
        base = load_xlsr_large()
        frontend = TruncatedSSL(base, num_layers=12)
        del base; gc.collect()
        feat_dim = 1024
        backend = AASISTBackend(input_dim=feat_dim, hidden_dim=256, num_heads=4, num_classes=2)
        model = SSLAntiSpoofModel(frontend, backend)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Freeze SSL frontend weights (only train layer aggregation weights + backend)
    for param in model.front_end.ssl.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[v2] {arch}: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total params")

    return model.to(device)


def build_dataloaders(args):
    """Build train/val dataloaders from available datasets."""
    try:
        from dataset.multi_dataset_loader import MultiDataset
        HAS_MULTI = True
    except ImportError:
        HAS_MULTI = False

    from dataset.asvspoof_loader import ASVspoofDataset

    datasets_train = []
    datasets_val = []

    # ASVspoof 2019 (existing data)
    asvspoof19_train_proto = "./dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    asvspoof19_train_audio = "./dataset/LA/LA/ASVspoof2019_LA_train/flac"
    asvspoof19_val_proto = "./dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    asvspoof19_val_audio = "./dataset/LA/LA/ASVspoof2019_LA_dev/flac"

    if os.path.exists(asvspoof19_train_proto) and os.path.isdir(asvspoof19_train_audio):
        print("[v2] Found ASVspoof 2019 train data")
        datasets_train.append(ASVspoofDataset(
            protocol_file=asvspoof19_train_proto,
            audio_dir=asvspoof19_train_audio,
            is_train=True, rawboost_algo=4
        ))
    if os.path.exists(asvspoof19_val_proto) and os.path.isdir(asvspoof19_val_audio):
        print("[v2] Found ASVspoof 2019 val data")
        datasets_val.append(ASVspoofDataset(
            protocol_file=asvspoof19_val_proto,
            audio_dir=asvspoof19_val_audio,
            is_train=False, rawboost_algo=0
        ))

    # ASVspoof 5
    asvspoof5_train_proto = args.asvspoof5_proto or "./dataset/ASVspoof5/train/protocol.txt"
    asvspoof5_train_audio = args.asvspoof5_audio or "./dataset/ASVspoof5/train/flac"
    if os.path.exists(asvspoof5_train_proto) and os.path.isdir(asvspoof5_train_audio):
        print("[v2] Found ASVspoof 5 train data")
        datasets_train.append(ASVspoofDataset(
            protocol_file=asvspoof5_train_proto,
            audio_dir=asvspoof5_train_audio,
            is_train=True, rawboost_algo=4
        ))

    # MLAAD
    if HAS_MULTI:
        mlaad_meta = args.mlaad_meta or "./dataset/MLAAD/metadata.csv"
        mlaad_audio = args.mlaad_audio or "./dataset/MLAAD/audio"
        if os.path.exists(mlaad_meta) and os.path.isdir(mlaad_audio):
            print("[v2] Found MLAAD data")
            datasets_train.append(MultiDataset(
                dataset_configs=[{"name": "mlaad", "protocol_file": mlaad_meta, "audio_dir": mlaad_audio, "format": "mlaad"}],
                is_train=True, codec_aug_prob=0.5
            ))

        # In-The-Wild
        itw_meta = args.itw_meta or next((p for p in [
            "./dataset/InTheWild/meta.csv",
            "D:/datasets/InTheWild/release_in_the_wild/meta.csv",
            "D:/datasets/InTheWild/meta.csv",
        ] if os.path.exists(p)), "./dataset/InTheWild/meta.csv")
        itw_audio = args.itw_audio or next((p for p in [
            "./dataset/InTheWild/release_in_the_wild",
            "D:/datasets/InTheWild/release_in_the_wild",
            "./dataset/InTheWild/audio",
        ] if os.path.isdir(p)), "./dataset/InTheWild/audio")
        if os.path.exists(itw_meta) and os.path.isdir(itw_audio):
            print("[v2] Found In-The-Wild data")
            datasets_train.append(MultiDataset(
                dataset_configs=[{"name": "inthewild", "protocol_file": itw_meta, "audio_dir": itw_audio, "format": "inthewild"}],
                is_train=True, codec_aug_prob=0.5
            ))

    if not datasets_train:
        print("[v2] WARNING: No real dataset found. Using synthetic data for testing.")
        datasets_train.append(ASVspoofDataset(is_train=True, num_synth_samples=200))
        datasets_val.append(ASVspoofDataset(is_train=False, num_synth_samples=50))

    if not datasets_val:
        datasets_val = datasets_train  # fallback

    train_ds = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    val_ds = ConcatDataset(datasets_val) if len(datasets_val) > 1 else datasets_val[0]

    if args.max_samples and args.max_samples < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.max_samples)))
        val_ds = Subset(val_ds, list(range(min(args.max_samples // 4, len(val_ds)))))

    print(f"[v2] Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def train(model, train_loader, val_loader, args, arch_name):
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"v2_{arch_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_eer = 100.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for step, (waveforms, labels, _) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
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
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

                for prob, label in zip(probs[:, 0].cpu().numpy(), labels.cpu().numpy()):
                    if label == 0:
                        bonafide_scores.append(prob)
                    else:
                        spoof_scores.append(prob)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        try:
            eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
        except Exception:
            eer = 100.0 - val_acc

        elapsed = time.time() - t0
        print(f"Epoch [{epoch}/{args.epochs}] ({elapsed:.0f}s) | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% EER={eer:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_eer = eer
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'arch': arch_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_eer': eer,
            }, ckpt_path)
            print(f"  → Saved best model (EER={eer:.2f}%) to {ckpt_path}")

    return {"arch": arch_name, "best_val_loss": best_val_loss, "best_eer": best_eer}


def main():
    parser = argparse.ArgumentParser(description="RevealAI v2 Training — wav2vec2+AASIST")
    parser.add_argument("--arch", type=str, default="fast", choices=["fast", "accurate", "both"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None)

    # Dataset paths (auto-detected if not provided)
    parser.add_argument("--asvspoof5_proto", type=str, default=None)
    parser.add_argument("--asvspoof5_audio", type=str, default=None)
    parser.add_argument("--mlaad_meta", type=str, default=None)
    parser.add_argument("--mlaad_audio", type=str, default=None)
    parser.add_argument("--itw_meta", type=str, default=None)
    parser.add_argument("--itw_audio", type=str, default=None)

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            device = torch.device("cuda" if free_mem >= 2.0 else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[v2] Device: {device}")

    train_loader, val_loader = build_dataloaders(args)

    archs = ["fast", "accurate"] if args.arch == "both" else [args.arch]
    results = []

    for arch in archs:
        print(f"\n{'='*60}")
        print(f"  Training: {arch.upper()} architecture")
        print(f"{'='*60}")
        model = build_model(arch, device)
        result = train(model, train_loader, val_loader, args, arch)
        results.append(result)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['arch']:<10} | Best Val Loss: {r['best_val_loss']:.4f} | Best EER: {r['best_eer']:.2f}%")


if __name__ == "__main__":
    main()
