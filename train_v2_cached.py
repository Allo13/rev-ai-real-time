"""
RevealAI v2 Training — Feature Caching Strategy

For memory-constrained GPUs: extract SSL features once on CPU,
cache to disk, then train only the lightweight AASIST backend on GPU.
This avoids holding the 95M-param wav2vec2 in VRAM during training.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores, spoof_scores):
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer * 100.0)


class CachedFeatureDataset(Dataset):
    def __init__(self, features_dir, split="train"):
        self.features_dir = os.path.join(features_dir, split)
        manifest = os.path.join(self.features_dir, "manifest.npz")
        data = np.load(manifest)
        self.labels = data["labels"]
        self.ids = data["ids"]
        self.n = len(self.labels)
        print(f"[cached] Loaded {split}: {self.n} samples, bonafide={np.sum(self.labels == 0)}, spoof={np.sum(self.labels == 1)}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.features_dir, f"{idx}.npy"))
        return torch.from_numpy(feat).float(), int(self.labels[idx]), str(self.ids[idx])


def extract_and_cache(args):
    """Extract SSL features from all data and save to disk."""
    cache_dir = args.cache_dir
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)

    extract_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] Loading SSL frontend on {extract_device}...")
    if args.arch == "fast":
        from models.ssl_frontend import load_wav2vec2_base, TruncatedSSL
        base = load_wav2vec2_base()
        frontend = TruncatedSSL(base, num_layers=12)
        del base
    else:
        from models.ssl_frontend import load_xlsr_large, TruncatedSSL
        base = load_xlsr_large()
        frontend = TruncatedSSL(base, num_layers=12)
        del base
    gc.collect()

    for param in frontend.parameters():
        param.requires_grad = False
    frontend = frontend.to(extract_device).eval()
    if extract_device.type == "cuda":
        torch.cuda.empty_cache()

    from train_v2 import build_dataloaders
    train_loader, val_loader = build_dataloaders(args)

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        split_dir = os.path.join(cache_dir, split)
        manifest_path = os.path.join(split_dir, "manifest.npz")

        if os.path.exists(manifest_path) and not args.force_cache:
            existing = np.load(manifest_path)
            print(f"[cache] {split} already cached ({len(existing['labels'])} samples), skipping. Use --force_cache to redo.")
            continue

        all_labels = []
        all_ids = []
        t0 = time.time()

        for i, (waveform, label, audio_id) in enumerate(loader.dataset):
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            waveform = waveform.unsqueeze(0).to(extract_device)

            with torch.no_grad():
                feat = frontend(waveform)
            feat_np = feat.squeeze(0).cpu().numpy()
            np.save(os.path.join(split_dir, f"{i}.npy"), feat_np)

            all_labels.append(label)
            all_ids.append(audio_id if isinstance(audio_id, str) else str(audio_id))

            if (i + 1) % 500 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(loader.dataset) - i - 1) / rate
                print(f"[cache] {split}: {i+1}/{len(loader.dataset)} ({rate:.1f} samples/s, ETA {eta/60:.0f}min)")

        np.savez(manifest_path,
                 labels=np.array(all_labels, dtype=np.int64),
                 ids=np.array(all_ids))
        elapsed = time.time() - t0
        print(f"[cache] {split}: cached {len(all_labels)} samples in {elapsed:.0f}s")

    del frontend
    gc.collect()
    print("[cache] Feature extraction complete!")


def train_backend(args):
    """Train only the AASIST backend on cached features."""
    cache_dir = args.cache_dir

    train_ds = CachedFeatureDataset(cache_dir, "train")
    val_ds = CachedFeatureDataset(cache_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    feat_dim = np.load(os.path.join(cache_dir, "train", "0.npy")).shape[-1]
    print(f"[train] Feature dim: {feat_dim}")

    from models.aasist_backend import AASISTBackend
    if args.arch == "fast":
        backend = AASISTBackend(input_dim=feat_dim, hidden_dim=160, num_heads=4, num_classes=2)
    else:
        backend = AASISTBackend(input_dim=feat_dim, hidden_dim=256, num_heads=4, num_classes=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = backend.to(device)
    trainable = sum(p.numel() for p in backend.parameters() if p.requires_grad)
    print(f"[train] AASIST backend: {trainable/1e6:.2f}M params on {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(backend.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_eer = 100.0
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"v2_{args.arch}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        backend.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for step, (features, labels, _) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = backend(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 200 == 0:
                print(f"  Step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total * 100

        backend.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        bonafide_scores = []
        spoof_scores = []

        with torch.no_grad():
            for features, labels, _ in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = backend(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
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
            save_full_model(backend, feat_dim, args, checkpoint_dir, epoch, val_loss, val_acc, eer)

    print(f"\n{'='*60}")
    print(f"  RESULTS: Best Val Loss: {best_val_loss:.4f} | Best EER: {best_eer:.2f}%")
    print(f"{'='*60}")


def save_full_model(backend, feat_dim, args, checkpoint_dir, epoch, val_loss, val_acc, eer):
    """Save a complete model checkpoint (frontend + backend) for inference."""
    print(f"  Building full model for checkpoint...")

    if args.arch == "fast":
        from models.ssl_frontend import load_wav2vec2_base, TruncatedSSL
        base = load_wav2vec2_base()
        frontend = TruncatedSSL(base, num_layers=12)
        del base
    else:
        from models.ssl_frontend import load_xlsr_large, TruncatedSSL
        base = load_xlsr_large()
        frontend = TruncatedSSL(base, num_layers=12)
        del base
    gc.collect()

    from models.aasist_backend import SSLAntiSpoofModel
    full_model = SSLAntiSpoofModel(frontend, backend.cpu())
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'model_state_dict': full_model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_eer': eer,
    }, ckpt_path)
    print(f"  → Saved best model (EER={eer:.2f}%) to {ckpt_path}")

    del full_model, frontend
    gc.collect()
    backend.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def main():
    parser = argparse.ArgumentParser(description="RevealAI v2 Cached Training")
    parser.add_argument("--arch", type=str, default="fast", choices=["fast", "accurate"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_dir", type=str, default="D:/datasets/feature_cache")
    parser.add_argument("--force_cache", action="store_true")
    parser.add_argument("--skip_cache", action="store_true", help="Skip extraction, use existing cache")

    parser.add_argument("--asvspoof5_proto", type=str, default=None)
    parser.add_argument("--asvspoof5_audio", type=str, default=None)
    parser.add_argument("--mlaad_meta", type=str, default=None)
    parser.add_argument("--mlaad_audio", type=str, default=None)
    parser.add_argument("--itw_meta", type=str, default=None)
    parser.add_argument("--itw_audio", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if not args.skip_cache:
        extract_and_cache(args)

    train_backend(args)


if __name__ == "__main__":
    main()
