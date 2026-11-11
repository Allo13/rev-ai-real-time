"""
Dataset Download Guide & Helper for RevealAI v2 Training

Datasets needed (in order of priority):
  1. ASVspoof 5 (2024) — best single upgrade
  2. MLAAD v2 (2024) — modern TTS systems, 23 languages
  3. In-The-Wild (2024) — real deepfakes from the internet

This script helps organize downloaded datasets into the expected directory structure.
"""

import os
import sys
import csv
import shutil
import argparse


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║              RevealAI v2 — Dataset Download Guide                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. ASVspoof 5 (2024) — HIGHEST PRIORITY                          ║
║     • Register at: https://www.asvspoof.org/                      ║
║     • Download the LA (Logical Access) partition                   ║
║     • Extract to: dataset/ASVspoof5/                               ║
║     • Expected structure:                                          ║
║       dataset/ASVspoof5/train/protocol.txt                         ║
║       dataset/ASVspoof5/train/flac/                                ║
║       dataset/ASVspoof5/dev/protocol.txt                           ║
║       dataset/ASVspoof5/dev/flac/                                  ║
║                                                                    ║
║  2. MLAAD v2 (Multilingual Audio Anti-spoofing Dataset)            ║
║     • Download from: https://zenodo.org/records/10512048           ║
║     • Extract to: dataset/MLAAD/                                   ║
║     • Expected structure:                                          ║
║       dataset/MLAAD/metadata.csv    (columns: path,label)          ║
║       dataset/MLAAD/audio/          (audio files)                  ║
║                                                                    ║
║  3. In-The-Wild                                                    ║
║     • Download from: https://zenodo.org/records/10511438           ║
║     • Extract to: dataset/InTheWild/                               ║
║     • Expected structure:                                          ║
║       dataset/InTheWild/meta.csv    (columns: file,label)          ║
║       dataset/InTheWild/audio/      (audio files)                  ║
║                                                                    ║
║  After downloading, run:                                           ║
║    python dataset/download_datasets.py --verify                    ║
║                                                                    ║
║  Then train:                                                       ║
║    python train_v2.py --arch both --epochs 10                      ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def verify_datasets():
    """Check which datasets are available and properly structured."""
    print("\n=== Dataset Verification ===\n")
    found_any = False

    # ASVspoof 2019 (existing)
    p = os.path.join(DATASET_DIR, "LA", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
    a = os.path.join(DATASET_DIR, "LA", "LA", "ASVspoof2019_LA_train", "flac")
    if os.path.exists(p) and os.path.isdir(a):
        n = len([f for f in os.listdir(a) if f.endswith('.flac')])
        print(f"  ✓ ASVspoof 2019 (LA train): {n} files")
        found_any = True
    else:
        print(f"  ✗ ASVspoof 2019: not found")

    # ASVspoof 5
    p = os.path.join(DATASET_DIR, "ASVspoof5", "train", "protocol.txt")
    a = os.path.join(DATASET_DIR, "ASVspoof5", "train", "flac")
    if os.path.exists(p) and os.path.isdir(a):
        n = len(os.listdir(a))
        print(f"  ✓ ASVspoof 5 (train): {n} files")
        found_any = True
    else:
        print(f"  ✗ ASVspoof 5: not found at dataset/ASVspoof5/")

    # MLAAD
    m = os.path.join(DATASET_DIR, "MLAAD", "metadata.csv")
    a = os.path.join(DATASET_DIR, "MLAAD", "audio")
    if os.path.exists(m) and os.path.isdir(a):
        n = len(os.listdir(a))
        with open(m) as f:
            rows = sum(1 for _ in f) - 1
        print(f"  ✓ MLAAD: {rows} entries, {n} audio files")
        found_any = True
    else:
        print(f"  ✗ MLAAD: not found at dataset/MLAAD/")

    # In-The-Wild
    m = os.path.join(DATASET_DIR, "InTheWild", "meta.csv")
    a = os.path.join(DATASET_DIR, "InTheWild", "audio")
    if os.path.exists(m) and os.path.isdir(a):
        n = len(os.listdir(a))
        with open(m) as f:
            rows = sum(1 for _ in f) - 1
        print(f"  ✓ In-The-Wild: {rows} entries, {n} audio files")
        found_any = True
    else:
        print(f"  ✗ In-The-Wild: not found at dataset/InTheWild/")

    if not found_any:
        print("\n  No datasets found! Follow the download instructions above.")
    else:
        print(f"\n  Ready to train: python train_v2.py --arch both --epochs 10")

    return found_any


def create_mlaad_metadata(mlaad_dir):
    """Generate metadata.csv for MLAAD if the directory has a different structure."""
    audio_dir = os.path.join(mlaad_dir, "audio")
    meta_path = os.path.join(mlaad_dir, "metadata.csv")

    if os.path.exists(meta_path):
        print(f"metadata.csv already exists at {meta_path}")
        return

    if not os.path.isdir(audio_dir):
        print(f"No audio directory found at {audio_dir}")
        return

    print(f"Scanning {audio_dir} for audio files...")
    rows = []
    for root, dirs, files in os.walk(audio_dir):
        for fname in files:
            if fname.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                rel_path = os.path.relpath(os.path.join(root, fname), audio_dir)
                # MLAAD typically has fake/ and real/ subdirectories
                if 'fake' in root.lower() or 'spoof' in root.lower():
                    label = 'spoof'
                elif 'real' in root.lower() or 'bonafide' in root.lower() or 'bona' in root.lower():
                    label = 'bonafide'
                else:
                    label = 'spoof'  # default assumption for MLAAD
                rows.append((rel_path, label))

    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for row in rows:
            writer.writerow(row)

    print(f"Created {meta_path} with {len(rows)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify which datasets are available")
    parser.add_argument("--create-mlaad-meta", type=str, help="Generate metadata.csv for MLAAD directory")
    parser.add_argument("--instructions", action="store_true", help="Show download instructions")
    args = parser.parse_args()

    if args.instructions or (not args.verify and not args.create_mlaad_meta):
        print(INSTRUCTIONS)

    if args.verify:
        verify_datasets()

    if args.create_mlaad_meta:
        create_mlaad_metadata(args.create_mlaad_meta)
