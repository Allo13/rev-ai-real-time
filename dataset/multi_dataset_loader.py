import os
import csv
import random
import subprocess
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import soundfile as sf
import librosa

from dataset.asvspoof_loader import TARGET_SR, RawBoostAugmenter


class CodecAugmenter:
    """
    Applies random audio codec compression via ffmpeg to simulate
    real-world transmission artifacts (YouTube, VoIP, streaming).
    """
    CODECS = [
        {"ext": ".opus", "enc": ["-c:a", "libopus", "-b:a"], "bitrates": ["32k", "48k", "64k", "96k"]},
        {"ext": ".aac",  "enc": ["-c:a", "aac", "-b:a"],     "bitrates": ["48k", "64k", "96k", "128k"]},
        {"ext": ".mp3",  "enc": ["-c:a", "libmp3lame", "-b:a"], "bitrates": ["64k", "96k", "128k"]},
        {"ext": ".amr",  "enc": ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1", "-b:a"], "bitrates": ["12.2k"]},
        {"ext": ".wav",  "enc": ["-c:a", "g722", "-ar", "16000", "-ac", "1", "-b:a"], "bitrates": ["64k"]},
    ]

    def __init__(self, prob=0.5):
        self.prob = prob
        self._ffmpeg_ok = None

    def _check_ffmpeg(self):
        if self._ffmpeg_ok is None:
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
                self._ffmpeg_ok = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._ffmpeg_ok = False
        return self._ffmpeg_ok

    def __call__(self, waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        if random.random() > self.prob:
            return waveform
        if not self._check_ffmpeg():
            return waveform

        codec = random.choice(self.CODECS)
        bitrate = random.choice(codec["bitrates"])

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                in_path = f_in.name
                sf.write(in_path, waveform, sr)

            out_path = in_path + codec["ext"]
            dec_path = in_path + ".dec.wav"

            cmd_enc = ["ffmpeg", "-y", "-i", in_path] + codec["enc"] + [bitrate, out_path]
            subprocess.run(cmd_enc, capture_output=True, timeout=10)

            cmd_dec = ["ffmpeg", "-y", "-i", out_path, "-ar", str(sr), "-ac", "1", dec_path]
            subprocess.run(cmd_dec, capture_output=True, timeout=10)

            aug, _ = sf.read(dec_path, dtype="float32")
            if aug.ndim > 1:
                aug = aug[:, 0]

            for p in [in_path, out_path, dec_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            if len(aug) == 0:
                return waveform
            return aug.astype(np.float32)

        except Exception:
            for p in [in_path, out_path, dec_path]:
                try:
                    os.unlink(p)
                except (OSError, UnboundLocalError):
                    pass
            return waveform


def _parse_asvspoof(protocol_file, audio_dir):
    """Parse standard ASVspoof protocol (2019/2021/5) format."""
    entries = []
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            audio_id = parts[1]
            key = parts[-1].lower()
            label = 0 if key == "bonafide" else 1

            audio_path = None
            for ext in [".flac", ".wav", ".mp3", ".ogg"]:
                candidate = os.path.join(audio_dir, audio_id + ext)
                if os.path.exists(candidate):
                    audio_path = candidate
                    break
            if audio_path is None:
                audio_path = os.path.join(audio_dir, audio_id + ".flac")

            entries.append((audio_path, label, audio_id))
    return entries


def _parse_mlaad(protocol_file, audio_dir):
    """Parse MLAAD metadata CSV: path,label (bonafide/spoof)."""
    entries = []
    with open(protocol_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get("path", "").strip()
            key = row.get("label", "").strip().lower()
            label = 0 if key == "bonafide" else 1
            audio_path = os.path.join(audio_dir, rel_path) if audio_dir else rel_path
            audio_id = os.path.splitext(os.path.basename(rel_path))[0]
            entries.append((audio_path, label, audio_id))
    return entries


def _parse_inthewild(protocol_file, audio_dir):
    """Parse In-The-Wild metadata CSV: file,label (bona-fide/spoof)."""
    entries = []
    with open(protocol_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get("file", "").strip()
            key = row.get("label", "").strip().lower()
            label = 0 if key in ("bona-fide", "bonafide") else 1
            audio_path = os.path.join(audio_dir, rel_path) if audio_dir else rel_path
            audio_id = os.path.splitext(os.path.basename(rel_path))[0]
            entries.append((audio_path, label, audio_id))
    return entries


PARSERS = {
    "asvspoof": _parse_asvspoof,
    "asvspoof5": _parse_asvspoof,
    "mlaad": _parse_mlaad,
    "inthewild": _parse_inthewild,
}


class MultiDataset(Dataset):
    """
    Unified dataset combining multiple anti-spoofing datasets.

    Each dataset_config is a dict with keys:
        name (str): human-readable name
        protocol_file (str): path to protocol/metadata file
        audio_dir (str): base directory for audio files
        format (str): one of "asvspoof", "asvspoof5", "mlaad", "inthewild"
    """

    def __init__(
        self,
        dataset_configs: list,
        max_len: int = 64000,
        is_train: bool = True,
        rawboost_algo: int = 4,
        codec_aug_prob: float = 0.5,
    ):
        self.max_len = max_len
        self.is_train = is_train
        self.rawboost = RawBoostAugmenter(algo=rawboost_algo if is_train else 0)
        self.codec_aug = CodecAugmenter(prob=codec_aug_prob) if is_train else None

        self.entries = []  # list of (audio_path, label, audio_id)

        for cfg in dataset_configs:
            fmt = cfg.get("format", "asvspoof")
            parser = PARSERS.get(fmt)
            if parser is None:
                raise ValueError(f"Unknown dataset format: {fmt}. Choose from {list(PARSERS.keys())}")

            proto = cfg.get("protocol_file", "")
            audio_dir = cfg.get("audio_dir", "")

            if not os.path.exists(proto):
                print(f"[MultiDataset] WARNING: protocol file not found: {proto} — skipping {cfg.get('name', fmt)}")
                continue

            parsed = parser(proto, audio_dir)
            print(f"[MultiDataset] Loaded {len(parsed)} samples from {cfg.get('name', fmt)} ({fmt})")
            self.entries.extend(parsed)

        bonafide = sum(1 for _, l, _ in self.entries if l == 0)
        spoof = len(self.entries) - bonafide
        print(f"[MultiDataset] Total: {len(self.entries)} samples (bonafide={bonafide}, spoof={spoof}), train={is_train}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        audio_path, label, audio_id = self.entries[idx]

        try:
            waveform, sr = sf.read(audio_path, dtype="float32")
        except Exception:
            try:
                waveform, sr = librosa.load(audio_path, sr=None)
            except Exception:
                waveform = np.zeros(self.max_len, dtype=np.float32)
                sr = TARGET_SR

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        # Codec augmentation (before padding/truncation so length may change)
        if self.is_train and self.codec_aug is not None:
            waveform = self.codec_aug(waveform, TARGET_SR)

        # Pad or truncate to max_len
        num_samples = len(waveform)
        if num_samples < self.max_len:
            tile_count = (self.max_len // max(num_samples, 1)) + 1
            waveform = np.tile(waveform, tile_count)[:self.max_len]
        elif num_samples > self.max_len:
            if self.is_train:
                start = random.randint(0, num_samples - self.max_len)
            else:
                start = (num_samples - self.max_len) // 2
            waveform = waveform[start : start + self.max_len]

        waveform_tensor = torch.from_numpy(waveform).float()

        # RawBoost augmentation
        if self.is_train:
            waveform_tensor = self.rawboost(waveform_tensor)

        return waveform_tensor, label, audio_id


def get_dataloaders(
    train_configs: list,
    val_configs: list = None,
    max_len: int = 64000,
    batch_size: int = 16,
    rawboost_algo: int = 4,
    codec_aug_prob: float = 0.5,
    num_workers: int = 4,
):
    """
    Build train and validation DataLoaders from dataset config lists.

    Args:
        train_configs: list of dicts, each with name/protocol_file/audio_dir/format
        val_configs: same format for validation (if None, uses train_configs without augmentation)
        max_len: samples per clip (64000 = 4s at 16kHz)
        batch_size: batch size
        rawboost_algo: RawBoost algorithm (0=off, 4=combined)
        codec_aug_prob: probability of applying codec augmentation per sample
        num_workers: dataloader workers

    Returns:
        (train_loader, val_loader)
    """
    train_ds = MultiDataset(
        train_configs,
        max_len=max_len,
        is_train=True,
        rawboost_algo=rawboost_algo,
        codec_aug_prob=codec_aug_prob,
    )

    if val_configs is None:
        val_configs = train_configs

    val_ds = MultiDataset(
        val_configs,
        max_len=max_len,
        is_train=False,
        rawboost_algo=0,
        codec_aug_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
