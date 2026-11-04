# Training Guide — Reveal AI

How to train the model better for improved real-world deepfake detection.

## Current Setup

- **Architecture**: Truncated XLSR-300M (8 frozen layers) + Fake-Mamba SSM (~0.44M trainable params)
- **Dataset**: ASVspoof 2019 LA (train split)
- **Results**: 0% EER, 100% accuracy, 84ms latency on ASVspoof 2019 LA eval set

The model performs perfectly on ASVspoof 2019, but may struggle with modern deepfakes (ElevenLabs, Bark, XTTS, etc.) since it only trained on older attack types (A01–A19).

---

## 1. Add More Diverse Datasets

The single biggest improvement. Download and add these alongside ASVspoof 2019:

| Dataset | Why | Link |
|---------|-----|------|
| **ASVspoof 2021 LA** | Adds codec/compression artifacts | https://www.asvspoof.org/ |
| **In-The-Wild** | Real YouTube deepfakes, not lab-generated | https://deepfake-demo.aisec.fraunhofer.de/in_the_wild |
| **WaveFake** | Modern neural vocoders (MelGAN, WaveGlow, HiFi-GAN) | https://github.com/RUB-SysSec/WaveFake |
| **MLAAD** | Multilingual deepfakes (not just English) | https://github.com/Nicolas-Balmelli/MLAAD |

**How to add a new dataset:**

1. Download and extract to `dataset/<name>/`
2. Create a loader in `dataset/` similar to `asvspoof_loader.py`
3. Use `torch.utils.data.ConcatDataset` to combine datasets in `train_ablation.py`:

```python
from torch.utils.data import ConcatDataset

asvspoof_ds = ASVspoofDataset(...)
wavefake_ds = WaveFakeDataset(...)  # you write this
combined = ConcatDataset([asvspoof_ds, wavefake_ds])
train_loader = DataLoader(combined, batch_size=16, shuffle=True)
```

---

## 2. Add Data Augmentation

Real-world audio has noise, compression, and reverb. Install:

```bash
pip install torch-audiomentations
```

Add augmentations in your dataset loader's `__getitem__`:

```python
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, Gain
import torchaudio

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, noise_dir="dataset/noise/"):
        self.base = base_dataset
        self.augment = Compose([
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
            # Download MUSAN noise dataset for this:
            # AddBackgroundNoise(noise_dir, min_snr_in_db=5, max_snr_in_db=20, p=0.5),
        ])

    def __getitem__(self, idx):
        waveform, label = self.base[idx]
        # waveform shape: (1, num_samples)
        if self.training:
            waveform = self.augment(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
        return waveform, label
```

You can also add **codec simulation** (very effective):

```python
import subprocess
import tempfile
import torchaudio

def simulate_codec(waveform, sr=16000):
    """Compress through MP3 to simulate WhatsApp/Zoom quality."""
    with tempfile.NamedTemporaryFile(suffix='.wav') as f_in, \
         tempfile.NamedTemporaryFile(suffix='.mp3') as f_mp3, \
         tempfile.NamedTemporaryFile(suffix='.wav') as f_out:
        torchaudio.save(f_in.name, waveform, sr)
        subprocess.run(['ffmpeg', '-y', '-i', f_in.name, '-b:a', '32k', f_mp3.name],
                       capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-i', f_mp3.name, f_out.name],
                       capture_output=True)
        compressed, _ = torchaudio.load(f_out.name)
    return compressed
```

---

## 3. Partially Unfreeze XLSR

Currently all XLSR layers are frozen. Unfreezing the top 2–3 layers lets the model learn deepfake-specific features:

In `train_ablation.py`, after creating the truncated XLSR model, add:

```python
# Freeze all XLSR layers first
for param in truncated_xlsr.parameters():
    param.requires_grad = False

# Unfreeze last 2 encoder layers
for layer in truncated_xlsr.encoder_layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
```

Use different learning rates:

```python
optimizer = torch.optim.AdamW([
    {'params': mamba_classifier.parameters(), 'lr': 1e-4},
    {'params': [p for p in truncated_xlsr.parameters() if p.requires_grad], 'lr': 1e-5},
], weight_decay=0.01)
```

---

## 4. Train Longer with Better Scheduling

Current training: 5 epochs. Increase to 20–30 with cosine annealing:

```python
epochs = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

for epoch in range(epochs):
    train_one_epoch(...)
    val_metrics = validate(...)
    scheduler.step()

    # Early stopping
    if val_metrics['eer'] < best_eer:
        best_eer = val_metrics['eer']
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print("Early stopping")
            break
```

---

## 5. Run Training

Basic training (current):

```bash
python train_ablation.py --layers 8 --epochs 5
```

Improved training (after making changes above):

```bash
python train_ablation.py --layers 8 --epochs 25
```

Monitor with TensorBoard:

```bash
tensorboard --logdir checkpoints/
```

---

## 6. Test on Real-World Deepfakes

After training, test with voices the model has never seen:

1. Generate a voice clip on https://elevenlabs.io (free tier)
2. Download it as MP3/WAV
3. Upload to Reveal AI's Detect tab
4. If the model says "real" → it needs more diverse training data (Step 1)

---

## Hardware Notes

- **GTX 1050 (4GB)**: Enough for frozen XLSR + Mamba training. If you unfreeze XLSR layers, reduce batch size to 8 or 4
- **Training time**: ~15 min per epoch on ASVspoof 2019 LA with frozen XLSR
- **Disk**: Dataset is ~25GB. Keep everything on D: drive
