# Layer-Truncated Fake-Mamba for Real-Time Audio Deepfake Detection

A PyTorch repository for thesis experiments evaluating layer-truncated Fairseq XLSR-300M front-ends paired with downstream Fake-Mamba state-space architecture classifiers for real-time audio spoofing detection on ASVspoof datasets.

---

## 📁 Repository Structure

```
├── setup_env.sh              # Environment initialization & CUDA kernel setup script
├── train_ablation.py         # Main experiment runner (Baseline, Run A [12L], Run B [8L])
├── requirements.txt          # Core dependencies manifest
├── dataset/
│   ├── __init__.py
│   └── asvspoof_loader.py    # ASVspoof Dataset loader + 16kHz resampling + RawBoost
├── models/
│   ├── __init__.py
│   ├── truncated_xlsr.py     # TruncatedXLSR front-end wrapper with learnable aggregation
│   └── fake_mamba.py         # FakeMambaClassifier back-end (1024->144, Mamba SSM, Gated Pooling)
└── utils/
    ├── __init__.py
    └── profiler.py           # Isolated benchmarking script for GFLOPs and RTF
```

---

## 🚀 Key Technical Features

1. **Strict Dependency Order (`setup_env.sh`)**: Ensures PyTorch CUDA version matches system NVCC prior to installing `causal-conv1d` and `mamba-ssm`.
2. **Audio Resampling & RawBoost (`dataset/asvspoof_loader.py`)**: Audio files are strictly resampled to 16 kHz to align with XLSR-300M requirements. RawBoost data augmentation simulates VoIP compression artifacts, channel noise, and linear/non-linear filtering.
3. **Layer-Truncated XLSR Wrapper (`models/truncated_xlsr.py`)**: Uses `copy.deepcopy` to prevent base model mutation, physically slices encoder layers (`[:num_layers]`), learns softmax aggregation weights, and asserts a `(Batch, Time, 1024)` output tensor format.
4. **Fake-Mamba Classifier (`models/fake_mamba.py`)**: Downprojects features from 1024 to 144, applies Mamba selective state space blocks, performs Gated Attention Pooling over time, and projects to 2 classes (Bonafide vs. Spoof).
5. **Real-Time Profiler (`utils/profiler.py`)**: Measures GFLOPs using `ptflops` (via a 1D audio input constructor) and evaluates Real-Time Factor (RTF) across 100 trials after 20 CUDA-synchronized warm-up passes.
6. **Isolated Ablation Experiments (`train_ablation.py`)**: Runs three sequential experiments saving checkpoints to separate isolated directories:
   - **Baseline**: Full 24-layer XLSR (`./checkpoints/baseline/`)
   - **Run A**: Truncated 12-layer XLSR (`./checkpoints/run_a_12/`)
   - **Run B**: Truncated 8-layer XLSR (`./checkpoints/run_b_8/`)

---

## 🛠️ Environment Setup

Run the initialization script:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## ⚡ Quick Start & Verification

Run the profiler self-test:

```bash
python -m utils.profiler
```

Run the complete ablation experiment:

```bash
python train_ablation.py --epochs 3 --batch_size 8 --run_mode all
```

To run a specific ablation run (e.g., Run A with 12 layers):

```bash
python train_ablation.py --epochs 5 --batch_size 16 --run_mode run_a_12
```

---

## 📊 Performance Benchmark Output

The profiler outputs key thesis metrics:
- **GFLOPs**: Total floating point operations for a 4.0-second audio clip.
- **RTF (Real-Time Factor)**: $\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$. $\text{RTF} < 1.0$ indicates real-time processing capability.
