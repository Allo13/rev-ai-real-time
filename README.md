# Layer-Truncated Fake-Mamba for Real-Time Audio Deepfake Detection

A PyTorch repository for thesis experiments evaluating layer-truncated Fairseq XLSR-300M front-ends paired with downstream Fake-Mamba state-space architecture classifiers for real-time audio spoofing detection on ASVspoof datasets.

---

## 📁 Repository Structure

```
├── setup_env.sh              # Environment initialization & CUDA kernel setup script
├── download_weights.sh       # Automated downloader for XLSR-300M weights & sample dataset
├── train_ablation.py         # Main experiment runner (Baseline, Run A [12L], Run B [8L])
├── requirements.txt          # Core dependencies manifest
├── dataset/
│   ├── __init__.py
│   ├── asvspoof_loader.py    # ASVspoof Dataset loader + 16kHz resampling + RawBoost
│   └── download_asvspoof.py  # Downloader for sample/full ASVspoof datasets
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

### 1. Prerequisites & Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 2. Run Automated Environment Setup

Execute the environment initialization script to install dependencies and compile CUDA extensions:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Or install base packages via `pip`:

```bash
pip install -r requirements.txt
```

---

## 📥 Downloading Weights & Datasets

### Automatic Download Script

Run the automated download script to fetch the official Fairseq XLSR-300M pre-trained weights and generate a local mini-sample ASVspoof dataset:

```bash
chmod +x download_weights.sh
./download_weights.sh
```

### Downloading Full ASVspoof Datasets

To download full datasets for full-scale training or evaluation, use `dataset/download_asvspoof.py`:

- **Mini Sample Dataset (Instant local test & debug)**:
  ```bash
  python dataset/download_asvspoof.py --target sample --output_dir ./dataset
  ```
- **ASVspoof 2019 LA (Core Training Set - 7.6 GB)**:
  ```bash
  python dataset/download_asvspoof.py --target la_2019 --output_dir ./dataset
  ```
- **ASVspoof 2021 LA Evaluation Set (7.2 GB)**:
  ```bash
  python dataset/download_asvspoof.py --target la_2021 --output_dir ./dataset
  ```
- **ASVspoof 2021 DF Evaluation Set**:
  ```bash
  python dataset/download_asvspoof.py --target df_2021 --output_dir ./dataset
  ```

---

## 💻 How to Execute Experiments

### 1. Standalone Real-Time Performance Profiling

To benchmark Real-Time Factor (RTF) and GFLOPs across truncated XLSR layers without running model training:

```bash
python -m utils.profiler
```

---

### 2. Running Training & Ablation Experiments

The main training script `train_ablation.py` runs experiments and saves checkpoints, TensorBoard logs, and summary JSON metrics.

#### Option A: Run Full Ablation Benchmark (Baseline, 12L, and 8L)

Run all three ablation configurations sequentially:

```bash
python train_ablation.py --epochs 5 --batch_size 8 --run_mode all
```

#### Option B: Run a Specific Ablation Model

- **Baseline (Full 24-Layer XLSR)**:
  ```bash
  python train_ablation.py --epochs 5 --batch_size 8 --run_mode baseline
  ```
- **Run A (12-Layer Truncated XLSR)**:
  ```bash
  python train_ablation.py --epochs 5 --batch_size 16 --run_mode run_a_12
  ```
- **Run B (8-Layer Truncated XLSR)**:
  ```bash
  python train_ablation.py --epochs 5 --batch_size 16 --run_mode run_b_8
  ```

---

### 3. Command Line Arguments Reference (`train_ablation.py`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--run_mode` | `str` | `all` | Mode: `all`, `baseline`, `run_a_12`, or `run_b_8` |
| `--epochs` | `int` | `5` | Number of training epochs per run |
| `--batch_size` | `int` | `8` | Batch size for training & evaluation |
| `--lr` | `float` | `1e-4` | Initial learning rate for AdamW optimizer |
| `--weights_path` | `str` | `./weights/xlsr_53_56k.pt` | Path to Fairseq XLSR-300M weights file |
| `--protocol_file` | `str` | `None` | Path to custom ASVspoof protocol text file |
| `--audio_dir` | `str` | `None` | Path to custom audio directory (`.flac` or `.wav`) |
| `--device` | `str` | `cuda` (or `cpu`) | Computing device target |

---

## 📊 Performance Benchmark & Thesis Metrics Output

After running experiments, output checkpoints and metrics are saved to:
- `./checkpoints/baseline/run_summary.json`
- `./checkpoints/run_a_12/run_summary.json`
- `./checkpoints/run_b_8/run_summary.json`
- `./checkpoints/ablation_study_summary.json`

Key evaluated thesis metrics include:
- **GFLOPs**: Floating point operations for a 4.0-second audio clip.
- **RTF (Real-Time Factor)**: $\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$. $\text{RTF} < 1.0$ indicates real-time processing capability.
- **EER (%)**: Equal Error Rate on validation set.
- **Accuracy (%)**: Bonafide vs. Spoof binary classification accuracy.
