#!/usr/bin/env bash
# setup_env.sh: Environment Initialization Script for Truncated XLSR + Fake-Mamba
# Strict dependency resolution order to prevent Mamba C++ / CUDA compilation failures.

set -e # Exit immediately on error

echo "=== [1/5] Creating Python Virtual Environment ==="
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi

if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Could not find virtualenv activation script in venv/Scripts or venv/bin."
    exit 1
fi

echo "=== [2/5] Installing PyTorch with Explicit CUDA Bindings ==="
# Adjust CUDA index url if targeting cu121 vs cu118 based on system nvcc
CUDA_VERSION="cu118"
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "Detected NVCC Version: ${NVCC_VER}"
    if [[ "$NVCC_VER" == "12."* ]]; then
        CUDA_VERSION="cu121"
    fi
fi

echo "Installing PyTorch for CUDA bindings (${CUDA_VERSION})..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

echo "=== [3/5] Verifying NVCC and PyTorch CUDA Compatibility ==="
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch Built-in CUDA Version: {torch.version.cuda}')
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
"

if command -v nvcc &> /dev/null; then
    echo "System NVCC info:"
    nvcc --version
else
    echo "WARNING: nvcc command not found in PATH. Building mamba-ssm/causal-conv1d CUDA extensions from source may require NVCC."
fi

echo "=== [4/5] Installing Causal-Conv1D and Mamba-SSM ==="
# causal-conv1d must be installed prior to mamba-ssm
python -m pip install causal-conv1d>=1.2.0 --no-build-isolation || {
    echo "Pre-built wheel for causal-conv1d unavailable or build failed. Attempting standard pip install..."
    python -m pip install causal-conv1d>=1.2.0 || true
}

python -m pip install mamba-ssm>=1.2.0 --no-build-isolation || {
    echo "Pre-built wheel for mamba-ssm unavailable or build failed. Attempting standard pip install..."
    python -m pip install mamba-ssm>=1.2.0 || true
}

echo "=== [5/5] Installing Core & Evaluation Dependencies ==="
python -m pip install fairseq || true
python -m pip install soundfile librosa ptflops tensorboard tqdm scipy scikit-learn matplotlib

echo "=== Environment Setup Complete! ==="
