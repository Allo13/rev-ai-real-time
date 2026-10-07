#!/usr/bin/env bash
# download_weights.sh: Automated script to fetch XLSR-300M weights and ASVspoof datasets

set -e

WEIGHTS_DIR="./weights"
DATASET_DIR="./dataset"

mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${DATASET_DIR}"

XLSR_URL="https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt"
XLSR_TARGET="${WEIGHTS_DIR}/xlsr_53_56k.pt"

echo "=== [1/2] Downloading Official Fairseq XLSR-300M Weights ==="
if [ -f "${XLSR_TARGET}" ]; then
    echo "Checkpoint already exists at ${XLSR_TARGET}. Skipping download."
else
    echo "Downloading ${XLSR_URL} to ${XLSR_TARGET}..."
    if command -v wget &> /dev/null; then
        wget "${XLSR_URL}" -O "${XLSR_TARGET}"
    elif command -v curl &> /dev/null; then
        curl -L "${XLSR_URL}" -o "${XLSR_TARGET}"
    else
        echo "ERROR: Neither wget nor curl found. Please manually download ${XLSR_URL} to ${XLSR_TARGET}"
        exit 1
    fi
    echo "Download completed: ${XLSR_TARGET}"
fi

echo "=== [2/2] ASVspoof Dataset Setup & Download ==="
echo "Generating mini-sample ASVspoof dataset for instant local testing..."
python dataset/download_asvspoof.py --target sample --output_dir ./dataset

echo ""
echo "To download full datasets automatically:"
echo "  Core Training Dataset (ASVspoof 2019 LA - 7.6 GB):"
echo "    python dataset/download_asvspoof.py --target la_2019"
echo ""
echo "  ASVspoof 2021 LA Evaluation Set (7.2 GB):"
echo "    python dataset/download_asvspoof.py --target la_2021"
echo ""
echo "  ASVspoof 2021 DF Evaluation Set:"
echo "    python dataset/download_asvspoof.py --target df_2021"
echo ""

echo "=== Setup & Download Script Completed ==="
