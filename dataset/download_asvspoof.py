import os
import sys
import time
import math
import wave
import struct
import zipfile
import tarfile
import urllib.request
import argparse
from typing import Dict, List


def download_file_with_progress(url: str, output_path: str, max_retries: int = 20):
    """
    Downloads a large file with live progress updates, HTTP Range resumption, 
    and automatic reconnection retries to handle ConnectionResetError / network drops.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine total remote size
    req_head = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    total_size = 0
    try:
        with urllib.request.urlopen(req_head) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
    except Exception:
        pass

    total_size_mb = total_size / (1024 * 1024) if total_size > 0 else 0.0
    retries = 0
    block_size = 1024 * 1024  # 1 MB chunk size

    while retries < max_retries:
        downloaded = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        if total_size > 0 and downloaded >= total_size:
            print(f"\nFile already fully downloaded ({downloaded / (1024 * 1024):.1f} MB): {output_path}")
            return

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        if downloaded > 0:
            headers['Range'] = f'bytes={downloaded}-'
            print(f"\nResuming download from byte {downloaded} ({downloaded / (1024 * 1024):.1f} MB)...")

        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # If server responds with 206 Partial Content, append ('ab'). Otherwise, write fresh ('wb').
                is_partial = (resp.status == 206)
                if not is_partial:
                    downloaded = 0
                    mode = 'wb'
                else:
                    mode = 'ab'

                with open(output_path, mode) as f:
                    while True:
                        buffer = resp.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100.0
                            sys.stdout.write(f"\rProgress: {downloaded / (1024 * 1024):.1f} MB / {total_size_mb:.1f} MB ({percent:.1f}%) [Attempt {retries+1}/{max_retries}]")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB [Attempt {retries+1}/{max_retries}]")
                            sys.stdout.flush()

            print("\nDownload complete!")
            return

        except Exception as e:
            retries += 1
            sys.stdout.write(f"\nConnection interrupted ({type(e).__name__}: {e}). Retrying in 5 seconds... ({retries}/{max_retries})\n")
            sys.stdout.flush()
            time.sleep(5)

    print(f"\nFailed to download after {max_retries} attempts. You can also use curl to resume: curl -C - -L '{url}' -o '{output_path}'")


def extract_archive(file_path: str, extract_to: str):
    """Extracts .zip or .tar.gz archives with automatic corrupted file detection."""
    print(f"Extracting {file_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archive file not found: {file_path}")

    try:
        if file_path.endswith(".zip"):
            if not zipfile.is_zipfile(file_path):
                print(f"\nWarning: '{file_path}' is corrupted or incomplete. Removing corrupted file so it can be re-downloaded cleanly...")
                os.remove(file_path)
                raise zipfile.BadZipFile(f"Corrupted file removed: {file_path}. Please re-run the download script.")
                
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        print("Extraction complete!")
    except zipfile.BadZipFile:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise RuntimeError(f"Archive '{file_path}' was corrupted. The invalid file has been deleted. Please re-run the download command to download a clean copy.")



def create_mini_asvspoof_sample(dataset_dir: str):
    """
    Creates a mini ASVspoof dataset sample with real 16 kHz audio files 
    and protocol text files so training can be run immediately.
    """
    sample_dir = os.path.join(dataset_dir, "LA_sample")
    audio_dir = os.path.join(sample_dir, "ASVspoof2019_LA_train", "flac")
    protocol_dir = os.path.join(sample_dir, "ASVspoof2019_LA_cm_protocols")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(protocol_dir, exist_ok=True)
    
    protocol_file = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    
    sr = 16000
    duration_sec = 4.0
    num_samples = int(sr * duration_sec)
    
    protocol_lines = []
    print(f"Generating 20 mini ASVspoof 16kHz audio samples in {audio_dir}...")
    
    for i in range(20):
        audio_id = f"LA_T_{1000100 + i}"
        is_bonafide = (i % 2 == 0)
        key = "bonafide" if is_bonafide else "spoof"
        system_id = "-" if is_bonafide else f"A{((i // 2) % 19) + 1:02d}"
        
        # Protocol line format: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID - KEY
        line = f"LA_{i:04d} {audio_id} {system_id} - {key}\n"
        protocol_lines.append(line)
        
        # Write 16kHz audio waveform using stdlib wave
        wav_path = os.path.join(audio_dir, f"{audio_id}.wav")
        freq = 440.0 if is_bonafide else 880.0
        
        with wave.open(wav_path, 'w') as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)       # 16-bit PCM
            wav_file.setframerate(sr)      # 16000 Hz
            frames = []
            for s in range(num_samples):
                t = s / float(sr)
                val = int(32767.0 * 0.4 * math.sin(2.0 * math.pi * freq * t))
                frames.append(struct.pack('<h', val))
            wav_file.writeframesraw(b''.join(frames))
        
    with open(protocol_file, 'w') as f:
        f.writelines(protocol_lines)
        
    print(f"\nMini ASVspoof sample dataset ready at: {sample_dir}")
    print(f"Protocol file: {protocol_file}")
    print(f"Audio directory: {audio_dir}")
    return protocol_file, audio_dir


def download_asvspoof_dataset(dataset_type: str = "la_2019", dataset_dir: str = "./dataset"):
    """
    Downloads and extracts specified ASVspoof dataset.
    Options:
    - 'la_2019': ASVspoof 2019 Logical Access (Core Train/Dev) [7.6 GB]
    - 'la_2021': ASVspoof 2021 Logical Access Evaluation Set [7.2 GB]
    - 'df_2021': ASVspoof 2021 DeepFake Evaluation Set
    - 'sample':  Mini 16kHz ASVspoof dataset for instant local training & debugging
    """
    dataset_dir = os.path.abspath(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    if dataset_type == "sample":
        return create_mini_asvspoof_sample(dataset_dir)
        
    elif dataset_type == "la_2019":
        url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
        zip_path = os.path.join(dataset_dir, "LA.zip")
        extract_dir = os.path.join(dataset_dir, "LA")
        
        if not os.path.exists(zip_path):
            download_file_with_progress(url, zip_path)
        extract_archive(zip_path, extract_dir)
        
        protocol_file = os.path.join(extract_dir, "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
        audio_dir = os.path.join(extract_dir, "LA", "ASVspoof2019_LA_train", "flac")
        return protocol_file, audio_dir

    elif dataset_type == "la_2021":
        url = "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz"
        tar_path = os.path.join(dataset_dir, "ASVspoof2021_LA_eval.tar.gz")
        extract_dir = os.path.join(dataset_dir, "LA_2021")
        
        if not os.path.exists(tar_path):
            download_file_with_progress(url, tar_path)
        extract_archive(tar_path, extract_dir)
        return extract_dir, extract_dir

    elif dataset_type == "df_2021":
        parts = [
            "ASVspoof2021_DF_eval_part00.tar.gz",
            "ASVspoof2021_DF_eval_part01.tar.gz",
            "ASVspoof2021_DF_eval_part02.tar.gz",
            "ASVspoof2021_DF_eval_part03.tar.gz"
        ]
        extract_dir = os.path.join(dataset_dir, "DF_2021")
        for part in parts:
            url = f"https://zenodo.org/records/4835108/files/{part}"
            part_path = os.path.join(dataset_dir, part)
            if not os.path.exists(part_path):
                download_file_with_progress(url, part_path)
            extract_archive(part_path, extract_dir)
        return extract_dir, extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof Dataset Automated Downloader and Sample Generator")
    parser.add_argument("--target", type=str, default="sample", choices=["sample", "la_2019", "la_2021", "df_2021"],
                        help="Target dataset to download ('sample' for instant mini-dataset, 'la_2019' for core ASVspoof 2019)")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output dataset directory")
    
    args = parser.parse_args()
    download_asvspoof_dataset(dataset_type=args.target, dataset_dir=args.output_dir)
