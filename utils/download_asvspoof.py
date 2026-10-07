import os
import sys
import json
import urllib.request
import urllib.parse
from typing import Dict, List

def get_zenodo_files(record_id: str) -> List[Dict[str, str]]:
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    files_info = []
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            for f in data.get("files", []):
                files_info.append({
                    "filename": f.get("key"),
                    "size_gb": f.get("size", 0) / (1024 ** 3),
                    "download_url": f.get("links", {}).get("self")
                })
    except Exception as e:
        print(f"Error querying Zenodo record {record_id}: {e}")
    return files_info

def get_edinburgh_links() -> List[Dict[str, str]]:
    # Standard bitstream links for ASVspoof 2019 (Handle 10283/3336)
    base_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/"
    files = [
        {"filename": "LA.zip", "description": "ASVspoof 2019 Logical Access (LA) [Train/Dev/Eval]", "url": base_url + "LA.zip"},
        {"filename": "PA.zip", "description": "ASVspoof 2019 Physical Access (PA)", "url": base_url + "PA.zip"}
    ]
    return files

if __name__ == "__main__":
    print("=== ASVspoof 2019 (Edinburgh DataShare) ===")
    for item in get_edinburgh_links():
        print(f"  {item['filename']}: {item['url']}")

    print("\n=== ASVspoof 2021 LA Eval (Zenodo 4837263) ===")
    for f in get_zenodo_files("4837263"):
        print(f"  {f['filename']} ({f['size_gb']:.2f} GB): {f['download_url']}")

    print("\n=== ASVspoof 2021 DF Eval (Zenodo 4835108) ===")
    for f in get_zenodo_files("4835108"):
        print(f"  {f['filename']} ({f['size_gb']:.2f} GB): {f['download_url']}")
