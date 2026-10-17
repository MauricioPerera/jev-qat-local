"""Verified resumable HTTP download of the official E2B QAT Q4 checkpoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_gemma_e4b_qat as download

download.REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
download.REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
download.DESTINATION = download.ROOT / "models" / "gemma-4-E2B-it-QAT-Q4_0"
download.STATE = download.ROOT / "compatibility" / "e2b-qat-q4"
download.PROGRESS = download.STATE / "download-progress.json"
download.RECEIPT = download.STATE / "download-receipt.json"
download.FILES = {
    "gemma-4-E2B_q4_0-it.gguf": {
        "size": 3349516256,
        "sha256": "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
    },
    "gemma-4-E2B-it-mmproj.gguf": {
        "size": 986833664,
        "sha256": "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14",
    },
}

if __name__ == "__main__":
    download.main()
