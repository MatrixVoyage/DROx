"""Weight Management & Checkpoint Downloader for DeepFake Video Detection (DROx).

Supports downloading verified champion checkpoints from GitHub Releases or Hugging Face,
verifying cryptographic SHA-256 checksums, and generating local demo weights for offline testing.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_REGISTRY = {
    "v5-champion": {
        "rel_path": "checkpoints/m13_3_f2f_seed42/best.pth",
        "sha256": "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7",
        "url": "https://github.com/MatrixVoyage/DROx/releases/download/v1.0.0/m13_3_f2f_seed42_best.pth",
        "description": "Milestone 13.3/13.5 Family-Invariant Multi-Branch Champion (120 MB)",
    },
    "m18-dual-stream": {
        "rel_path": "checkpoints/m18_dual_stream_seed42/best.pth",
        "sha256": "fe5fec435b688525e19f6b533a8257f73a63a4446d574f674be42f3a587073ce",
        "url": "https://github.com/MatrixVoyage/DROx/releases/download/v1.0.0/m18_dual_stream_seed42_best.pth",
        "description": "Milestone 18 Kinematic Differential Dual-Stream Detector (139 MB)",
    },
    "v1-baseline": {
        "rel_path": "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth",
        "sha256": None,
        "url": "https://github.com/MatrixVoyage/DROx/releases/download/v1.0.0/ffpp_c23_baseline_soft_lstm_best.pth",
        "description": "Milestone 7 ResNeXt-50 + BiLSTM Soft-Attention Baseline (680 MB)",
    },
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hexadecimal digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def download_file(url: str, dest_path: Path) -> bool:
    """Downloads a file from url to dest_path with progress indication."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    print(f"\n[Connecting] {url} ...")
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB / {mb_tot:.1f} MB [{pct:.1f}%]")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB ...")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, str(tmp_path), reporthook=reporthook)
        print()
        if tmp_path.is_file() and tmp_path.stat().st_size > 1000:
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.rename(dest_path)
            return True
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            return False
    except Exception as e:
        print(f"\n[Warning] Remote download failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def check_status():
    """Prints the local presence and integrity status of all registered models."""
    print("\n" + "=" * 70)
    print("DROx CHECKPOINT INVENTORY & INTEGRITY STATUS")
    print("=" * 70)
    for name, item in CHECKPOINTS_REGISTRY.items():
        p = PROJECT_ROOT / item["rel_path"]
        print(f"\nModel: {name}")
        print(f"  Description: {item['description']}")
        print(f"  Target Path: {item['rel_path']}")
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  Status:      [PRESENT] ({size_mb:.1f} MB)")
            if item.get("sha256"):
                print("  Verifying cryptographic SHA-256 ... ", end="", flush=True)
                local_sha = compute_sha256(p)
                if local_sha == item["sha256"].lower():
                    print("[PASS] (Matches pre-registered paper audit)")
                else:
                    print(f"[MISMATCH]\n    Expected: {item['sha256']}\n    Found:    {local_sha}")
        else:
            print("  Status:      [MISSING]")
            print(f"  Download:    python scripts/download_weights.py --model {name}")


def generate_demo_weights(model_key: str = "v5-champion"):
    """Generates an initialized dummy checkpoint for immediate offline CLI/API testing."""
    item = CHECKPOINTS_REGISTRY.get(model_key, CHECKPOINTS_REGISTRY["v5-champion"])
    dest = PROJECT_ROOT / item["rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"[Notice] Checkpoint already exists at {dest}. Skipping demo generation.")
        return

    print(f"[Generating Demo Weights] Creating initialized state_dict for {model_key}...")
    from src.utils.config import load_config
    from src.models.detector import DeepfakeDetector
    cfg = load_config("configs/default.yaml")
    det = DeepfakeDetector(cfg.model)
    payload = {
        "model_state_dict": det.state_dict(),
        "epoch": 0,
        "config": cfg,
        "note": "Initialized demo weights generated by scripts/download_weights.py --generate-demo",
    }
    torch.save(payload, dest)
    print(f"[Success] Saved initialized demo weights to {dest} ({dest.stat().st_size / (1024*1024):.1f} MB).")


def main():
    parser = argparse.ArgumentParser(description="Download and verify DROx model checkpoints.")
    parser.add_argument(
        "--model",
        type=str,
        default="v5-champion",
        choices=list(CHECKPOINTS_REGISTRY.keys()),
        help="Model checkpoint to download (default: v5-champion).",
    )
    parser.add_argument("--all", action="store_true", help="Download all model checkpoints.")
    parser.add_argument("--check", action="store_true", help="Check local status and verify SHA-256 checksums.")
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate local initialized dummy weights for offline testing without network download.",
    )

    args = parser.parse_args()

    if args.check:
        check_status()
        return

    if args.generate_demo:
        models = list(CHECKPOINTS_REGISTRY.keys()) if args.all else [args.model]
        for m in models:
            generate_demo_weights(m)
        return

    targets = list(CHECKPOINTS_REGISTRY.keys()) if args.all else [args.model]

    for model_name in targets:
        item = CHECKPOINTS_REGISTRY[model_name]
        dest = PROJECT_ROOT / item["rel_path"]
        print(f"\nTarget: {model_name} -> {item['rel_path']}")
        if dest.is_file():
            print(f"  [OK] Local checkpoint already exists at {dest}.")
            if item.get("sha256"):
                print("  Verifying checksum ... ", end="", flush=True)
                local_sha = compute_sha256(dest)
                if local_sha == item["sha256"].lower():
                    print("[PASS]")
                else:
                    print(f"[MISMATCH]\n    Expected: {item['sha256']}\n    Found:    {local_sha}")
            continue

        print(f"  Attempting download from: {item['url']}")
        success = download_file(item["url"], dest)
        if success:
            print(f"  [Success] Downloaded {model_name} to {dest}.")
            if item.get("sha256"):
                local_sha = compute_sha256(dest)
                if local_sha == item["sha256"].lower():
                    print("  [Checksum Verified] SHA-256 matches official release.")
                else:
                    print("  [Warning] Checksum does not match official release.")
        else:
            print("\n" + "-" * 70)
            print(f"[Notice] Remote release asset not yet attached for '{model_name}'.")
            print("  If you are running from a freshly cloned repository, you can:")
            print("  1. Run in Demo Mode: The web dashboard and API automatically run with")
            print("     initialized weights without needing any downloaded file.")
            print(f"  2. Or generate demo weights locally: python scripts/download_weights.py --generate-demo --model {model_name}")
            print(f"  3. Or copy your trained 'best.pth' directly into:\n     {item['rel_path']}")
            print("-" * 70)


if __name__ == "__main__":
    main()
