from __future__ import annotations

"""
Helper script to upload trained checkpoints and evaluation metrics to Hugging Face Hub.
Run this script on the server where checkpoints were generated:
    python upload_to_hf.py --hf_token <YOUR_HF_WRITE_TOKEN> --repo_id <YOUR_USERNAME>/ANLP-A1-Checkpoints
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face Hub")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="shardul0750/ANLP-A1-Checkpoints",
        help="Hugging Face repository ID (e.g. username/repo-name)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Local directory containing model checkpoints",
    )
    args = parser.parse_args()

    if not args.hf_token:
        print("Error: No Hugging Face token provided.")
        print("Please provide --hf_token <YOUR_TOKEN> or export HF_TOKEN='hf_...'")
        print("\nGet your Write token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.hf_token)

    print(f"1. Checking / creating Hugging Face repository: {args.repo_id}...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        print(f"   [OK] Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"   Notice during repo creation: {e}")

    print(f"\n2. Uploading checkpoints from '{args.ckpt_dir}'...")
    if not os.path.exists(args.ckpt_dir):
        print(f"Error: Directory '{args.ckpt_dir}' does not exist.")
        sys.exit(1)

    # List found checkpoints
    found_ckpts = []
    for root, dirs, files in os.walk(args.ckpt_dir):
        for f in files:
            if f.endswith(".pt") or f.endswith(".json"):
                rel_path = os.path.relpath(os.path.join(root, f), args.ckpt_dir)
                found_ckpts.append(rel_path)

    print(f"   Found {len(found_ckpts)} files to upload: {found_ckpts}")

    api.upload_folder(
        folder_path=args.ckpt_dir,
        repo_id=args.repo_id,
        repo_type="model",
        token=args.hf_token,
    )

    print("\n" + "=" * 60)
    print("SUCCESS: All checkpoints uploaded to Hugging Face Hub!")
    print(f"View model repository at: https://huggingface.co/{args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
