"""
Download Utility for HAL Traces from Hugging Face

This is a standalone utility to download and decrypt HAL traces from the Hugging Face repository.
No abstractions - just a simple function to download files.
"""

import base64
import json
import os
import tempfile
import zipfile
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from huggingface_hub import HfFileSystem


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PASSWORD = "hal1234"
DEFAULT_REPO_ID = "agent-evals/hal_traces"
DEFAULT_REVISION = "main"


# ============================================================================
# DECRYPTION UTILITIES
# ============================================================================


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def decrypt_data(encrypted_data_b64: str, salt_b64: str, password: str) -> bytes:
    """Decrypt encrypted data using password and salt."""
    ct = base64.b64decode(encrypted_data_b64)
    salt = base64.b64decode(salt_b64)
    f = Fernet(derive_key(password, salt))
    return f.decrypt(ct)


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================


def download_and_decrypt_file(
    zip_filename: str,
    output_path: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    password: str = DEFAULT_PASSWORD,
    hf_token: Optional[str] = None,
) -> str:
    """
    Download a ZIP file from Hugging Face, decrypt it, and save to output path.

    Args:
        zip_filename: Name of the ZIP file in the repo (e.g., 'scienceagentbench_...._UPLOAD.zip')
        output_path: Where to save the decrypted JSON file
        repo_id: Hugging Face dataset repository ID
        revision: Branch/tag/commit to download from
        password: Password for decryption
        hf_token: Optional Hugging Face token (uses env var if None)

    Returns:
        Path to the saved JSON file
    """
    print(f"📥 Downloading {zip_filename}...")

    fs = HfFileSystem(token=hf_token)
    repo_path = f"datasets/{repo_id}@{revision}"
    full_path = f"{repo_path}/{zip_filename}"

    # Download and decrypt
    try:
        with fs.open(full_path, "rb") as hf_file, zipfile.ZipFile(hf_file) as zf:
            # Get the first file in the ZIP
            info = next(i for i in zf.infolist() if not i.is_dir())

            with zf.open(info, "r") as member:
                container = json.load(member)

        # Decrypt
        plaintext = decrypt_data(
            container["encrypted_data"], container["salt"], password
        )

        # Save to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(plaintext)

        print(f"✅ Downloaded and saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error downloading {zip_filename}: {e}")
        raise


def download_benchmark_files(
    benchmark_name: str,
    output_directory: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    password: str = DEFAULT_PASSWORD,
    hf_token: Optional[str] = None,
    max_files: Optional[int] = None,
) -> list[str]:
    """
    Download all files for a specific benchmark from Hugging Face.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'scienceagentbench')
        output_directory: Directory to save files
        repo_id: Hugging Face dataset repository ID
        revision: Branch/tag/commit to download from
        password: Password for decryption
        hf_token: Optional Hugging Face token (uses env var if None)
        max_files: Maximum number of files to download (None = all)

    Returns:
        List of paths to downloaded files
    """
    print(f"\n🔍 Finding {benchmark_name} files in repository...")

    fs = HfFileSystem(token=hf_token)
    repo_path = f"datasets/{repo_id}@{revision}"

    # List all files in the repo
    try:
        all_files = fs.ls(repo_path, detail=False)
    except Exception as e:
        print(f"❌ Error listing repository: {e}")
        return []

    # Filter for benchmark files
    benchmark_files = [
        f.split("/")[-1]
        for f in all_files
        if benchmark_name in f and f.endswith("_UPLOAD.zip")
    ]

    print(f"   Found {len(benchmark_files)} files")

    if max_files:
        benchmark_files = benchmark_files[:max_files]
        print(f"   Limiting to {max_files} files")

    downloaded_paths = []

    for zip_file in benchmark_files:
        # Create output path
        json_filename = zip_file.replace(".zip", ".json")
        output_path = os.path.join(output_directory, json_filename)

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"⏭️  Skipping {json_filename} (already exists)")
            downloaded_paths.append(output_path)
            continue

        try:
            path = download_and_decrypt_file(
                zip_filename=zip_file,
                output_path=output_path,
                repo_id=repo_id,
                revision=revision,
                password=password,
                hf_token=hf_token,
            )
            downloaded_paths.append(path)
        except Exception as e:
            print(f"⚠️  Failed to download {zip_file}: {e}")
            continue

    return downloaded_paths


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Example usage of download utilities."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download HAL traces from Hugging Face"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name (e.g., scienceagentbench)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hal_traces",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum number of files to download"
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"{args.benchmark}_data")

    downloaded = download_benchmark_files(
        benchmark_name=args.benchmark,
        output_directory=output_dir,
        max_files=args.max_files,
    )

    print(f"\n✅ Downloaded {len(downloaded)} files to {output_dir}")


if __name__ == "__main__":
    main()



