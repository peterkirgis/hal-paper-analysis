import base64, json, tempfile, zipfile
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from huggingface_hub import HfFileSystem

# ---- Defaults (edit if needed) ----
DEFAULT_PASSWORD = "hal1234"
DEFAULT_REPO_ID = "agent-evals/hal_traces"
DEFAULT_REVISION = "main"

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def _decrypt_token_bytes(encrypted_data_b64: str, salt_b64: str, password: str) -> bytes:
    ct = base64.b64decode(encrypted_data_b64)
    salt = base64.b64decode(salt_b64)
    f = Fernet(_derive_key(password, salt))
    return f.decrypt(ct)

def hf_download_decrypt_to_tempfile(
    zip_name: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    password: str = DEFAULT_PASSWORD,
    member_name: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """
    Download a ZIP stored in a Hugging Face *dataset* repo, read the encrypted JSON
    container inside, decrypt it, and write the plaintext JSON to a NamedTemporaryFile.

    Args:
        zip_name:        File name inside the repo (e.g. 'assistantbench_...._UPLOAD.zip')
        repo_id:         Dataset repo id (owner/repo)
        revision:        Branch/tag/commit (e.g. 'main')
        password:        Password used to derive the Fernet key (matches uploader)
        member_name:     Optional specific member filename inside the ZIP. If None, uses the first file entry.
        hf_token:        Optional HF token; if None, uses env var (HUGGING_FACE_HUB_TOKEN)

    Returns:
        Path to a temporary .json file containing the decrypted plaintext.
    """
    fs = HfFileSystem(token=hf_token)
    repo_path = f"datasets/{repo_id}@{revision}"
    full_path = f"{repo_path}/{zip_name}"

    # Open the remote ZIP and load the JSON "container" member
    with fs.open(full_path, "rb") as hf_file, zipfile.ZipFile(hf_file) as zf:
        if member_name is not None:
            info = zf.getinfo(member_name)
        else:
            # Pick the first non-directory entry
            info = next(i for i in zf.infolist() if not i.is_dir())
        with zf.open(info, "r") as member:
            container = json.load(member)

    # Decrypt to plaintext bytes
    plaintext = _decrypt_token_bytes(
        container["encrypted_data"],
        container["salt"],
        password,
    )

    # Write plaintext to a tempfile and return its path
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with tf:
        tf.write(plaintext)
        tf.flush()
        return tf.name