"""Generate the RSA key pair (organizers only).

The public key is committed; the private key goes into the SUBMISSION_PRIVATE_KEY
GitHub secret and must never be committed.
"""
import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", type=Path, default=Path("encryption/public_key.pem"))
    ap.add_argument("--private", type=Path, default=Path("encryption/private_key.pem"))
    ap.add_argument("--key-size", type=int, default=4096)
    ap.add_argument("--force", action="store_true", help="overwrite existing keys")
    a = ap.parse_args()

    if a.private.exists() and not a.force:
        raise SystemExit(f"{a.private} already exists (use --force to overwrite; this invalidates "
                         "all encrypted submissions and requires re-running organizer/build_dataset.py).")

    key = rsa.generate_private_key(public_exponent=65537, key_size=a.key_size)
    a.private.parent.mkdir(parents=True, exist_ok=True)
    a.private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))
    a.public.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo))
    print(f"Public key : {a.public}  (commit this)")
    print(f"Private key: {a.private}  (put in GitHub secret SUBMISSION_PRIVATE_KEY, never commit)")
