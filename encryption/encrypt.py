"""Encrypt a file with the competition public key.

    python encryption/encrypt.py predictions.npz encryption/public_key.pem predictions.npz.enc
"""
import argparse
from pathlib import Path

from crypto import encrypt_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("public_key_pem", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    encrypt_file(a.input, a.public_key_pem, a.output)
    print(f"Encrypted file written to: {a.output}")
