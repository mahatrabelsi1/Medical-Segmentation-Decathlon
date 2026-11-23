"""Decrypt a file with the competition private key (organizers / CI only)."""
import argparse
from pathlib import Path

from crypto import decrypt_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("private_key_pem", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    decrypt_file(a.input, a.private_key_pem, a.output)
    print(f"Decrypted file written to: {a.output}")
