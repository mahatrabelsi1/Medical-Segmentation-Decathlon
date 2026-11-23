"""Hybrid RSA-OAEP + Fernet encryption used for submissions and hidden labels."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def normalize_pem(key: bytes) -> bytes:
    """Repair common copy/paste damage to a PEM key (literal "\\n", CRLF, quotes, surrounding spaces)."""
    k = key.strip().strip(b'"').strip(b"'")
    k = k.replace(b"\\r", b"").replace(b"\\n", b"\n").replace(b"\r", b"")
    lines = [ln.strip() for ln in k.split(b"\n") if ln.strip()]
    return b"\n".join(lines) + b"\n"


def encrypt_bytes(data: bytes, public_key_pem: bytes) -> bytes:
    public_key = serialization.load_pem_public_key(public_key_pem)
    sym = Fernet.generate_key()
    payload = {
        "version": 1,
        "alg": "RSA_OAEP_SHA256+FERNET",
        "enc_key_b64": base64.b64encode(public_key.encrypt(sym, _OAEP)).decode("ascii"),
        "ciphertext_b64": base64.b64encode(Fernet(sym).encrypt(data)).decode("ascii"),
    }
    return json.dumps(payload).encode("utf-8")


def decrypt_bytes(blob: bytes, private_key_pem: bytes) -> bytes:
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported encrypted payload version")
    private_key = serialization.load_pem_private_key(normalize_pem(private_key_pem), password=None)
    sym = private_key.decrypt(base64.b64decode(payload["enc_key_b64"]), _OAEP)
    return Fernet(sym).decrypt(base64.b64decode(payload["ciphertext_b64"]))


def encrypt_file(src: Path, public_key: Path, dst: Path) -> None:
    dst.write_bytes(encrypt_bytes(Path(src).read_bytes(), Path(public_key).read_bytes()))


def decrypt_file(src: Path, private_key: Path, dst: Path) -> None:
    dst.write_bytes(decrypt_bytes(Path(src).read_bytes(), Path(private_key).read_bytes()))
