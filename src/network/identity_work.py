"""Hashcash-style cost proof bound to one persistent Ed25519 identity."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path


MAX_IDENTITY_WORK_BITS = 28


def _digest(node_id: str, nonce: int, network_id: str = "LIXYMAIN") -> bytes:
    return hashlib.sha256(
        b"LixySwarm identity work v1\x00"
        + network_id.encode("ascii")
        + bytes.fromhex(node_id)
        + int(nonce).to_bytes(8, "big")
    ).digest()


def leading_zero_bits(value: bytes) -> int:
    total = 0
    for byte in value:
        if byte == 0:
            total += 8
            continue
        total += 8 - byte.bit_length()
        break
    return total


def verify_identity_work(
    node_id: str,
    proof: dict,
    *,
    minimum_bits: int,
    network_id: str = "LIXYMAIN",
) -> bool:
    try:
        if not isinstance(proof, dict) or set(proof) != {
            "version", "bits", "nonce", "digest"
        }:
            return False
        if proof["version"] != 1:
            return False
        bits = int(proof["bits"])
        nonce = int(proof["nonce"])
        if not minimum_bits <= bits <= MAX_IDENTITY_WORK_BITS:
            return False
        if not 0 <= nonce < 2 ** 64:
            return False
        digest = _digest(node_id, nonce, network_id)
        return (
            proof["digest"] == digest.hex()
            and leading_zero_bits(digest) >= bits
        )
    except (TypeError, ValueError, OverflowError):
        return False


def load_or_mine_identity_work(
    path: str | Path,
    node_id: str,
    *,
    bits: int,
    network_id: str = "LIXYMAIN",
) -> dict:
    bits = int(bits)
    if not 0 <= bits <= MAX_IDENTITY_WORK_BITS:
        raise ValueError("identity work bits are out of range")
    destination = Path(path)
    if destination.exists():
        try:
            value = json.loads(destination.read_text())
            proof = value.get("proof", {})
            if (
                value.get("node_id") == node_id
                and verify_identity_work(
                    node_id, proof, minimum_bits=bits, network_id=network_id
                )
            ):
                return proof
        except (OSError, json.JSONDecodeError):
            pass
    if bits == 0:
        nonce = 0
    else:
        nonce = secrets.randbits(64)
        while leading_zero_bits(_digest(node_id, nonce, network_id)) < bits:
            nonce = (nonce + 1) % (2 ** 64)
    digest = _digest(node_id, nonce, network_id)
    proof = {
        "version": 1,
        "bits": bits,
        "nonce": nonce,
        "digest": digest.hex(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "version": 1,
        "node_id": node_id,
        "network_id": network_id,
        "proof": proof,
    }, sort_keys=True, separators=(",", ":")))
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    return proof
