"""Personal state encryption. An unavailable OS key store has no file fallback."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


class PersonalStorageUnavailable(RuntimeError):
    pass


def os_backend():
    import keyring
    backend = keyring.get_keyring()
    approved = ("keyring.backends.SecretService", "keyring.backends.Windows", "keyring.backends.macOS")
    if not type(backend).__module__.startswith(approved):
        raise PersonalStorageUnavailable("El almacén protegido del sistema no está disponible")
    if type(backend).__module__.startswith("keyring.backends.SecretService"):
        import secretstorage
        bus = secretstorage.dbus_init()
        collection = secretstorage.Collection(bus, backend.preferred_collection) if hasattr(backend, "preferred_collection") else secretstorage.get_collection_by_alias(bus, "default")
        if collection.is_locked():
            raise PersonalStorageUnavailable("El almacén protegido del sistema está bloqueado")
    return backend


class PersonalStore:
    def __init__(self, directory: Path, *, backend=None):
        self.directory = directory
        try:
            backend = os_backend() if backend is None else backend
            account = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
            stored = backend.get_password("org.lixyswarm.personal.v1", account)
            if stored is None:
                if (directory / "personal.enc").exists():
                    raise PersonalStorageUnavailable("Falta la clave de la memoria personal existente")
                stored = base64.b64encode(os.urandom(32)).decode("ascii")
                backend.set_password("org.lixyswarm.personal.v1", account, stored)
            self.key = base64.b64decode(stored, validate=True)
            if len(self.key) != 32:
                raise PersonalStorageUnavailable("La clave personal protegida no es válida")
        except PersonalStorageUnavailable:
            raise
        except Exception as exc:
            raise PersonalStorageUnavailable("No se pudo abrir el almacenamiento personal protegido") from exc

    def save(self, value: dict):
        from src.network.telemetry import atomic_json
        nonce = os.urandom(12)
        encoded=json.dumps(value).encode()
        if len(encoded)>2*1024**2:raise ValueError("El contenido personal supera su límite local")
        ciphertext = ChaCha20Poly1305(self.key).encrypt(nonce, encoded, b"LixySwarm-personal-v1")
        atomic_json(self.directory / "personal.enc", {"version": 1, "data": base64.b64encode(nonce + ciphertext).decode()})

    def load(self):
        path = self.directory / "personal.enc"
        if not path.exists():
            return {}
        if path.stat().st_size>3*1024**2:raise ValueError("El almacenamiento personal supera su límite local")
        value = json.loads(path.read_text())
        if value.get("version") != 1:
            raise ValueError("Unsupported personal storage version")
        data = base64.b64decode(value["data"], validate=True)
        return json.loads(ChaCha20Poly1305(self.key).decrypt(data[:12], data[12:], b"LixySwarm-personal-v1"))
