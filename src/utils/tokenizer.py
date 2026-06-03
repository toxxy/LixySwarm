"""
Carga robusta del tokenizer GPT-2 para Lixy.

`tiktoken.get_encoding("gpt2")` descarga assets desde Azure cuando no hay cache.
Si Azure no responde, el arranque del chat puede quedarse bloqueado antes de
cargar el checkpoint. Este modulo precalienta el cache desde HuggingFace, que
sirve los mismos archivos con los hashes oficiales que valida tiktoken.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "checkpoints" / "tiktoken_cache"

GPT2_ASSETS = {
    "vocab_bpe": {
        "official_url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "mirror_url": "https://huggingface.co/gpt2/resolve/main/merges.txt",
        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    },
    "encoder_json": {
        "official_url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "mirror_url": "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    },
}


class TokenizerLoadError(RuntimeError):
    """Error accionable al cargar el tokenizer GPT-2."""


def _cache_path(cache_dir: Path, official_url: str) -> Path:
    return cache_dir / hashlib.sha1(official_url.encode()).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_valid_cached_asset(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    return _sha256(path.read_bytes()) == expected_hash


def gpt2_cache_ready(cache_dir: str | Path | None = None) -> bool:
    """Retorna True si el cache local contiene los dos assets GPT-2 validos."""
    cache_path = Path(cache_dir or DEFAULT_CACHE_DIR)
    return all(
        _read_valid_cached_asset(
            _cache_path(cache_path, asset["official_url"]),
            asset["sha256"],
        )
        for asset in GPT2_ASSETS.values()
    )


def warm_gpt2_tokenizer_cache(
    cache_dir: str | Path | None = None,
    timeout_s: float = 20.0,
) -> Path:
    """
    Descarga los assets GPT-2 equivalentes desde HuggingFace al cache de tiktoken.

    Los nombres de archivo son los SHA1 de las URLs oficiales, porque ese es el
    contrato interno de cache que usa `tiktoken.load.read_file_cached`.
    """
    cache_path = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    for asset in GPT2_ASSETS.values():
        destination = _cache_path(cache_path, asset["official_url"])
        if _read_valid_cached_asset(destination, asset["sha256"]):
            continue

        try:
            with urlopen(asset["mirror_url"], timeout=timeout_s) as response:
                data = response.read()
        except (OSError, URLError) as exc:
            raise TokenizerLoadError(
                "No pude descargar los assets del tokenizer GPT-2 desde HuggingFace. "
                f"Reintenta con red o copia un cache valido en {cache_path}."
            ) from exc

        actual_hash = _sha256(data)
        if actual_hash != asset["sha256"]:
            raise TokenizerLoadError(
                "El asset GPT-2 descargado no coincide con el hash esperado "
                f"({actual_hash} != {asset['sha256']})."
            )

        fd, tmp_name = tempfile.mkstemp(prefix=f"{destination.name}.", suffix=".tmp", dir=cache_path)
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(data)
            os.replace(tmp_name, destination)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    return cache_path


def get_gpt2_encoding(
    cache_dir: str | Path | None = None,
    allow_download: bool | None = None,
):
    """
    Devuelve el encoding GPT-2 de tiktoken sin depender de Azure en caliente.

    Por default intenta calentar cache desde HuggingFace si faltan archivos. Para
    desactivar esa descarga, define `LIXY_TOKENIZER_AUTO_DOWNLOAD=0`.
    """
    cache_path = Path(cache_dir or os.environ.get("TIKTOKEN_CACHE_DIR") or DEFAULT_CACHE_DIR)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_path)

    if allow_download is None:
        allow_download = os.environ.get("LIXY_TOKENIZER_AUTO_DOWNLOAD", "1") != "0"

    if not gpt2_cache_ready(cache_path):
        if allow_download:
            warm_gpt2_tokenizer_cache(cache_path)
        else:
            raise TokenizerLoadError(
                "Falta el cache local del tokenizer GPT-2 y la descarga automatica esta desactivada. "
                f"Ejecuta: python3 -m src.utils.tokenizer --cache-dir {cache_path}"
            )

    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2")
    except Exception as exc:
        raise TokenizerLoadError(
            "No pude inicializar tiktoken con el cache GPT-2 local. "
            f"Cache usado: {cache_path}"
        ) from exc


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Precalienta el cache GPT-2 de tiktoken")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    cache_path = warm_gpt2_tokenizer_cache(args.cache_dir, timeout_s=args.timeout)
    print(f"Cache GPT-2 listo en {cache_path}")


if __name__ == "__main__":
    main()
