"""App-pinned compute downloads; never obtained from a network peer."""
from __future__ import annotations
from dataclasses import dataclass
import platform
import sys


@dataclass(frozen=True)
class EnginePart:
    url:str
    sha256:str
    bytes:int


@dataclass(frozen=True)
class EngineDownload:
    url: str
    sha256: str
    bytes: int
    manifest_sha256: str
    installed_bytes: int
    profile: str = 'reference-cpu'
    parts: tuple[EnginePart,...] = ()


BASE_URL = "https://github.com/toxxy/LixySwarm/releases/download/cpu-engine-0.4.0a1/"
CPU_ENGINES: dict[tuple[str, str], EngineDownload] = {
    ("linux", "x86_64"): EngineDownload(BASE_URL+"LixySwarmEngineCPU-ubuntu-22.04.zip",
        "afdb0e7fcbbdbaa9711acdb766e3ca88afcd47aea2eedf2d3f6316bfec89f5e4",266362786,
        "be3aeedc76ac88be20959582f31a038bdc6bf5d395fd2ab9d107eac15666dbcb",838694716),
    ("win32", "x86_64"): EngineDownload(BASE_URL+"LixySwarmEngineCPU-windows-2025.zip",
        "10f1afabfd99a95fb39911906a29d699199c586cd95f2730bd31c304905ad3ea",162385719,
        "6d52dfb14db5ffb88cdecb894276c0bc35477e345682457a68f2729cfe0d130d",477189675),
    ("darwin", "arm64"): EngineDownload(BASE_URL+"LixySwarmEngineCPU-macos-15.zip",
        "343f5bb96d5fa80f5fa0542c92cff676d2e78e5d2253341307484591e7dbac78",196265924,
        "5cb6904bb7a7add50bcef9ecb7add4ebe4f3fd777d2bd385c4167411311b866e",716376795),
}


def cpu_engine():
    machine = {"AMD64": "x86_64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    return CPU_ENGINES.get((sys.platform, machine))
