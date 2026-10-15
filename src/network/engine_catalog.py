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
    regular_file_bytes: int | None = None


def engine_regular_bytes(entry):
    """Conservative legacy bound, or an explicit app-pinned regular-file size."""
    if type(entry.installed_bytes) is not int or entry.installed_bytes<=0:
        raise ValueError('Invalid locally pinned logical engine size')
    size = getattr(entry,'regular_file_bytes',None)
    if size is None:return entry.installed_bytes
    if type(size) is not int or not 0 < size <= entry.installed_bytes:
        raise ValueError('Invalid locally pinned regular engine size')
    return size


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


LANGUAGE_URL='https://github.com/toxxy/LixySwarm/releases/download/language-engine-0.4.0a1/'
EAGER_LANGUAGE_URL='https://github.com/toxxy/LixySwarm/releases/download/language-engine-0.4.0a19/'
LANGUAGE_ENGINES={
    'cpu':EngineDownload(LANGUAGE_URL+'LixySwarmLanguageCPU-linux-x86_64.zip',
        'c075af9813e8ea8124c0dfae38b7ad30014cdad3d7b7749af3deec5e0cfa5f3b',286019554,
        '231e0866dd651fe38eb211d248295d4321ff25b5fd0ca7bc42048ae03b1723b6',879096011,profile='language-cpu'),
    'cuda':EngineDownload(EAGER_LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip',
        '10a46a823e0b9bf396493671123645ba047d9c6229f72dc075d99e8d8a9b5a2d',2524941592,
        '6270e1ee65f1886aa27db4a06c63961461e06a0c3a7784e2ef1280e369a7b247',6004344627,profile='language-cuda-math',parts=(
            EnginePart(EAGER_LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part1','04cc14e887fd3979d8327a0cd873ce77cd824e5b67981f38a616900e2f8e678f',1073741824),
            EnginePart(EAGER_LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part2','7f078fdf9c17e1f181fe656d4d4fc53803a3ec38d4a50b64093723edc0eb5298',1073741824),
            EnginePart(EAGER_LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part3','71180781da1dbafcd49bdab8af9b0636fc4fd62efac93b2dc7d7afac5f23de01',377457944)),
        regular_file_bytes=4038463047),
}


def language_engine(device):
    # These first binaries were built and checked on Ubuntu 24.04.
    if sys.platform!='linux' or platform.machine()!='x86_64':return None
    try:
        if tuple(int(x) for x in platform.libc_ver()[1].split('.')[:2])<(2,39):return None
    except ValueError:return None
    return LANGUAGE_ENGINES.get(device)
