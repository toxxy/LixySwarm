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
LANGUAGE_ENGINES={
    'cpu':EngineDownload(LANGUAGE_URL+'LixySwarmLanguageCPU-linux-x86_64.zip',
        'c075af9813e8ea8124c0dfae38b7ad30014cdad3d7b7749af3deec5e0cfa5f3b',286019554,
        '231e0866dd651fe38eb211d248295d4321ff25b5fd0ca7bc42048ae03b1723b6',879096011,profile='language-cpu'),
    'cuda':EngineDownload(LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip',
        '32159cba0106fd124c3bdb07361d0cf91d4188b07ba30f202f20f54950674e74',2725324662,
        'f93d4310d0178cfb7f61ac829da507adff992b01e29b91ced2e71bc1c7159bb8',7200954897,profile='language-cuda',parts=(
            EnginePart(LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part1','6129e35d465fa761c46b9f592e8ea4ebd47a6e2ebc2a7f02afee9a4246fa9a08',1073741824),
            EnginePart(LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part2','2290e7c14d76c7ca2bea14bd91d9f5ba1aaabba0047404755ac13bde12d7ab14',1073741824),
            EnginePart(LANGUAGE_URL+'LixySwarmLanguageCUDA-linux-x86_64.zip.part3','17852d3f75a0d071897e72a4da1afdb998d123708fe0a704450beda867ceb367',577841014))),
}


def language_engine(device):
    # These first binaries were built and checked on Ubuntu 24.04.
    if sys.platform!='linux' or platform.machine()!='x86_64':return None
    try:
        if tuple(int(x) for x in platform.libc_ver()[1].split('.')[:2])<(2,39):return None
    except ValueError:return None
    return LANGUAGE_ENGINES.get(device)
