import hashlib
import io
import json
import platform
import sys
import threading
import subprocess
import zipfile
from dataclasses import replace

import pytest

from src.network.engine_catalog import EngineDownload
from src.network.engine_download import PreparationCancelled, prepare_engine
from src.network.reference_compute import EngineSettings


def archive(extra=None):
    name="LixySwarmEngineCPU.exe" if sys.platform=="win32" else "LixySwarmEngineCPU"
    binary=b"test-engine"
    metadata={"version":1,"name":"LixySwarmEngineCPU","platform":sys.platform,"machine":platform.machine(),
              "torch":"2.12.0+cpu","operations":["reference.inference.v1","training.gradient.v1","reference.candidate.v1"],
              "entrypoint":name,"files":[{"path":name,"bytes":len(binary),"sha256":hashlib.sha256(binary).hexdigest()}]}
    manifest=json.dumps(metadata).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as zipped:
        zipped.writestr(name,binary);zipped.writestr("engine-package.json",manifest)
        if extra:zipped.writestr(*extra)
    data=buffer.getvalue()
    entry=EngineDownload("https://example.invalid/engine.zip",hashlib.sha256(data).hexdigest(),len(data),
                         hashlib.sha256(manifest).hexdigest(),len(binary)+len(manifest)+(len(extra[1]) if extra else 0))
    return entry,data


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {}


def test_download_and_archive_are_verified_without_executing_package_code(tmp_path):
    entry,data=archive();progress=[]
    command=prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,
                           progress=lambda *values:progress.append(values),opener=lambda *_args,**_kwargs:Response(data))
    assert EngineSettings.load(tmp_path).command_for("cpu")==[command]
    assert "complete" in [row[0] for row in progress]
    subprocess.run([sys.executable,"-c","import sys; import src.network.engine_download; assert 'torch' not in sys.modules"],check=True)


def test_cancelled_download_resumes_exact_bytes_and_does_not_enable_engine(tmp_path):
    entry,data=archive();part=tmp_path/"engine-downloads"/(entry.sha256+".part")
    part.parent.mkdir();part.write_bytes(data[:50]);cancel=threading.Event();cancel.set()
    with pytest.raises(PreparationCancelled):
        prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,cancel_event=cancel)
    assert not EngineSettings.load(tmp_path).enabled and part.stat().st_size==50
    def resume(request,**_):
        assert request.headers["Range"]=="bytes=50-"
        return Response(data[50:],206,{"Content-Range":f"bytes 50-{len(data)-1}/{len(data)}"})
    prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,opener=resume)
    assert EngineSettings.load(tmp_path).enabled


def test_unexpected_range_corruption_and_size_are_not_installed(tmp_path):
    entry,data=archive();root=tmp_path/"engine-downloads";root.mkdir()
    (root/(entry.sha256+".part")).write_bytes(data[:50])
    with pytest.raises(ValueError,match="range"):
        prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,
                       opener=lambda *_a,**_k:Response(data[50:],206,{"Content-Range":"bytes 1-2/3"}))
    (root/(entry.sha256+".part")).unlink()
    with pytest.raises(ValueError,match="hash"):
        prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,
                       opener=lambda *_a,**_k:Response(b"x"*len(data)))
    assert not EngineSettings.load(tmp_path).enabled


@pytest.mark.parametrize("path",["../escape","/escape","a\\escape","engine-package.json"])
def test_even_a_pinned_archive_cannot_escape_or_overwrite_entries(tmp_path,path):
    entry,data=archive((path,b"malformed"))
    with pytest.raises(ValueError):
        prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,
                       opener=lambda *_a,**_k:Response(data))
    assert not EngineSettings.load(tmp_path).enabled


def test_installing_cpu_preserves_the_locally_installed_cuda_engine(tmp_path):
    entry,data=archive()
    EngineSettings(enabled=True,command=[sys.executable],devices=["cpu","cuda"],training_requesters=["a"*64]).save(tmp_path)
    cpu=prepare_engine(entry,tmp_path,allowed=lambda:True,rate_mbps=lambda:1000000,
                       opener=lambda *_a,**_k:Response(data))
    installed=EngineSettings.load(tmp_path)
    assert installed.command_for("cuda")==[sys.executable]
    assert installed.command_for("cpu")==[cpu] and installed.training_requesters==["a"*64]
