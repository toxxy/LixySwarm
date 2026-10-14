"""Supervise local compute engines without importing their tensor libraries."""
from __future__ import annotations
import json
import os
import subprocess
import threading
import time

import psutil


class ComputeProcessError(RuntimeError):
    pass


def terminate_owned_process(process):
    if process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    process.kill()
    process.wait(timeout=5)


class ComputeSupervisor:
    def run(self, command, request, execution, *, allowed, ram_gb):
        encoded = json.dumps(request,allow_nan=False).encode()
        if len(encoded)>16*1024 or ram_gb<=0:
            raise ComputeProcessError("Invalid local compute bounds")
        environment = dict(os.environ); environment.pop("PYTHONPATH",None)
        if request.get("device")=="cpu":
            environment["CUDA_VISIBLE_DEVICES"]=""
        kwargs = {"start_new_session":True} if os.name!="nt" else {"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   env=environment,**kwargs)
        captured, excessive = {}, threading.Event()
        def drain(name, pipe, bound):
            blocks=[]; size=0
            while block:=pipe.read(4096):
                size+=len(block)
                if size<=bound:
                    blocks.append(block)
                else:
                    excessive.set()
            captured[name]=b"".join(blocks)
        readers=[threading.Thread(target=drain,args=("output",process.stdout,256*1024),daemon=True),
                 threading.Thread(target=drain,args=("error",process.stderr,64*1024),daemon=True)]
        for reader in readers: reader.start()
        try:
            process.stdin.write(encoded); process.stdin.close()
            try:
                measured = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                measured = None
            while process.poll() is None:
                execution.raise_if_cancelled()
                if not allowed() or time.time()>=execution.deadline:
                    raise ComputeProcessError("Compute permission withdrawn or deadline exceeded")
                if excessive.is_set():
                    raise ComputeProcessError("Compute engine exceeded its output bound")
                try:
                    if measured is not None and measured.memory_info().rss > ram_gb*1024**3:
                        raise ComputeProcessError("Compute engine exceeded its RAM lease")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.05)
            for reader in readers: reader.join(timeout=2)
            execution.raise_if_cancelled()
            if not allowed() or excessive.is_set() or process.returncode != 0:
                raise ComputeProcessError("Compute engine did not complete an authorized result")
            value=json.loads(captured.get("output",b""))
            if not isinstance(value,dict):
                raise ComputeProcessError("Compute result must be an object")
            return value
        finally:
            terminate_owned_process(process)
            for reader in readers: reader.join(timeout=2)
            for pipe in (process.stdin,process.stdout,process.stderr):
                if pipe is not None: pipe.close()
