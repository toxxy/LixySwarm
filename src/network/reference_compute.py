"""Typed reference work dispatched by the physical node to a local engine."""
from __future__ import annotations
import json
import math
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .compute_process import ComputeProcessError, ComputeSupervisor
from .telemetry import atomic_json

TRAINING_OPERATION="training.gradient.v1"
INFERENCE_OPERATION="reference.inference.v1"
PUBLIC_DATASET_ID="df3c694642f2eacef9e0734e1e958a14d1052d3d6927457e311e87a90f93c97b"
NODE_ID=re.compile(r"^[0-9a-f]{64}$")


@dataclass
class EngineSettings:
    version:int=1
    enabled:bool=False
    command:list[str]=field(default_factory=list)
    training_requesters:list[str]=field(default_factory=list)
    devices:list[str]=field(default_factory=lambda:["cpu","cuda"])
    cpu_command:list[str]=field(default_factory=list)

    def validate(self):
        if type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool:
            raise ValueError("Invalid local engine settings")
        if not isinstance(self.command,list) or not 0<=len(self.command)<=4 or any(not isinstance(x,str) or not x or len(x)>4096 for x in self.command):
            raise ValueError("Invalid locally installed engine command")
        if self.enabled and (not self.command or not Path(self.command[0]).is_absolute()):
            raise ValueError("Compute requires a locally installed engine")
        if not isinstance(self.training_requesters,list) or len(self.training_requesters)>16 or any(not NODE_ID.fullmatch(x) for x in self.training_requesters):
            raise ValueError("Invalid known pilot requesters")
        if not isinstance(self.devices,list) or not 1<=len(self.devices)<=2 or len(set(self.devices))!=len(self.devices) or any(x not in {"cpu","cuda"} for x in self.devices):
            raise ValueError("Invalid locally installed engine devices")
        if not isinstance(self.cpu_command,list) or len(self.cpu_command)>4 or any(not isinstance(x,str) or not x or len(x)>4096 for x in self.cpu_command) or (self.cpu_command and not Path(self.cpu_command[0]).is_absolute()):
            raise ValueError("Invalid locally installed CPU engine command")

    def command_for(self,device):
        if device=="cpu" and self.cpu_command: return self.cpu_command
        return self.command if device in self.devices else []

    def save(self,directory):
        self.validate(); atomic_json(Path(directory)/"compute-engine.json",asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/"compute-engine.json"
        if not path.exists(): return cls()
        if path.stat().st_size>20*1024: raise ValueError("Engine settings exceed their bound")
        value=json.loads(path.read_text())
        base={"version","enabled","command","training_requesters"}
        if not base<=set(value) or set(value)-base-{"devices","cpu_command"}:
            raise ValueError("Invalid engine settings schema")
        result=cls(**value); result.validate(); return result


class ReferenceComputeService:
    def __init__(self,knowledge,directory,resources):
        self.knowledge,self.directory,self.resources=knowledge,Path(directory),resources
        self.settings=None; self.engine=EngineSettings(); self.ready=False
        self.supervisor=ComputeSupervisor(); self.lock=threading.RLock()
        self.changed=threading.Event()
        self.active=0; self.completed=0
        path=self.directory/"compute-status.json"
        if path.exists():
            try:
                if path.stat().st_size>1024: raise ValueError("Compute counter exceeds its bound")
                value=json.loads(path.read_text()); count=value.get("completed",0)
                if type(count) is not int or not 0<=count<=10**12: raise ValueError("Invalid compute counter")
                self.completed=count
            except (OSError,ValueError,TypeError):
                self.completed=0
        knowledge.work.register_handler(INFERENCE_OPERATION,"inference",self.inference,permit_cached=lambda execution,_:self.ready)
        knowledge.work.register_handler(TRAINING_OPERATION,"training",self.training,
            permit_cached=lambda execution,_:self.ready and execution.origin_node_id in self.engine.training_requesters)

    def update(self,settings):
        self.settings=settings
        try: self.engine=EngineSettings.load(self.directory)
        except (OSError,ValueError,TypeError): self.engine=EngineSettings()
        self.ready=bool(self.engine.enabled and settings.connected and not settings.paused
            and settings.policy.mode in {"balanced","maximum"} and settings.policy.cpu_percent>0
            and settings.policy.max_cpu_jobs>0 and settings.policy.max_ram_gb>=1
            and self.knowledge.complete and self._installed())

    def _installed(self):
        command=self.engine.command_for("cpu")
        return bool(command) and all(Path(arg).is_file() for arg in command)

    def snapshot(self):
        with self.lock:
            installed=self.engine.enabled and self._installed()
            return {"installed":installed,"ready":self.ready,"active":self.active,"completed":self.completed,
                    "devices":list(self.engine.devices) if installed else []}

    def inference(self,payload,execution):
        if set(payload)!={"model_artifact_id","prompt"} or not isinstance(payload["prompt"],str) or not 1<=len(payload["prompt"].encode())<=256:
            raise ValueError("Invalid reference inference request")
        return self._run("inference",payload,execution)

    def training(self,payload,execution):
        if execution.origin_node_id not in self.engine.training_requesters:
            raise ValueError("Productive training is restricted to the known pilot")
        if set(payload)!={"model_artifact_id","dataset_artifact_id","start_token","token_count"} or payload["dataset_artifact_id"]!=PUBLIC_DATASET_ID:
            raise ValueError("Only the pinned public reference corpus is accepted")
        if type(payload["start_token"]) is not int or payload["start_token"]<0 or type(payload["token_count"]) is not int or not 2<=payload["token_count"]<=65:
            raise ValueError("Invalid bounded reference token range")
        if not self.knowledge.store.has(PUBLIC_DATASET_ID):
            self.knowledge.artifacts.fetch(PUBLIC_DATASET_ID,peer_id=execution.origin_node_id,
                timeout_s=min(60,execution.remaining_s()),cancel_event=execution.cancel_event)
        return self._run("training",payload,execution)

    def _run(self,action,payload,execution):
        execution.raise_if_cancelled()
        if not self.ready or execution.resource_requirements().ram_gb<1:
            raise ComputeProcessError("Compute is unavailable or its RAM lease is too small")
        release=self.knowledge.registry.verify_active(self.knowledge.catalog.policy,self.knowledge.store)
        if payload["model_artifact_id"]!=release.model_artifact_id:
            raise ValueError("Compute requires the exact active checkpoint")
        policy=self.settings.policy
        permission=asdict(policy); engine_permission=asdict(self.engine)
        gpu=execution.resource_requirements().gpu_required
        command=self.engine.command_for("cuda" if gpu else "cpu")
        if not command or not all(Path(arg).is_file() for arg in command):
            raise ComputeProcessError("The locally installed engine does not support this device")
        if gpu and (not self.resources["has_gpu"] or policy.gpu_percent<=0):
            raise ComputeProcessError("GPU work is not available within current permission")
        if gpu and action=="training" and execution.resource_requirements().ram_gb<2:
            raise ComputeProcessError("The CUDA gradient runtime requires a two-GiB RAM lease")
        job_directory=self.directory/"compute"/uuid.uuid4().hex
        job_directory.mkdir(parents=True,mode=0o700)
        request={"action":action,"payload":payload,"device":"cuda" if gpu else "cpu",
            "model_path":str(self.knowledge.store.object_path(release.model_artifact_id)),
            "output_directory":str(job_directory),"deadline":execution.deadline,
            "cpu_threads":max(1,min(8,int(self.resources["cpu_cores"]*policy.cpu_percent/100))),
            "gpu_fraction":policy.gpu_percent/100/max(1,policy.max_gpu_jobs)}
        if action=="training": request["dataset_path"]=str(self.knowledge.store.object_path(PUBLIC_DATASET_ID))
        with self.lock: self.active+=1
        self.changed.set()
        try:
            result=self.supervisor.run(command,request,execution,
                allowed=lambda:self.ready and asdict(self.settings.policy)==permission and asdict(self.engine)==engine_permission,
                                      ram_gb=execution.resource_requirements().ram_gb)
            if action=="training":
                path=Path(result.pop("gradient_path"))
                if not path.resolve().is_relative_to(job_directory.resolve()):
                    raise ValueError("Engine result escaped its job directory")
                artifact=self.knowledge.store.import_file(path,kind="gradient",media_type="application/x-npz")
                if artifact.artifact_id!=result["gradient"]["artifact_id"]:
                    raise ValueError("Engine gradient content does not match")
                result["gradient"]=artifact.to_dict()
            else:
                if set(result)!={"model_id","next_byte_logits"} or result["model_id"]!=release.model_artifact_id or len(result["next_byte_logits"])!=256 or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in result["next_byte_logits"]):
                    raise ValueError("Invalid engine inference result")
            with self.lock:
                self.completed+=1
                atomic_json(self.directory/"compute-status.json",{"completed":self.completed})
                self.changed.set()
            return result
        finally:
            with self.lock: self.active-=1
            self.changed.set()
            shutil.rmtree(job_directory)
