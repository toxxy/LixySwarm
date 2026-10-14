"""Optional public language inference under the physical node's permissions.

Only local configuration can select a command, base or adapter. Peers supply
bounded conversation input, never code, paths, model checkpoints or policies.
This service neither trains from conversations nor publishes their contents.
"""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass,asdict,field
from pathlib import Path

from .compute_process import ComputeSupervisor,ComputeProcessError
from .telemetry import atomic_json
from src.learning.public_bases import BASES

LANGUAGE_OPERATION="language.inference.v1"


def validate_messages(messages):
    if not isinstance(messages,list) or not 1<=len(messages)<=8:raise ValueError("Invalid bounded language messages")
    size=0
    for row in messages:
        if not isinstance(row,dict) or set(row)!={"role","content"} or row["role"] not in {"user","assistant"} or not isinstance(row["content"],str) or not row["content"]:
            raise ValueError("Only explicit user/assistant conversation is accepted")
        size+=len(row["content"].encode())
    if size>4096 or messages[-1]["role"]!="user":raise ValueError("Language input exceeds its bound or has no current question")


@dataclass
class LanguageSettings:
    version:int=1
    enabled:bool=False
    command:list[str]=field(default_factory=list)
    base_name:str="qwen3-1.7b"
    base_directory:str=""
    adapter_file:str=""
    adapter_id:str=""

    def validate(self):
        if type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool or self.base_name not in BASES:raise ValueError("Invalid known language engine")
        if not isinstance(self.command,list) or len(self.command)>4 or any(not isinstance(x,str) or not x or len(x)>4096 for x in self.command):raise ValueError("Invalid local language command")
        if self.enabled:
            if not self.command or any(not Path(x).is_absolute() for x in [self.command[0],self.base_directory,self.adapter_file]):raise ValueError("Language paths must be installed locally")
            if len(self.adapter_id)!=64 or any(x not in "0123456789abcdef" for x in self.adapter_id):raise ValueError("Invalid language adapter id")

    def save(self,directory):
        self.validate();atomic_json(Path(directory)/"language-engine.json",asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/"language-engine.json"
        if not path.exists():return cls()
        if path.stat().st_size>20*1024:raise ValueError("Language settings exceed their bound")
        value=json.loads(path.read_text())
        if set(value)!=set(asdict(cls())):raise ValueError("Invalid language settings schema")
        result=cls(**value);result.validate();return result


class LanguageComputeService:
    def __init__(self,knowledge,directory,resources):
        self.knowledge,self.directory,self.resources=knowledge,Path(directory),resources
        self.engine=LanguageSettings();self.settings=None;self.ready=False;self.active=0;self.completed=0
        self.lock=threading.RLock();self.changed=threading.Event();self.supervisor=ComputeSupervisor()
        path=self.directory/"language-status.json"
        if path.exists():
            try:
                if path.stat().st_size>1024:raise ValueError("Language counter exceeds its bound")
                value=json.loads(path.read_text())["completed"]
                if type(value) is not int or not 0<=value<=10**12:raise ValueError("Invalid language counter")
                self.completed=value
            except (OSError,ValueError,TypeError,KeyError):pass
        knowledge.work.register_handler(LANGUAGE_OPERATION,"inference",self.inference,
            permit_cached=lambda execution,result:self.ready and result.get("model_id") in self.approved_models())

    def _ancestors(self):
        """Only predecessors of the locally selected adaptation may answer."""
        from src.release.language_versions import LanguageVersionRegistry
        from .artifact_store import ArtifactStore
        try:
            registry=LanguageVersionRegistry(self.directory/'public/language',self.engine.base_name)
            chain=registry.history(registry.head()['release']['release_id'])
            selected=max(i for i,row in enumerate(chain) if row['release']['model_artifact_id']==self.engine.adapter_id)
            store=ArtifactStore(self.directory/'public/language/artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
            return {row['release']['model_artifact_id']:row['release']['release_id'] for row in chain[:selected]
                if store.object_path(row['release']['model_artifact_id']).is_file()}
        except (OSError,ValueError,TypeError,KeyError,StopIteration):return {}

    def approved_models(self):
        if not self.ready:return []
        return list(dict.fromkeys([self.engine.adapter_id,*self._ancestors()]))

    def update(self,settings):
        self.settings=settings
        try:self.engine=LanguageSettings.load(self.directory)
        except (OSError,ValueError,TypeError):self.engine=LanguageSettings()
        self.ready=bool(self.engine.enabled and settings.connected and not settings.paused and settings.policy.mode in {"balanced","maximum"}
            and settings.policy.cpu_percent>0 and settings.policy.max_cpu_jobs>0 and settings.policy.max_ram_gb>=self.required_ram()
            and settings.policy.gpu_percent>0 and settings.policy.max_gpu_jobs>0 and self.resources["has_gpu"]
            and self.resources["gpu_vram_gb"]*settings.policy.gpu_percent/100/max(1,settings.policy.max_gpu_jobs)>= (4 if self.engine.base_name=="qwen3-1.7b" else 2)
            and all(Path(arg).is_file() for arg in self.engine.command)
            and Path(self.engine.base_directory).is_dir() and Path(self.engine.adapter_file).is_file())

    def required_ram(self):return 5 if self.engine.base_name=="qwen3-1.7b" else 4

    def snapshot(self):
        return {"installed":self.engine.enabled,"ready":self.ready,"model_id":self.engine.adapter_id if self.engine.enabled else "",
                "base":self.engine.base_name,"active":self.active,"completed":self.completed}

    def inference(self,payload,execution):
        if set(payload)!={"model_artifact_id","messages"}:raise ValueError("Invalid language operation schema")
        validate_messages(payload["messages"])
        model_id=payload['model_artifact_id'];ancestors=self._ancestors() if model_id!=self.engine.adapter_id else {}
        if model_id!=self.engine.adapter_id and model_id not in ancestors:raise ValueError("Only the exact locally approved adapter or its accepted ancestors can answer")
        if not self.ready or not execution.resource_requirements().gpu_required or execution.resource_requirements().ram_gb<self.required_ram():raise ComputeProcessError("Language inference requires a permitted GPU and sufficient RAM lease")
        permission=asdict(self.settings.policy);configuration=asdict(self.engine)
        def allowed():
            execution.raise_if_cancelled()
            return self.ready and asdict(self.settings.policy)==permission and asdict(self.engine)==configuration
        base_directory=self.engine.base_directory;adapter_file=self.engine.adapter_file
        if model_id!=self.engine.adapter_id:
            from .language_install import materialize_accepted_version
            base=materialize_accepted_version(self.directory,self.engine.base_name,ancestors[model_id],allowed=allowed)
            base_directory=str(base);adapter_file=str(base/'adapter.safetensors')
        request={"action":"inference","device":"cuda","base_name":self.engine.base_name,"base_directory":base_directory,
                 "adapter_file":adapter_file,"adapter_id":model_id,"messages":payload["messages"],
                 "deadline":execution.deadline,"cpu_threads":max(1,min(8,int(self.resources["cpu_cores"]*self.settings.policy.cpu_percent/100))),
                 "gpu_fraction":self.settings.policy.gpu_percent/100/max(1,self.settings.policy.max_gpu_jobs)}
        with self.lock:self.active+=1
        self.changed.set()
        try:
            result=self.supervisor.run(self.engine.command,request,execution,
                allowed=allowed,
                ram_gb=execution.resource_requirements().ram_gb)
            if set(result)!={"model_id","answer","output_tokens"} or result["model_id"]!=model_id or not isinstance(result["answer"],str) or not result["answer"] or len(result["answer"].encode())>8192 or type(result["output_tokens"]) is not int or not 1<=result["output_tokens"]<=128:raise ValueError("Invalid language result")
            with self.lock:
                self.completed+=1;atomic_json(self.directory/"language-status.json",{"completed":self.completed})
            return result
        finally:
            with self.lock:self.active-=1
            self.changed.set()
