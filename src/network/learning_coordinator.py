"""Durable bounded learning coordinated by an existing physical LSP node."""
from __future__ import annotations
import json
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,dataclass,field
from pathlib import Path

from src.contribution.resource_governor import ResourceRequirements
from src.release import ReleaseManifest
from .lsp import LSPIdentity
from .reference_compute import PUBLIC_DATASET_ID,TRAINING_OPERATION,NODE_ID
from .telemetry import atomic_json
from .work_protocol import ResultReceipt,WorkResult


@dataclass
class LearningSettings:
    version:int=1
    enabled:bool=False
    cycles:int=1
    workers:list[str]=field(default_factory=list)
    signing_key_file:str=""

    def validate(self):
        if type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool or type(self.cycles) is not int or not 1<=self.cycles<=10:
            raise ValueError("Invalid bounded learning settings")
        if not isinstance(self.workers,list) or len(self.workers)>2 or len(set(self.workers))!=len(self.workers) or any(not isinstance(x,str) or not NODE_ID.fullmatch(x) for x in self.workers):
            raise ValueError("The pilot needs two distinct physical workers")
        if self.enabled and (len(self.workers)!=2 or not Path(self.signing_key_file).is_absolute()):
            raise ValueError("Learning needs workers and a locally authorized publisher")

    def save(self,directory):
        self.validate(); atomic_json(Path(directory)/"learning-controls.json",asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/"learning-controls.json"
        if not path.exists(): return cls()
        if path.stat().st_size>8192: raise ValueError("Learning settings exceed their bound")
        value=json.loads(path.read_text())
        if set(value)!={"version","enabled","cycles","workers","signing_key_file"}: raise ValueError("Invalid learning schema")
        result=cls(**value); result.validate(); return result


class LearningCoordinator:
    def __init__(self,knowledge,compute,directory):
        self.knowledge,self.compute,self.directory=knowledge,compute,Path(directory)
        self.path=self.directory/"learning"/"ledger.json"
        self.lock=threading.RLock(); self.changed=threading.Event()
        self.settings=LearningSettings(); self.thread=None; self.cancel=threading.Event()
        self.closed=False; self.next_attempt=0
        self.ledger=self._load()

    def _load(self):
        if not self.path.exists():
            cursor=0
            active=self.knowledge.registry.active()
            if active is not None and active.config_artifact_id:
                try:
                    path=self.knowledge.store.object_path(active.config_artifact_id)
                    if path.stat().st_size>64*1024: raise ValueError("Evaluation report exceeds its bound")
                    report=json.loads(path.read_text())["learning"]
                    if report["dataset_id"]==PUBLIC_DATASET_ID:
                        cursor=min(item["start_token"] for item in report["contributions"])+65
                except (OSError,ValueError,KeyError,TypeError):
                    cursor=0
            return {"status":"idle","round":0,"cursor":cursor,"contributions":[]}
        if self.path.stat().st_size>64*1024: raise ValueError("Learning ledger exceeds its bound")
        return json.loads(self.path.read_text())

    def _save(self,**values):
        with self.lock:
            self.ledger.update(values); atomic_json(self.path,self.ledger); self.changed.set()

    def snapshot(self):
        with self.lock:
            return {**{key:self.ledger.get(key) for key in ("status","round","cursor","accepted_sequence","error")},
                    "paused":self.cancel.is_set() and self.ledger.get("status") in {"collecting","evaluating","promoting"}}

    def _training_requirements(self,peer):
        requester=self.knowledge.node.identity.node_id_hex
        if peer==requester:
            gpu=("cuda" in self.compute.engine.devices and self.compute.settings.policy.gpu_percent>0
                 and self.knowledge.resources["has_gpu"])
        else:
            resources=next((item.get("resources",{}) for item in self.knowledge.node.peers()
                            if item["node_id"]==peer),{})
            gpu=resources.get("has_gpu") is True
        return ResourceRequirements(kind="training",gpu_required=gpu,ram_gb=2,disk_gb=0.01)

    def update(self):
        try: self.settings=LearningSettings.load(self.directory)
        except (OSError,ValueError,TypeError): self.settings=LearningSettings()
        allowed=self.settings.enabled and self.compute.ready and not self.closed
        if not allowed:
            self.cancel.set(); return
        if self.thread is not None and self.thread.is_alive(): return
        if self.ledger.get("round",0)>=self.settings.cycles:
            if self.ledger.get("status")!="complete": self._save(status="complete")
            return
        if time.monotonic()<self.next_attempt: return
        self.cancel=threading.Event()
        self.thread=threading.Thread(target=self._run,args=(self.cancel,),daemon=True,name="lixy-learning")
        self.thread.start()

    def _authorized(self,cancel,configuration):
        return not cancel.is_set() and self.compute.ready and asdict(self.settings)==configuration and not self.closed

    def _recover_promotion(self):
        if self.ledger.get("status")!="promoting": return
        release=ReleaseManifest.from_dict(self.ledger["release"])
        self.knowledge.catalog.policy.authorize(release)
        with self.knowledge.release_lock:
            self.knowledge.registry.accept(release,self.knowledge.catalog.policy,self.knowledge.store)
            active=self.knowledge.registry.active()
            if active.release_id!=release.release_id:
                self.knowledge.registry.activate(release.release_id,self.knowledge.catalog.policy,self.knowledge.store)
        self.knowledge.promoted(release)
        self._finish_round(release.sequence)

    def _finish_round(self,sequence):
        self._save(status="idle",round=self.ledger["round"]+1,cursor=self.ledger["cursor"]+65,
                   contributions=[],accepted_sequence=sequence,release=None,error=None)

    def _run(self,cancel):
        configuration=asdict(self.settings)
        try:
            self._recover_promotion()
            if self.ledger["round"]>=self.settings.cycles: self._save(status="complete"); return
            signer=LSPIdentity.load(self.settings.signing_key_file)
            if signer is None or signer.node_id_hex not in self.knowledge.catalog.policy.trusted_signers:
                raise ValueError("Publisher key is not authorized by this catalog")
            active=self.knowledge.registry.verify_active(self.knowledge.catalog.policy,self.knowledge.store)
            base=self.ledger.get("base_release")
            if base is not None and base["release_id"]!=active.release_id:
                # Stale partial work never crosses a checkpoint boundary.
                self._save(contributions=[])
            self._save(status="collecting",base_release=active.to_dict(),error=None)
            if not self.knowledge.store.has(PUBLIC_DATASET_ID):
                raise ValueError("The pinned public corpus must be prepared locally")
            requester=self.knowledge.node.identity.node_id_hex
            if requester not in self.settings.workers:
                raise ValueError("The coordinator must be one of these physical workers")
            saved={item["peer_id"]:item for item in self.ledger["contributions"]}
            def collect(pair):
                position,peer=pair
                if peer in saved:
                    previous=saved[peer]; result=WorkResult.from_dict(previous["result"])
                    if self.knowledge.store.has(previous["gradient_id"]) and ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=peer,expected_requester_id=requester):
                        return previous
                if not self._authorized(cancel,configuration): raise RuntimeError("Learning permission withdrawn")
                start=position*7200+self.ledger["cursor"]%7135
                payload={"model_artifact_id":active.model_artifact_id,"dataset_artifact_id":PUBLIC_DATASET_ID,"start_token":start,"token_count":65}
                if peer==requester:
                    requirements=self._training_requirements(peer)
                    result=self.knowledge.work.execute_local(TRAINING_OPERATION,payload,requirements,timeout_s=60,cancel_event=cancel)
                else:
                    if peer not in self.knowledge.node.connected_node_ids: self.knowledge.node.connect_node(peer)
                    requirements=self._training_requirements(peer)
                    result=self.knowledge.work.submit(TRAINING_OPERATION,payload,requirements,peer_id=peer,timeout_s=60,cancel_event=cancel)
                if result.status!="ok": raise RuntimeError(f"Physical shard failed: {result.error}")
                if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=peer,expected_requester_id=requester):
                    raise ValueError("Shard receipt does not match the physical assignment")
                gradient=result.output["gradient"]["artifact_id"]
                if peer!=requester: self.knowledge.artifacts.fetch(gradient,peer_id=peer,timeout_s=60,cancel_event=cancel)
                value={"peer_id":peer,"start_token":start,"token_count":65,"gradient_id":gradient,"result":result.to_dict()}
                with self.lock:
                    items=[item for item in self.ledger["contributions"] if item["peer_id"]!=peer]+[value]
                    self._save(contributions=items)
                return value
            with ThreadPoolExecutor(max_workers=2) as pool:
                contributions=list(pool.map(collect,enumerate(self.settings.workers)))
            if not self._authorized(cancel,configuration): return
            self._save(status="evaluating")
            policy=self.compute.settings.policy
            lease,reason=self.knowledge.governor.acquire(ResourceRequirements(kind="training",ram_gb=2,disk_gb=0.02))
            if lease is None: raise RuntimeError(f"Candidate evaluation denied: {reason}")
            job=self.directory/"learning"/"candidate-job"; job.mkdir(parents=True,exist_ok=True,mode=0o700)
            class Execution:
                deadline=time.time()+60
                def raise_if_cancelled(inner):
                    if not self._authorized(cancel,configuration): raise RuntimeError("Learning cancelled")
            request={"action":"candidate","device":"cpu","cpu_threads":max(1,min(8,int(self.knowledge.resources["cpu_cores"]*policy.cpu_percent/100))),
                "gpu_fraction":0,"deadline":Execution.deadline,"output_directory":str(job),
                "public_directory":str(self.directory/"public"),"store_quota":self.knowledge.store.max_total_bytes,
                "trust_policy":asdict(self.knowledge.catalog.policy),"base_release":active.to_dict(),
                "requester_node_id":requester,"trusted_workers":self.settings.workers,
                "dataset_id":PUBLIC_DATASET_ID,"contributions":contributions}
            try:
                with lease:
                    evaluated=self.compute.supervisor.run(self.compute.engine.command_for("cpu"),request,Execution(),
                        allowed=lambda:self._authorized(cancel,configuration),ram_gb=2)
            finally:
                shutil.rmtree(job)
            self._save(evaluation=evaluated["report"])
            if evaluated["accepted"]:
                report=evaluated["report"]
                before,after=report["before_loss"],report["after_loss"]
                if not isinstance(before,(int,float)) or not isinstance(after,(int,float)) or not math.isfinite(before) or not math.isfinite(after) or before-after<0.001:
                    raise ValueError("Candidate does not meet the fixed evaluation improvement")
                if self.knowledge.registry.active().release_id!=active.release_id:
                    self._save(status="idle",contributions=[],base_release=None)
                    return
                release=ReleaseManifest.from_dict(evaluated["release"]).sign(signer)
                if release.previous_release_id!=active.release_id or release.sequence!=active.sequence+1:
                    raise ValueError("Candidate does not extend the assigned checkpoint")
                self._save(status="promoting",release=release.to_dict())
                self._recover_promotion()
            else:
                self._finish_round(active.sequence)
        except (OSError,ValueError,RuntimeError) as error:
            self._save(error=str(error)[:200])
        finally:
            self.next_attempt=time.monotonic()+5

    def close(self):
        self.closed=True; self.cancel.set()
        if self.thread is not None: self.thread.join(timeout=10)
