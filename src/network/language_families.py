"""Public language packages share the existing physical node and work identity."""
import json
import threading
from pathlib import Path

from src.contribution import ResourceRequirements
from src.release.language_family import BASES,load_bundle,dependencies,verify_bundle
from .artifact_store import ArtifactStore,ArtifactService,ArtifactError
from .telemetry import atomic_json


class LanguageFamilies:
    def __init__(self,knowledge,directory):
        self.knowledge=knowledge;self.root=Path(directory)/"public/language"
        self.store=ArtifactStore(self.root/"artifacts",max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
        self.artifacts=ArtifactService(knowledge.work,self.store,namespace="language.artifact",receive_budget=knowledge.budget)
        for operation,handler in [("language.artifact.describe.v1",self.artifacts._describe),
                                  ("language.artifact.read-chunk.v1",self._chunk),
                                  ("language.family.v1",self._family)]:
            knowledge.work.register_handler(operation,"artifact",self._guard(handler),permit_cached=self._permit_cached)

    def _guard(self,handler):
        def guarded(payload,execution):
            execution.raise_if_cancelled()
            if not self.knowledge.allowed:raise ArtifactError("Public family sharing is paused or disabled")
            return handler(payload,execution)
        return guarded

    def _chunk(self,payload,execution):
        value=self.artifacts._read_chunk(payload,execution)
        self.knowledge.budget.wait(len(value["data"])+2048,cancel_event=self.knowledge.cancel,deadline=execution.deadline)
        return value

    def _permit_cached(self,execution,output):
        if not self.knowledge.allowed:return False
        if execution.operation=="language.artifact.read-chunk.v1":
            self.knowledge.budget.wait(len(output["data"])+2048,cancel_event=self.knowledge.cancel,deadline=execution.deadline)
        return self.knowledge.allowed

    def _family(self,payload,execution):
        if not isinstance(payload,dict) or set(payload)!={"base_name"}:raise ArtifactError("Invalid language-family request")
        bundle=load_bundle(self.root,payload["base_name"])
        for identifier,size in dependencies(bundle):
            manifest=self.store.manifest(identifier)
            if size is not None and manifest.size!=size:raise ArtifactError("Language dependency size mismatch")
        return bundle

    def snapshot(self):
        available=[]
        if self.knowledge.allowed:
            for name in BASES:
                try:self._family({"base_name":name},None);available.append(name)
                except (OSError,ValueError,TypeError):pass
        return {"available":available,"stored_bytes":self.store.total_bytes(),"sharing":bool(self.knowledge.allowed)}


def fetch_family(work,store,root,name,provider,*,cancel_event=None,progress=lambda *_:None,receive_budget=None,preflight=lambda *_:None):
    """Resume immutable dependencies, then make the complete family discoverable."""
    result=work.submit("language.family.v1",{"base_name":name},ResourceRequirements(kind="artifact"),
                       peer_id=provider,timeout_s=30,cancel_event=cancel_event)
    if result.status!="ok":raise ArtifactError("Public language family is unavailable")
    bundle=result.output;release,descriptor=verify_bundle(bundle)
    if descriptor["base_name"]!=name:raise ArtifactError("Peer supplied a different language family")
    preflight(bundle)
    service=ArtifactService(work,store,namespace="language.artifact",receive_budget=receive_budget,register_handlers=False)
    for identifier,size in dependencies(bundle):
        service.fetch(identifier,peer_id=provider,timeout_s=3600,cancel_event=cancel_event,expected_size=size)
        manifest=store.manifest(identifier)
        if size is not None and manifest.size!=size:raise ArtifactError("Public dependency has the wrong size")
        progress(identifier,manifest.size)
    atomic_json(Path(root)/"bundles"/f"{name}.json",bundle)
    return bundle
