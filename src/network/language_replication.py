"""Repair accepted language knowledge through interchangeable public providers."""
import json

from src.release.language_versions import (LanguageVersionRegistry,dependencies,verify_chain,
    verify_evaluation,MAX_EVALUATION_BYTES)
from src.release.manifest import ReleaseError
from .artifact_store import ArtifactService,ArtifactError
from .language_version_service import fetch_history


def replicate_version(work,store,root,name,providers,*,cancel_event=None,receive_budget=None,
                      allowed=lambda:True,preflight=lambda *_:None,progress=lambda *_:None):
    if not isinstance(providers,list) or not 1<=len(providers)<=8 or len(set(providers))!=len(providers):
        raise ValueError('Use one to eight distinct language providers')
    registry=LanguageVersionRegistry(root,name);current=registry.head()
    chosen=registry.history(current['release']['release_id']);metadata_providers=[];errors=[]
    def check():
        if not allowed() or cancel_event is not None and cancel_event.is_set():
            raise ArtifactError('Language replication permission withdrawn or cancelled')
    for provider in providers:
        check()
        try:
            chain=fetch_history(work,name,provider,cancel_event=cancel_event,timeout_s=30)
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            errors.append({'provider':provider,'stage':'history','error_type':type(error).__name__})
            continue
        common=min(len(chosen),len(chain))-1
        if chain[common]['release']['release_id']!=chosen[common]['release']['release_id']:
            raise ReleaseError('Language providers advertise a fork')
        if len(chain)>len(chosen):chosen=chain
        metadata_providers.append(provider)
    verify_chain(chosen);preflight(chosen)
    transfer=ArtifactService(work,store,namespace='language.artifact',register_handlers=False,receive_budget=receive_budget)
    acquired={};checked=set()
    def fetch(identifier,size):
        check()
        if identifier in checked:return store.object_path(identifier)
        if store.has(identifier) and store.manifest(identifier).size==size:
            checked.add(identifier);progress(identifier,size,'local');return store.object_path(identifier)
        failures=[]
        # A partial replica can still hold this immutable object even when its
        # history endpoint refuses to advertise an incomplete model family.
        for provider in providers:
            check()
            try:
                path=transfer.fetch(identifier,peer_id=provider,expected_size=size,timeout_s=3600,
                    cancel_event=cancel_event,progress=lambda done,total:progress(identifier,done,provider))
                checked.add(identifier);acquired[identifier]=provider
                return path
            except (OSError,ValueError,RuntimeError,TimeoutError) as error:
                failures.append(type(error).__name__)
        raise ArtifactError('All language providers failed the same immutable object: '+','.join(failures))
    # Check transition evidence before downloading an updated adapter or base.
    for index,bundle in enumerate(chosen[1:],1):
        descriptor=bundle['descriptor']
        path=fetch(descriptor['evaluation_id'],descriptor['evaluation_bytes'])
        if path.stat().st_size>MAX_EVALUATION_BYTES or store.manifest(descriptor['evaluation_id']).kind!='evaluation':
            raise ReleaseError('Language version has no bounded evaluation artifact')
        verify_evaluation(json.loads(path.read_text()),chosen[index-1],bundle)
    for bundle in chosen:
        for identifier,size in dependencies(bundle):fetch(identifier,size)
    check()
    release=registry.install(chosen,store)
    return {'release_id':release.release_id,'sequence':release.sequence,'model_id':release.model_artifact_id,
        'providers':list(providers),'metadata_providers':metadata_providers,
        'acquired_objects':acquired,'history_failures':errors,
        'active_expert_changed':False,'resource_permissions_changed':False}
