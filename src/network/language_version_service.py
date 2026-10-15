"""Versioned family discovery on the existing physical worker identity."""
import time
from pathlib import Path

from src.contribution import ResourceRequirements
from src.release.language_versions import (HEX, MAX_CHAIN, LanguageVersionRegistry,
    dependencies, validate_bundle, verify_chain)
from src.release.manifest import ReleaseError
from .artifact_store import ArtifactService
from .work_protocol import ResultReceipt
from src.release.language_family import canonical

HEAD_OPERATION = 'language.version-head.v1'
HEAD_OPERATION_V2 = 'language.version-head.v2'
HEAD_OPERATION_V3 = 'language.version-head.v3'
HISTORY_OPERATION = 'language.version-history.v1'


class LanguageVersionService:
    def __init__(self, families):
        self.families = families
        work = families.knowledge.work
        for operation, handler in [(HEAD_OPERATION, self.head), (HEAD_OPERATION_V2,self.current_head), (HEAD_OPERATION_V3,self.precision_head), (HISTORY_OPERATION, self.history)]:
            work.register_handler(operation, 'artifact', families._guard(handler), permit_cached=families._permit_cached)
        work.node.capabilities['language_versions_v2']=True
        work.node.capabilities['language_versions_v3']=True

    def head(self, payload, execution):
        return self._head(payload,execution,maximum_descriptor=2)

    def current_head(self,payload,execution):
        return self._head(payload,execution,maximum_descriptor=3)

    def precision_head(self,payload,execution):
        return self._head(payload,execution,maximum_descriptor=4)

    def _head(self,payload,execution,*,maximum_descriptor):
        if not isinstance(payload, dict) or set(payload) != {'base_name'}:
            raise ReleaseError('Invalid language-head request')
        # Signed history remains useful if this replica loses an object: other
        # providers may hold the missing content. Complete-family availability
        # is advertised separately by LanguageFamilies.
        registry=LanguageVersionRegistry(self.families.root, payload['base_name'])
        bundle = registry.head()
        if bundle['descriptor']['version']>maximum_descriptor:
            bundle=next(row for row in reversed(registry.history(bundle['release']['release_id'])) if row['descriptor']['version']<=maximum_descriptor)
        self._charge(bundle,execution)
        return bundle

    def _charge(self,output,execution):
        self.families.knowledge.budget.wait(len(canonical(output))+2048,
            cancel_event=self.families.knowledge.cancel,deadline=execution.deadline)

    def history(self, payload, execution):
        if (not isinstance(payload, dict) or set(payload) != {'base_name', 'release_id'}
                or not HEX.fullmatch(str(payload['release_id']))):
            raise ReleaseError('Invalid language-history request')
        chain = LanguageVersionRegistry(self.families.root, payload['base_name']).history(payload['release_id'])
        page = list(reversed(chain))[:2]
        output={'bundles': page, 'next_release_id': page[-1]['release']['previous_release_id']}
        self._charge(output,execution)
        return output


def fetch_history(work, name, provider, *, cancel_event=None, timeout_s=60):
    deadline = time.monotonic()+min(max(float(timeout_s), 1), 3600)
    def request(operation, payload):
        if time.monotonic() >= deadline or cancel_event is not None and cancel_event.is_set():
            raise ReleaseError('Language history timed out or was cancelled')
        result = work.submit(operation, payload, ResourceRequirements(kind='artifact'), peer_id=provider,
            timeout_s=deadline-time.monotonic(), cancel_event=cancel_event)
        if result.status != 'ok' or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=provider, expected_requester_id=work.node.identity.node_id_hex):
            raise ReleaseError('Language history response is unavailable or unsigned')
        return result.output
    peers=getattr(work.node,'peers',lambda:[])()
    version_two=any(peer.get('node_id')==provider and peer.get('capabilities',{}).get('language_versions_v2') is True for peer in peers)
    version_three=any(peer.get('node_id')==provider and peer.get('capabilities',{}).get('language_versions_v3') is True for peer in peers)
    head = request(HEAD_OPERATION_V3 if version_three else HEAD_OPERATION_V2 if version_two else HEAD_OPERATION, {'base_name': name})
    target, descriptor = validate_bundle(head)
    if descriptor['base_name'] != name:
        raise ReleaseError('Language head belongs to another family')
    cursor, descending = target.release_id, []
    while cursor:
        page = request(HISTORY_OPERATION, {'base_name': name, 'release_id': cursor})
        if (not isinstance(page, dict) or set(page) != {'bundles', 'next_release_id'}
                or not isinstance(page['bundles'], list) or not 1 <= len(page['bundles']) <= 2):
            raise ReleaseError('Invalid language history page')
        for bundle in page['bundles']:
            release, row = validate_bundle(bundle)
            if release.release_id != cursor or row['base_name'] != name:
                raise ReleaseError('Language history continuation does not match')
            descending.append(bundle)
            if len(descending) > MAX_CHAIN:
                raise ReleaseError('Language history exceeds its bound')
            cursor = release.previous_release_id
        if page['next_release_id'] != cursor:
            raise ReleaseError('Language history continuation was changed')
    result = list(reversed(descending))
    if verify_chain(result).release_id != target.release_id:
        raise ReleaseError('Language history does not reach its announced head')
    return result


def acquire_history(work, store, root, name, provider, *, cancel_event=None, receive_budget=None,
                    preflight=lambda *_: None, progress=lambda *_: None):
    chain = fetch_history(work, name, provider, cancel_event=cancel_event)
    preflight(chain)
    client = ArtifactService(work, store, namespace='language.artifact', register_handlers=False, receive_budget=receive_budget)
    done = set()
    for bundle in chain:
        for identifier, size in dependencies(bundle):
            if identifier in done:
                continue
            client.fetch(identifier, peer_id=provider, expected_size=size, timeout_s=3600, cancel_event=cancel_event)
            done.add(identifier)
            progress(identifier, size)
    # The whole chain and every evaluation pass before a durable head changes.
    return LanguageVersionRegistry(root, name).install(chain, store)
