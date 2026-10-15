"""Lossless transfer, legacy peers, bounded decoding and consent at both namespaces."""
import base64
import hashlib
import random
import time
import zlib
from types import SimpleNamespace

import pytest

from src.contribution import ResourceRequirements
from src.network.artifact_codec import CHUNK_BYTES, ENCODING, decode_chunk, encode_chunk
from src.network.artifact_store import ArtifactError, ArtifactService, ArtifactStore
from src.network.client_state import ClientSettings
from src.network.language_families import LanguageFamilies
from src.network.public_knowledge import KnowledgeSettings, PublicKnowledgeService
from src.network.work_protocol import WorkUnit
from test_public_knowledge import RESOURCES, node, run_fixture


@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 49, CHUNK_BYTES-1, CHUNK_BYTES])
def test_lossless_round_trip_including_unaligned_tail(length):
    chunk = (bytes(range(256))*((length+255)//256))[:length]
    value = encode_chunk(chunk)
    assert decode_chunk(value, length) == chunk
    assert len(value['data']) <= 4*((length+2)//3)
    if length > 1024:
        assert value['encoding'] == ENCODING


def test_incompressible_chunks_keep_raw_transport_without_expanding():
    content = random.Random(1014).randbytes(CHUNK_BYTES)
    encoded = encode_chunk(content)
    assert encoded['encoding'] == 'raw'
    assert decode_chunk(encoded, len(content)) == content
    for value in [b'', b'x'*(CHUNK_BYTES+1)]:
        with pytest.raises(ValueError):
            encode_chunk(value)


@pytest.mark.parametrize('problem', ['codec', 'codec_type', 'base64', 'oversize', 'bomb', 'truncated', 'trailing', 'concatenated', 'not_zlib', 'length'])
def test_malformed_encoded_chunks_fail_before_unbounded_allocation(problem):
    packed = zlib.compress(b'\0'*100)
    value = {'encoding': ENCODING, 'data': base64.b64encode(packed).decode()}
    if problem == 'codec':
        value['encoding'] = 'peer-selected-codec'
    elif problem == 'codec_type':
        value['encoding'] = []
    elif problem == 'base64':
        value['data'] = '!!!'
    elif problem == 'oversize':
        value['data'] = 'A'*1000
    else:
        packed = {'bomb': zlib.compress(b'\0'*1_000_000), 'truncated': packed[:-1],
                  'trailing': packed+b'extra', 'concatenated': packed+packed,
                  'not_zlib': b'bad', 'length': zlib.compress(b'\0'*99)}[problem]
        value['data'] = base64.b64encode(packed).decode()
    with pytest.raises(ValueError):
        decode_chunk(value, 100)


@pytest.mark.parametrize('legacy_provider,prefer_compressed', [(False, True), (True, True), (False, False)])
def test_negotiation_resumes_exact_artifact_and_preserves_legacy_operations(tmp_path, legacy_provider, prefer_compressed):
    source = tmp_path/'source'; content = bytes(range(256))*1000; source.write_bytes(content)
    provider_store = ArtifactStore(tmp_path/'provider')
    manifest = provider_store.import_file(source, kind='gradient')
    provider = ArtifactService(None, provider_store, register_handlers=False)
    calls = []
    class Work:
        def submit(self, operation, payload, *_args, **_kwargs):
            calls.append(operation)
            if operation.endswith('.describe.v1'):
                output = provider._describe(payload, None)
                if legacy_provider:
                    output.pop('chunk_encodings')
            else:
                output = (provider._read_encoded_chunk if operation.endswith('.v2') else provider._read_chunk)(payload, None)
            return SimpleNamespace(status='ok', output=output)
    # An interrupted legacy transfer may leave a prefix unaligned to four bytes.
    store = ArtifactStore(tmp_path/'receiver')
    partial, _ = store.begin_receive(manifest); partial.write_bytes(content[:17])
    service = ArtifactService(Work(), store, register_handlers=False, prefer_compressed=prefer_compressed)
    observed = []
    downloaded = service.fetch(manifest.artifact_id, peer_id='a'*64, expected_size=len(content), observer=observed.append)
    assert downloaded.read_bytes() == content and hashlib.sha256(downloaded.read_bytes()).hexdigest() == manifest.artifact_id
    version = 'v2' if prefer_compressed and not legacy_provider else 'v1'
    assert all(operation.endswith(version) for operation in calls[1:])
    assert sum(row['bytes'] for row in observed) == len(content)-17
    if version == 'v2':
        assert sum(row['encoded_data_bytes'] for row in observed) < len(content)//4


@pytest.mark.parametrize('namespace', ['artifact', 'language.artifact'])
def test_encoded_artifact_work_and_cache_respect_sharing_and_wire_budget(tmp_path, namespace):
    catalog, _ = run_fixture(tmp_path/'catalog')
    service = PublicKnowledgeService(node(tmp_path/'node'), tmp_path/'node', RESOURCES, catalog=catalog)
    families = LanguageFamilies(service, tmp_path/'node') if namespace != 'artifact' else None
    store = families.store if families else service.store
    path = tmp_path/'gradient'; path.write_bytes(bytes(range(64))*1024)
    manifest = store.import_file(path, kind='gradient')
    calls = []; outputs = []
    service.node.send_work_result = lambda _peer, value: outputs.append(value) or True
    try:
        service.update(ClientSettings(), KnowledgeSettings(enabled=True, consented_at=time.time()))
        service.budget.wait = lambda size, **_kwargs: calls.append(size)
        identity = service.node.identity.node_id_hex
        work = WorkUnit.create(origin_node_id=identity, operation=namespace+'.read-chunk.v2', kind='artifact',
            requirements=ResourceRequirements(kind='artifact'), payload={'artifact_id': manifest.artifact_id, 'offset': 0, 'length': len(path.read_bytes())}, timeout_s=30)
        service.work._execute_offer(work.to_dict(), identity)
        assert outputs[-1]['status'] == 'ok' and outputs[-1]['output']['encoding'] == ENCODING
        charge = len(outputs[-1]['output']['data'])+2048
        service.work._execute_offer(work.to_dict(), identity)
        assert calls == [charge, charge]
        service.allowed = False
        service.work._execute_offer(work.to_dict(), identity)
        assert outputs[-1]['status'] == 'rejected' and calls == [charge, charge]
        fresh = WorkUnit.create(origin_node_id=identity, operation=work.operation, kind='artifact',
            requirements=ResourceRequirements(kind='artifact'), payload=work.payload, timeout_s=30)
        service.work._execute_offer(fresh.to_dict(), identity)
        assert outputs[-1]['status'] != 'ok' and calls == [charge, charge]
    finally:
        service.close()
