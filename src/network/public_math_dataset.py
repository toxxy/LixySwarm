"""Acquire only pinned public math data through the existing artifact protocol.

Transfer proofs omit the data itself. They must be reconstructed from the
verified local artifact before checking the original worker receipt.
"""
import base64
from copy import deepcopy
import hashlib
import math
import re

from src.contribution import ResourceRequirements
from src.learning.public_math import ARTIFACTS, canonical, curriculum_id, load_snapshot
from .artifact_codec import CHUNK_BYTES, ENCODING, encode_chunk
from .artifact_store import ArtifactError, ArtifactManifest, ArtifactService
from .work_protocol import ResultReceipt, WorkResult

OPERATIONS = {'language.artifact.describe.v1', 'language.artifact.read-chunk.v1', 'language.artifact.read-chunk.v2'}
OFFER_FIELDS = {'job_id', 'origin_node_id', 'operation', 'kind', 'requirements', 'payload', 'created_at', 'deadline', 'nonce'}


def transfer_proof(offer, result, provider):
    if offer.get('operation') not in OPERATIONS or result.status != 'ok':
        raise ValueError('Dataset proof requires a successful supported artifact operation')
    if not ResultReceipt.from_dict(result.receipt).verify(result, expected_worker_id=provider,
            expected_requester_id=offer['origin_node_id']):
        raise ValueError('Dataset transfer result has another worker or requester')
    body = deepcopy(result.to_dict())
    data = body['output'].pop('data', None)
    return {'provider_id': provider, 'offer': deepcopy(offer), 'result_without_chunk_data': body,
            'chunk_data_omitted': data is not None,
            'encoded_data_sha256': hashlib.sha256(data.encode('ascii')).hexdigest() if data is not None else None}


def verify_transfer_proofs(proofs, store, *, requester_id, providers, splits=('train', 'evaluation')):
    """Verify historical offers, reconstructed result bodies and complete coverage."""
    if not isinstance(proofs, list) or len(proofs) > 256 or len(canonical(proofs)) > 1024**2:
        raise ValueError('Dataset transfer proofs exceed their bound')
    expected = {ARTIFACTS[split][0]: ARTIFACTS[split][1] for split in splits}
    coverage = {identifier: [] for identifier in expected}
    jobs = set()
    for proof in proofs:
        if set(proof) != {'provider_id', 'offer', 'result_without_chunk_data', 'chunk_data_omitted', 'encoded_data_sha256'}:
            raise ValueError('Invalid dataset proof schema')
        offer = proof['offer']; provider = proof['provider_id']
        if (not isinstance(offer, dict) or set(offer) != OFFER_FIELDS or provider not in providers
                or offer['origin_node_id'] != requester_id or offer['operation'] not in OPERATIONS
                or offer['kind'] != 'artifact' or offer['requirements'] != ResourceRequirements(kind='artifact').__dict__
                or not isinstance(offer['nonce'], str) or not re.fullmatch('[0-9a-f]{32}', offer['nonce'])
                or hashlib.sha256(canonical({k: v for k, v in offer.items() if k != 'job_id'})).hexdigest() != offer['job_id']):
            raise ValueError('Dataset proof offer does not bind the fixed artifact request')
        payload = offer['payload']; identifier = payload.get('artifact_id')
        if identifier not in expected:
            raise ValueError('Dataset proof requests an unapproved artifact')
        body = deepcopy(proof['result_without_chunk_data'])
        output = body['output']
        if offer['operation'].endswith('describe.v1'):
            if set(payload) != {'artifact_id'} or proof['chunk_data_omitted'] is not False or proof['encoded_data_sha256'] is not None:
                raise ValueError('Invalid dataset manifest proof')
            if set(output) != {'manifest', 'chunk_encodings'} or output['chunk_encodings'] != ['raw', ENCODING]:
                raise ValueError('Unknown dataset manifest response')
            manifest = ArtifactManifest.from_dict(output['manifest'])
            if manifest.artifact_id != identifier or manifest.size != expected[identifier] or manifest.kind != 'dataset':
                raise ValueError('Dataset manifest has another content identity, size or kind')
        else:
            if (set(payload) != {'artifact_id', 'offset', 'length'} or proof['chunk_data_omitted'] is not True
                    or type(payload['offset']) is not int or type(payload['length']) is not int
                    or not 0 <= payload['offset'] < expected[identifier] or not 1 <= payload['length'] <= CHUNK_BYTES
                    or payload['offset']+payload['length'] > expected[identifier] or 'data' in output):
                raise ValueError('Invalid dataset chunk proof bounds')
            if set(output) != ({'offset', 'chunk_sha256', 'encoding'} if offer['operation'].endswith('v2') else {'offset', 'chunk_sha256'}):
                raise ValueError('Unknown dataset chunk response')
            chunk = store.read_chunk(identifier, payload['offset'], payload['length'])
            if output.get('encoding', 'raw') == 'raw':
                data = base64.b64encode(chunk).decode('ascii')
            elif offer['operation'].endswith('v2') and output.get('encoding') == ENCODING:
                encoded = encode_chunk(chunk)
                if encoded['encoding'] != ENCODING:
                    raise ValueError('Dataset chunk encoding cannot be reproduced')
                data = encoded['data']
            else:
                raise ValueError('Unknown dataset chunk encoding')
            if (hashlib.sha256(data.encode('ascii')).hexdigest() != proof['encoded_data_sha256']
                    or output.get('offset') != payload['offset'] or output.get('chunk_sha256') != hashlib.sha256(chunk).hexdigest()):
                raise ValueError('Dataset chunk proof differs from verified local bytes')
            output['data'] = data
            coverage[identifier].append((payload['offset'], payload['offset']+payload['length']))
        result = WorkResult.from_dict(body)
        if (result.job_id != offer['job_id'] or result.job_id in jobs or result.status != 'ok' or result.error
                or any(type(t) not in (int, float) or not math.isfinite(t)
                       for t in [offer['created_at'], offer['deadline'], result.finished_at])
                or not offer['created_at'] <= result.finished_at <= offer['deadline']
                or not 1 <= offer['deadline']-offer['created_at'] <= 3600
                or not ResultReceipt.from_dict(result.receipt).verify(result, expected_worker_id=provider,
                                                                    expected_requester_id=requester_id)):
            raise ValueError('Dataset proof has an unbound, repeated, late or unsigned result')
        jobs.add(result.job_id)
    complete = {}
    for identifier, ranges in coverage.items():
        ranges.sort()
        complete[identifier] = bool(ranges and ranges[0][0] == 0 and ranges[-1][1] == expected[identifier]
                                   and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
    return {'verified_signed_results': len(jobs), 'complete_byte_coverage': complete,
            'curriculum_id': curriculum_id(), 'chunk_data_reconstructed_before_signature_verification': True}


def replicate_dataset(work, store, providers, *, splits=('train', 'evaluation'), receive_budget=None,
                      cancel_event=None, allowed=lambda: True, preflight=lambda *_: None, observer=None):
    if (not providers or not splits or len(set(splits)) != len(splits) or any(split not in ARTIFACTS for split in splits)):
        raise ValueError('Dataset replication requires providers and distinct pinned splits')
    if not allowed():
        raise ArtifactError('Dataset replication permission withdrawn')
    preflight(sum(size for identifier, size in (ARTIFACTS[split] for split in splits) if not store.has(identifier)))
    service = ArtifactService(work, store, namespace='language.artifact', register_handlers=False, receive_budget=receive_budget)
    acquired = []
    for split in splits:
        if not allowed():
            raise ArtifactError('Dataset replication permission withdrawn')
        identifier, size = ARTIFACTS[split]
        local = store.has(identifier)
        partial = (store.incoming/(identifier+'.part')).exists()
        error = None
        for provider in providers:
            try:
                path = service.fetch(identifier, peer_id=provider, expected_size=size, timeout_s=180,
                                     cancel_event=cancel_event, observer=observer)
                if not allowed():
                    raise ArtifactError('Dataset replication permission withdrawn')
                manifest = store.manifest(identifier)
                if manifest.kind != 'dataset':
                    raise ArtifactError('Pinned dataset has an unexpected artifact kind')
                snapshot = load_snapshot(path, split)
                acquired.append({'split': split, 'artifact_id': identifier, 'bytes': size, 'examples': len(snapshot['rows']),
                                 'initially_local': local, 'initial_partial_file': partial,
                                 'provider_id': None if local else provider})
                break
            except (OSError, ValueError) as failure:
                error = failure
                if not allowed() or cancel_event is not None and cancel_event.is_set():
                    raise ArtifactError('Dataset replication permission withdrawn') from failure
        else:
            raise ArtifactError('No provider supplied the exact pinned dataset') from error
    return {'success': True, 'curriculum_id': curriculum_id(), 'acquired': acquired,
            'private_inputs': False, 'training_started': False, 'expert_or_native_progress_changed': False}
