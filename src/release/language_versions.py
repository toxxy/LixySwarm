"""Evaluated adapter descendants anchored to the immutable public genesis.

This is the known-operator pilot policy. A transport provider cannot choose a
publisher, upstream model, dataset or quality gate. No tensor library is used.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from src.learning.public_bases import BASES
from src.learning.public_conversation import public_corpus, corpus_digest
from src.learning.public_quality import preservation_verdict,digest_cases as factual_digest
from src.learning.public_language_gate import preservation_verdict as instruction_verdict,digest_cases as instruction_digest
from src.network.artifact_store import ArtifactManifest
from src.network.telemetry import atomic_json
from src.network.work_protocol import ResultReceipt, WorkResult
from src.network.client_state import NodeStateLock
from . import language_family as genesis
from .manifest import ReleaseError, ReleaseManifest, TrustPolicy
from .adapter_format import validate_adapter

MAX_CHAIN = 64
MAX_EVALUATION_BYTES = 256*1024
HEX = re.compile(r'^[0-9a-f]{64}$')
WORKERS = {
    'c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255',
    'ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0',
}


def gate_id():
    definition={'version':1,'dataset_id':corpus_digest(public_corpus()),'facts':factual_digest(),
        'instructions':instruction_digest(),'minimum_relative_nll_improvement':.001,'minimum_absolute_nll_improvement':1e-7,
        'minimum_instruction_passed':37,'mandatory_instructions':['en_instruction','en_public_context','es_instruction','es_public_context'],
        'physical_workers':sorted(WORKERS),'minimum_tokens':16,'maximum_contributions':16,'maximum_shard_examples':8,
        'decoding':'greedy','max_output_tokens':128,'scope':'Known-operator public instruction/context pilot'}
    return hashlib.sha256(genesis.canonical(definition)).hexdigest()


def validate_bundle(bundle):
    """Verify one bundle's signature/bindings; anchoring requires verify_chain."""
    if not isinstance(bundle, dict) or set(bundle) != {'release', 'descriptor'}:
        raise ReleaseError('Invalid versioned language bundle')
    descriptor = bundle['descriptor']
    if isinstance(descriptor, dict) and descriptor.get('version') == 1:
        return genesis.verify_bundle(bundle)
    fields = {'version', 'base_name', 'base_hash', 'files', 'adapter_id', 'adapter_bytes', 'evaluation_id', 'evaluation_bytes'}
    if not isinstance(descriptor, dict) or set(descriptor) != fields or type(descriptor['version']) is not int or descriptor['version'] != 2:
        raise ReleaseError('Invalid language update descriptor')
    name = descriptor['base_name']
    if name not in BASES or not HEX.fullmatch(str(descriptor['adapter_id'])) or type(descriptor['adapter_bytes']) is not int or not 0 < descriptor['adapter_bytes'] <= 32*1024**2:
        raise ReleaseError('Invalid bounded language adapter')
    if type(descriptor['evaluation_bytes']) is not int or not 0 < descriptor['evaluation_bytes'] <= MAX_EVALUATION_BYTES:
        raise ReleaseError('Language evaluation exceeds its bound')
    # Reuse the closed upstream schema; only adapter/evaluation identity vary.
    upstream = {k: v for k, v in descriptor.items() if k != 'evaluation_bytes'}
    upstream.update(version=1, adapter_id=genesis.ADAPTERS[name],
        adapter_bytes=4599712 if name == 'qwen3-0.6b' else 6434736)
    genesis.validate_descriptor(upstream)
    release = ReleaseManifest.from_dict(bundle['release'])
    if (release.sequence < 1 or release.sequence >= MAX_CHAIN or release.model_format != 'safetensors-v1'
            or release.model_artifact_id != descriptor['adapter_id']
            or release.config_artifact_id != hashlib.sha256(genesis.canonical(descriptor)).hexdigest()
            or release.tokenizer_artifact_id != BASES[name]['tokenizer.json']):
        raise ReleaseError('Language update does not bind its dependencies')
    TrustPolicy(trusted_signers=(genesis.SIGNER,)).authorize(release)
    return release, descriptor


def verify_chain(chain):
    if not isinstance(chain, list) or not 1 <= len(chain) <= MAX_CHAIN:
        raise ReleaseError('Invalid language history length')
    first, descriptor = genesis.verify_bundle(chain[0])
    name = descriptor['base_name']
    previous = first
    for bundle in chain[1:]:
        release, row = validate_bundle(bundle)
        if (row['base_name'] != name or release.sequence != previous.sequence+1
                or release.previous_release_id != previous.release_id):
            raise ReleaseError('Language history is forked, skipped or from another family')
        previous = release
    return previous


def dependencies(bundle):
    release, row = validate_bundle(bundle)
    if release.sequence == 0:
        return genesis.dependencies(bundle)
    return [(r['artifact_id'], r['bytes']) for r in row['files']]+[
        (release.model_artifact_id, row['adapter_bytes']),
        (release.config_artifact_id, len(genesis.canonical(row))),
        (row['evaluation_id'], row['evaluation_bytes'])]


def verify_evaluation(report, parent_bundle, candidate_bundle):
    parent, before = validate_bundle(parent_bundle)
    candidate, after = validate_bundle(candidate_bundle)
    return _verify_evaluation(report,parent,before,candidate,after)


def _verify_evaluation(report,parent,before,candidate,after):
    fields = {'version','gate_id', 'private_inputs', 'base_name', 'base_hash', 'dataset_id', 'parent_release_id',
        'adapter_id', 'baseline_nll', 'candidate_nll', 'quality_before', 'quality_after',
        'instruction_before','instruction_after','contributions'}
    if (not isinstance(report, dict) or set(report) != fields or type(report['version']) is not int or report['version'] != 1
            or report['private_inputs'] is not False or report['base_name'] != before['base_name']
            or report['gate_id']!=gate_id()
            or report['base_hash'] != before['base_hash'] or after['base_hash'] != before['base_hash']
            or report['parent_release_id'] != parent.release_id or candidate.previous_release_id != parent.release_id
            or candidate.sequence!=parent.sequence+1 or before['base_name']!=after['base_name']
            or report['adapter_id'] != candidate.model_artifact_id or report['dataset_id'] != corpus_digest(public_corpus())):
        raise ReleaseError('Language evaluation is not bound to public data and this transition')
    baseline, value = report['baseline_nll'], report['candidate_nll']
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (baseline, value))
            or value >= baseline-max(1e-7, baseline*.001)):
        raise ReleaseError('Language candidate does not improve its fixed evaluation')
    for rows in (report['quality_before'], report['quality_after'],report['instruction_before'],report['instruction_after']):
        if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {'case', 'answer'}
                or not isinstance(row['answer'], str) or len(row['answer'].encode()) > 1024 for row in rows):
            raise ReleaseError('Invalid bounded factual evaluation')
    if not preservation_verdict(report['quality_before'], report['quality_after'])['eligible_for_promotion']:
        raise ReleaseError('Language candidate regresses factual preservation')
    if not instruction_verdict(report['instruction_before'],report['instruction_after'])['eligible_for_promotion']:
        raise ReleaseError('Language candidate regresses instructions or context')
    proof=verify_contributions(report['contributions'],base_name=before['base_name'],
        adapter_id=parent.model_artifact_id,dataset_id=report['dataset_id'])
    if len(genesis.canonical(report)) > MAX_EVALUATION_BYTES:
        raise ReleaseError('Language evaluation exceeds its bound')
    return {**proof,'candidate_id':candidate.release_id}


def verify_contributions(contributions,*,base_name,adapter_id,dataset_id,require_distributed=True):
    """Check signed physical receipts; promotion always uses both-worker mode."""
    if (type(require_distributed) is not bool or base_name not in BASES
            or dataset_id!=corpus_digest(public_corpus())):
        raise ReleaseError('Invalid public contribution policy or dataset')
    if not isinstance(contributions, list) or not (2 if require_distributed else 1) <= len(contributions) <= 16:
        raise ReleaseError('Distributed language promotion requires contributions from both PCs')
    workers, jobs, ranges, tokens = set(), set(), [], 0
    for row in contributions:
        result = WorkResult.from_dict(row)
        receipt = ResultReceipt.from_dict(result.receipt)
        if (result.status != 'ok' or receipt.worker_node_id not in WORKERS
                or receipt.requester_node_id not in WORKERS | {genesis.SIGNER}
                or not receipt.verify(result, expected_worker_id=receipt.worker_node_id,
                    expected_requester_id=receipt.requester_node_id) or result.job_id in jobs):
            raise ReleaseError('Invalid or repeated physical contribution receipt')
        output = result.output
        expected = {'base_hash', 'adapter_id', 'dataset_id', 'start', 'count', 'tokens', 'loss', 'retention_loss',
            'gradient_id', 'gradient_bytes', 'gradient'}
        if (set(output) != expected or output['base_hash'] != genesis.base_hash(base_name)
                or output['adapter_id'] != adapter_id or output['dataset_id'] != dataset_id
                or type(output['start']) is not int or type(output['count']) is not int
                or not 1 <= output['count'] <= 8 or not 0 <= output['start'] <= len(public_corpus()['train'])-output['count']
                or type(output['tokens']) is not int or not 1 <= output['tokens'] <= 1536
                or any(type(output[k]) not in (int, float) or not math.isfinite(output[k]) for k in ('loss', 'retention_loss'))):
            raise ReleaseError('Language contribution is from another version, dataset or shard')
        manifest = ArtifactManifest.from_dict(output['gradient'], max_bytes=32*1024**2)
        if (manifest.kind != 'gradient' or manifest.artifact_id != output['gradient_id']
                or type(output['gradient_bytes']) is not int or manifest.size != output['gradient_bytes']):
            raise ReleaseError('Language contribution does not bind its gradient')
        jobs.add(result.job_id)
        workers.add(receipt.worker_node_id)
        ranges.append((output['start'], output['start']+output['count']))
        tokens += output['tokens']
    ranges.sort()
    if ((require_distributed and (workers != WORKERS or tokens < 16))
            or any(a[1] > b[0] for a, b in zip(ranges, ranges[1:]))):
        raise ReleaseError('Language promotion requires distinct physical workers and disjoint useful shards')
    return {'physical_workers':sorted(workers),'tokens':tokens}


def build_update(chain,adapter,evaluation,report,publisher):
    """Sign only after the public transition gate passes; no transport authority."""
    parent=verify_chain(chain);before=chain[-1]['descriptor']
    if publisher.node_id_hex!=genesis.SIGNER:
        raise ReleaseError('Only the locally trusted pilot publisher may sign')
    adapter.validate(max_bytes=32*1024**2);evaluation.validate(max_bytes=MAX_EVALUATION_BYTES)
    encoded=genesis.canonical(report)
    if (adapter.kind!='model' or adapter.artifact_id==parent.model_artifact_id or evaluation.kind!='evaluation'
            or evaluation.artifact_id!=hashlib.sha256(encoded).hexdigest() or evaluation.size!=len(encoded)):
        raise ReleaseError('Candidate/evaluation artifacts do not bind a new adapter')
    descriptor={'version':2,'base_name':before['base_name'],'base_hash':before['base_hash'],'files':before['files'],
        'adapter_id':adapter.artifact_id,'adapter_bytes':adapter.size,'evaluation_id':evaluation.artifact_id,
        'evaluation_bytes':evaluation.size}
    draft=ReleaseManifest.create(model_artifact_id=adapter.artifact_id,model_format='safetensors-v1',
        sequence=parent.sequence+1,previous_release_id=parent.release_id,
        config_artifact_id=hashlib.sha256(genesis.canonical(descriptor)).hexdigest(),
        tokenizer_artifact_id=parent.tokenizer_artifact_id,minimum_client_version='0.4.0a8')
    _verify_evaluation(report,parent,before,draft,descriptor)
    bundle={'release':draft.sign(publisher).to_dict(),'descriptor':descriptor}
    validate_bundle(bundle)
    return bundle


class LanguageVersionRegistry:
    def __init__(self, root, name):
        if name not in BASES:
            raise ReleaseError('Unknown language family')
        self.name = name
        self.root = Path(root)/'versions'/name
        self.bundles = self.root/'bundles'
        self.head_path = self.root/'head.json'

    def head(self):
        if not self.head_path.exists():
            return genesis.INITIAL_BUNDLES[self.name]
        value = json.loads(self.head_path.read_text())
        if not isinstance(value, dict) or set(value) != {'release_id'} or not HEX.fullmatch(str(value['release_id'])):
            raise ReleaseError('Invalid language head')
        chain = self.history(value['release_id'])
        verify_chain(chain)
        return chain[-1]

    def history(self, identifier):
        descending = []
        for _ in range(MAX_CHAIN):
            if not HEX.fullmatch(str(identifier)):
                raise ReleaseError('Invalid language version identity')
            initial = genesis.INITIAL_BUNDLES[self.name]
            if identifier == initial['release']['release_id']:
                descending.append(initial)
                break
            path = self.bundles/f'{identifier}.json'
            if path.stat().st_size > 96*1024:
                raise ReleaseError('Language history bundle exceeds its bound')
            bundle = json.loads(path.read_text())
            release, descriptor = validate_bundle(bundle)
            if release.release_id != identifier or descriptor['base_name'] != self.name:
                raise ReleaseError('Language history identity/family mismatch')
            descending.append(bundle)
            identifier = release.previous_release_id
        else:
            raise ReleaseError('Language history exceeds its bound')
        result = list(reversed(descending))
        verify_chain(result)
        return result

    def install(self, chain, store):
        with NodeStateLock(self.root/'transaction'):
            return self._install(chain,store)

    def _install(self,chain,store):
        target = verify_chain(chain)
        if chain[0]['descriptor']['base_name'] != self.name:
            raise ReleaseError('Language history belongs to another family')
        current = validate_bundle(self.head())[0]
        if target.sequence < current.sequence or chain[current.sequence]['release']['release_id'] != current.release_id:
            raise ReleaseError('Language head cannot go backwards or fork')
        checked = set()
        for index, bundle in enumerate(chain):
            if store.manifest(bundle['release']['model_artifact_id']).kind!='model':
                raise ReleaseError('Language adapter is not a model artifact')
            for identifier, size in dependencies(bundle):
                if identifier not in checked:
                    if not store.has(identifier) or store.manifest(identifier).size != size:
                        raise ReleaseError('Language history dependencies are incomplete or corrupt')
                    checked.add(identifier)
            if index:
                if bundle['release']['model_artifact_id']==chain[index-1]['release']['model_artifact_id']:
                    raise ReleaseError('Language update has no new adapter')
                if store.manifest(bundle['descriptor']['evaluation_id']).kind!='evaluation':
                    raise ReleaseError('Language update has no evaluation artifact')
                configuration=store.object_path(genesis.BASES[self.name]['config.json'])
                validate_adapter(store.object_path(bundle['release']['model_artifact_id']),
                    json.loads(configuration.read_text()),bundle['descriptor']['base_hash'])
                evaluation = store.object_path(bundle['descriptor']['evaluation_id'])
                verify_evaluation(json.loads(evaluation.read_text()), chain[index-1], bundle)
        self.bundles.mkdir(parents=True, exist_ok=True)
        for bundle in chain[1:]:
            atomic_json(self.bundles/f"{bundle['release']['release_id']}.json", bundle)
        atomic_json(self.head_path, {'release_id': target.release_id})
        return target
