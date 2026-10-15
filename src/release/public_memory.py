"""Signed public-memory pilot: closed public sources, no conversation imports.

The initial policy admits two synthetic-registry epochs and independently
verified accepted-language metadata. It does not admit arbitrary free text,
private memories, neural vectors, or authority supplied by transport peers.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from src.learning.public_memory_gate import memory_cases
from src.network.chat_versions import verify_catalog
from src.network.telemetry import atomic_json
from src.network.artifact_store import ArtifactStore,digest_file
from .language_family import SIGNER

SPACE_ID='lixyswarm-public-text-registry-v1'
MAX_DOCUMENT_BYTES=512*1024
MAX_STORE_BYTES=8*1024**2
MAX_CHAIN=2
HEX=re.compile('[0-9a-f]{64}')
DOMAIN=b'LixySwarm public memory manifest v1\x00'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


POLICY_ID=digest({'version':1,'space_id':SPACE_ID,'source':'public-registry-demo-and-accepted-language-v1',
    'epochs':[0,1],'publisher':SIGNER,'max_bytes':MAX_DOCUMENT_BYTES,'max_chain':MAX_CHAIN,
    'scope':'global','personal_inputs':False,'language_family':'qwen3-1.7b'})


def public_cards(epoch,language):
    if type(epoch) is not int or epoch not in {0,1}:raise ValueError('Unknown public memory epoch')
    head=verify_catalog(language)
    if head.sequence<1:raise ValueError('Public model memory requires an accepted learned version')
    cards=[]
    for row in memory_cases():
        statement=row['statements'][-1 if epoch else 0]
        # Namespace the fictional registry so these facts never masquerade as
        # personal facts about an equally named device or project.
        if row['case'].startswith('es_'):
            statement='Registro público ficticio: '+statement
            question='Según el registro público ficticio, '+row['question']
        else:
            statement='Fictional public registry: '+statement
            question='According to the fictional public registry, '+row['question']
        cards.append({'record_id':row['case'],'kind':'synthetic-registry','text':statement,
                      'example_question':question})
    cards.extend([
        {'record_id':'language-sequence','kind':'accepted-language','text':
         f'La instantánea publicada del catálogo público verificado de qwen3-1.7b tiene la secuencia aprendida {head.sequence}. '
         f'The published snapshot of the verified public qwen3-1.7b catalog has learned sequence {head.sequence}.',
         'example_question':'Según el catálogo público, ¿cuál es la secuencia aprendida de qwen3-1.7b? Responde solo el número.'},
        {'record_id':'language-adapter','kind':'accepted-language','text':
         f'El hash del adaptador aceptado en la instantánea publicada del catálogo público de qwen3-1.7b es {head.model_artifact_id}. '
         f'The accepted adapter hash in the published public qwen3-1.7b catalog snapshot is {head.model_artifact_id}.',
         'example_question':'According to the public catalog, what is the accepted qwen3-1.7b adapter hash? Reply only with the hash.'}])
    return cards


def build_document(epoch,language):
    return {'version':1,'scope':'global','space_id':SPACE_ID,'policy_id':POLICY_ID,
        'sources':{'synthetic_epoch':epoch,'accepted_language':deepcopy(language)},'cards':public_cards(epoch,language)}


def language_source(directory):
    """Read only the accepted, explicitly public language registry and reports."""
    from .language_versions import LanguageVersionRegistry
    registry=LanguageVersionRegistry(Path(directory)/'public/language','qwen3-1.7b')
    chain=registry.history(registry.head()['release']['release_id'])
    store=ArtifactStore(Path(directory)/'public/language/artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    reports={}
    for bundle in chain[1:]:
        identifier=bundle['descriptor']['evaluation_id'];path=store.object_path(identifier)
        if path.is_symlink() or path.stat().st_size>MAX_DOCUMENT_BYTES:raise ValueError('Public language source exceeds memory bounds')
        reports[identifier]=json.loads(path.read_text())
    value={'chain':chain,'evaluations':reports};verify_catalog(value);return value


def verify_document(value,epoch):
    if (not isinstance(value,dict) or set(value)!={'version','scope','space_id','policy_id','sources','cards'}
            or type(value['version']) is not int
            or len(canonical(value))>MAX_DOCUMENT_BYTES or value['scope']!='global'
            or value['space_id']!=SPACE_ID or value['policy_id']!=POLICY_ID
            or not isinstance(value['sources'],dict) or set(value['sources'])!={'synthetic_epoch','accepted_language'}
            or type(value['sources']['synthetic_epoch']) is not int or value['sources']['synthetic_epoch']!=epoch):raise ValueError('Invalid closed public memory document')
    if value!=build_document(epoch,value['sources']['accepted_language']):
        raise ValueError('Public memory may only derive from its exact verified public sources')
    return value


def verify_manifest(value):
    fields={'version','network_id','scope','space_id','policy_id','sequence','previous_release_id',
        'content_id','content_bytes','created_at','publisher_id','release_id','signature'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('Invalid public memory manifest')
    if (type(value['version']) is not int or value['version']!=1 or value['network_id']!='LIXYMAIN'
            or value['scope']!='global' or value['space_id']!=SPACE_ID or value['policy_id']!=POLICY_ID
            or type(value['sequence']) is not int or not 0<=value['sequence']<MAX_CHAIN
            or value['publisher_id']!=SIGNER or type(value['content_bytes']) is not int
            or not 1<=value['content_bytes']<=MAX_DOCUMENT_BYTES
            or type(value['created_at']) not in (int,float) or not math.isfinite(value['created_at'])
            or not 0<value['created_at']<=time.time()+60
            or any(not isinstance(value[k],str) or not HEX.fullmatch(value[k]) for k in ['content_id','release_id'])
            or not isinstance(value['previous_release_id'],str)
            or value['sequence']==0 and value['previous_release_id']!=''
            or value['sequence']>0 and not HEX.fullmatch(value['previous_release_id'])):
        raise ValueError('Incompatible public memory authority, scope or bounds')
    body={k:v for k,v in value.items() if k not in {'release_id','signature'}}
    if digest(body)!=value['release_id']:raise ValueError('Public memory manifest hash mismatch')
    try:
        signature=base64.b64decode(value['signature'],validate=True)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(SIGNER)).verify(signature,DOMAIN+canonical(body))
    except (TypeError,ValueError,InvalidSignature) as error:raise ValueError('Invalid public memory signature') from error
    return value


def verify_chain(chain):
    if not isinstance(chain,list) or not 1<=len(chain)<=MAX_CHAIN:raise ValueError('Invalid public memory chain length')
    previous=''
    for sequence,row in enumerate(chain):
        verify_manifest(row)
        if row['sequence']!=sequence or row['previous_release_id']!=previous:
            raise ValueError('Public memory ancestry is incomplete or reordered')
        previous=row['release_id']
    return chain[-1]


def sign_document(document,chain,publisher):
    sequence=len(chain)
    if publisher.node_id_hex!=SIGNER:raise ValueError('Only the local pilot memory publisher may sign')
    if chain:verify_chain(chain)
    verify_document(document,sequence)
    body={'version':1,'network_id':'LIXYMAIN','scope':'global','space_id':SPACE_ID,'policy_id':POLICY_ID,
        'sequence':sequence,'previous_release_id':chain[-1]['release_id'] if chain else '',
        'content_id':digest(document),'content_bytes':len(canonical(document)),'created_at':time.time(),'publisher_id':SIGNER}
    value={**body,'release_id':digest(body),'signature':base64.b64encode(publisher.sign(DOMAIN+canonical(body))).decode()}
    verify_chain(chain+[value]);return value


class PublicMemoryRegistry:
    def __init__(self,directory):
        self.root=Path(directory)/'public/memory';self.path=self.root/'catalog.json'
        self.store=ArtifactStore(self.root/'artifacts',max_artifact_bytes=MAX_DOCUMENT_BYTES,max_total_bytes=MAX_STORE_BYTES)
        self.cached=None;self.stamp=None
        self.documents={}

    def chain(self):
        if not self.path.exists():return []
        stat=self.path.stat()
        if self.path.is_symlink() or stat.st_size>32*1024:raise ValueError('Public memory catalog exceeds its bounds')
        stamp=(stat.st_mtime_ns,stat.st_size,stat.st_ino)
        if stamp!=self.stamp:
            value=json.loads(self.path.read_text())
            if not isinstance(value,dict) or set(value)!={'chain'}:raise ValueError('Invalid public memory catalog')
            verify_chain(value['chain']);self.cached=value['chain'];self.stamp=stamp
        return deepcopy(self.cached)

    def document(self,manifest=None):
        chain=self.chain()
        if manifest is None:
            if not chain:return None
            manifest=chain[-1]
        verify_manifest(manifest)
        path=self.store.object_path(manifest['content_id'])
        if path.is_symlink() or not path.is_file():raise ValueError('Public memory must be a regular public object')
        stat=path.stat();stamp=(stat.st_mtime_ns,stat.st_size,stat.st_ino)
        metadata=self.store.manifest(manifest['content_id'])
        if metadata.kind!='dataset' or metadata.size!=manifest['content_bytes']:raise ValueError('Public memory artifact type mismatch')
        cached=self.documents.get(manifest['content_id'])
        if cached is not None and cached[0]==stamp:return deepcopy(cached[1])
        if digest_file(path)!=(manifest['content_id'],manifest['content_bytes']):raise ValueError('Public memory content mismatch')
        value=json.loads(path.read_text());verify_document(value,manifest['sequence'])
        self.documents[manifest['content_id']]=(stamp,value);return deepcopy(value)

    def install(self,chain):
        from src.network.client_state import NodeStateLock
        with NodeStateLock(self.root/'transactions'):
            return self._install(chain)

    def _install(self,chain):
        head=verify_chain(chain);current=self.chain()
        if current:
            common=min(len(current),len(chain))-1
            if current[common]['release_id']!=chain[common]['release_id']:raise ValueError('Public memory cannot fork')
            if len(chain)<len(current):return current[-1]
        previous_language=None
        for row in chain:
            document=self.document(row)
            language=document['sources']['accepted_language']['chain']
            if previous_language:
                if len(language)<len(previous_language) or language[len(previous_language)-1]!=previous_language[-1]:
                    raise ValueError('Public memory cannot regress or fork its accepted-language source')
            previous_language=language
        atomic_json(self.path,{'chain':chain});self.stamp=None;return head
