"""Public language-family dependencies: signed adapter, immutable upstream base.

The descriptor contains names from a closed upstream allowlist, never commands,
paths supplied by peers, personal checkpoints or conversation-derived data.
"""
import hashlib
import json
from pathlib import Path

from src.learning.public_bases import BASES,base_hash
from .manifest import ReleaseManifest,TrustPolicy,ReleaseError
from .language_catalog import INITIAL_BUNDLES,EVALUATION_BYTES

SIGNER="53cea6456aced3a5a31ad24505aaa47d88e1827d1e7976592c91b54fc0e8980f"
ADAPTERS={"qwen3-0.6b":"3ff34a673bc4d8d83c0ba33286acfd2a1ee318c8ffbafce16caba345beb0b67f",
          "qwen3-1.7b":"badef4394af180ab75ca25f0d9d47f2406b92809b53b5845ab65d451cf9c4b3c"}
SIZES={"qwen3-0.6b":{"model.safetensors":1503300328},
       "qwen3-1.7b":{"model-00001-of-00002.safetensors":3441185608,
                     "model-00002-of-00002.safetensors":622329984,"model.safetensors.index.json":25605}}
COMMON_SIZES={"config.json":726,"generation_config.json":239,"tokenizer.json":11422654,
              "tokenizer_config.json":9732,"LICENSE":11343}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def validate_descriptor(value):
    if not isinstance(value,dict) or set(value)!={"version","base_name","base_hash","files","adapter_id","adapter_bytes","evaluation_id"}:
        raise ReleaseError("Invalid language dependency descriptor")
    name=value["base_name"]
    if type(value["version"]) is not int or value["version"]!=1 or name not in BASES or value["base_hash"]!=base_hash(name):
        raise ReleaseError("Unknown pinned language base")
    if value["adapter_id"]!=ADAPTERS[name] or value["adapter_bytes"]!= (4599712 if name=="qwen3-0.6b" else 6434736):
        raise ReleaseError("Unapproved public language adapter")
    files=value["files"]
    if not isinstance(files,list) or len(files)!=len(BASES[name]):raise ReleaseError("Missing language dependencies")
    seen=set();sizes={**COMMON_SIZES,**SIZES[name]}
    for row in files:
        if not isinstance(row,dict) or set(row)!={"name","artifact_id","bytes"}:raise ReleaseError("Invalid dependency fields")
        filename=row["name"]
        if filename not in BASES[name] or filename in seen or row["artifact_id"]!=BASES[name][filename] or type(row["bytes"]) is not int or row["bytes"]!=sizes[filename]:
            raise ReleaseError("Language dependency differs from the immutable upstream pin")
        seen.add(filename)
    identifier=value["evaluation_id"]
    if not isinstance(identifier,str) or len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):
        raise ReleaseError("Invalid public evaluation identity")
    if len(canonical(value))>16*1024:raise ReleaseError("Language dependencies exceed their bound")
    return value


def verify_bundle(bundle):
    if not isinstance(bundle,dict) or set(bundle)!={"release","descriptor"}:raise ReleaseError("Invalid language bundle")
    descriptor=validate_descriptor(bundle["descriptor"])
    release=ReleaseManifest.from_dict(bundle["release"])
    if release.sequence!=0 or release.model_format!="safetensors-v1" or release.model_artifact_id!=descriptor["adapter_id"] or release.config_artifact_id!=hashlib.sha256(canonical(descriptor)).hexdigest() or release.tokenizer_artifact_id!=BASES[descriptor["base_name"]]["tokenizer.json"]:
        raise ReleaseError("Language release does not bind its dependencies")
    TrustPolicy(trusted_signers=(SIGNER,),pinned_genesis_release_id=INITIAL_BUNDLES[descriptor["base_name"]]["release"]["release_id"]).authorize(release)
    return release,descriptor


def dependencies(bundle):
    release,descriptor=verify_bundle(bundle)
    return [(row["artifact_id"],row["bytes"]) for row in descriptor["files"]]+[
        (release.model_artifact_id,descriptor["adapter_bytes"]),(release.config_artifact_id,len(canonical(descriptor))),
        (descriptor["evaluation_id"],EVALUATION_BYTES[descriptor["base_name"]])]


def load_bundle(directory,name):
    if name not in BASES:raise ReleaseError("Unknown language family")
    path=Path(directory)/"bundles"/f"{name}.json"
    if path.stat().st_size>64*1024:raise ReleaseError("Language bundle exceeds its bound")
    bundle=json.loads(path.read_text());verify_bundle(bundle)
    return bundle
