"""Publish only the pinned public bases/adapters, with signed dependency lists."""
import argparse
import hashlib
import json
from pathlib import Path

from src.learning.public_bases import BASES,base_hash
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.lsp import LSPIdentity
from src.network.telemetry import atomic_json
from src.release.manifest import ReleaseManifest
from src.release.language_family import ADAPTERS,SIGNER,SIZES,COMMON_SIZES,canonical,verify_bundle,load_bundle,INITIAL_BUNDLES


def publish(root,name,base_directory,adapter_file,report_file,identity):
    if identity.node_id_hex!=SIGNER:raise ValueError("Only the pilot public publisher is configured")
    if name not in BASES:raise ValueError("Unknown public base")
    report=json.loads(report_file.read_text())
    if report.get("private_inputs") is not False or report.get("eligible_for_integration") is not True or report.get("regressions")!=[] or report.get("adapter_sha256")!=ADAPTERS[name] or report.get("base_model_hash")!=base_hash(name):
        raise ValueError("Only the evaluated public adapter is eligible")
    if digest_file(adapter_file)[0]!=ADAPTERS[name]:raise ValueError("Adapter does not match the evaluated public pin")
    sizes={**COMMON_SIZES,**SIZES[name]}
    for filename,identifier in BASES[name].items():
        path=base_directory/filename
        if path.is_symlink() or digest_file(path)!=(identifier,sizes[filename]):raise ValueError("Public upstream dependency mismatch")
    store=ArtifactStore(root/"artifacts",max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    evaluation=store.import_file(report_file,kind="evaluation",media_type="application/json")
    descriptor={"version":1,"base_name":name,"base_hash":base_hash(name),
        "files":[{"name":filename,"artifact_id":identifier,"bytes":sizes[filename]} for filename,identifier in sorted(BASES[name].items())],
        "adapter_id":ADAPTERS[name],"adapter_bytes":adapter_file.stat().st_size,"evaluation_id":evaluation.artifact_id}
    from src.release.language_family import validate_descriptor
    validate_descriptor(descriptor)
    for filename in BASES[name]:store.import_file(base_directory/filename,kind="model" if 'safetensors' in filename else "other")
    store.import_file(adapter_file,kind="model")
    temporary=store.incoming/f"descriptor-{name}.json";temporary.write_bytes(canonical(descriptor))
    configuration=store.commit_generated(temporary,kind="other",media_type="application/json")
    path=root/"bundles"/f"{name}.json"
    if path.exists():
        bundle=load_bundle(root,name)
        if bundle["descriptor"]!=descriptor:raise ValueError("Do not replace an existing genesis family")
    else:
        release=ReleaseManifest.create(model_artifact_id=ADAPTERS[name],model_format="safetensors-v1",sequence=0,
            config_artifact_id=configuration.artifact_id,tokenizer_artifact_id=BASES[name]["tokenizer.json"],minimum_client_version="0.4.0a5",
            created_at=INITIAL_BUNDLES[name]["release"]["created_at"]).sign(identity)
        bundle={"release":release.to_dict(),"descriptor":descriptor};verify_bundle(bundle);atomic_json(path,bundle)
    return bundle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--publisher-key",type=Path,required=True)
    parser.add_argument("--base-name",choices=BASES,required=True)
    parser.add_argument("--base-directory",type=Path,required=True)
    parser.add_argument("--adapter-file",type=Path,required=True)
    parser.add_argument("--report-file",type=Path,required=True)
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    identity=LSPIdentity.load(str(args.publisher_key))
    if identity is None:raise ValueError("Existing publisher identity is required")
    bundle=publish(args.directory,args.base_name,args.base_directory,args.adapter_file,args.report_file,identity)
    print(json.dumps({"base":args.base_name,"release_id":bundle["release"]["release_id"],"adapter_id":bundle["descriptor"]["adapter_id"]}))


if __name__=="__main__":main()
