"""Sign only deterministic public registry data and verified public model facts."""
import argparse
import json
from pathlib import Path
import uuid

from src.network.client_state import ClientSettings
from src.network.public_knowledge import KnowledgeSettings,managed_public_bytes
from src.network.lsp import LSPIdentity
from src.release.public_memory import (PublicMemoryRegistry,build_document,language_source,
    canonical,sign_document)


def publish(directory,epoch,key):
    controls=ClientSettings.load(directory)
    if not KnowledgeSettings.load(directory).enabled:raise ValueError('Publishing requires existing public-storage consent')
    registry=PublicMemoryRegistry(directory);chain=registry.chain()
    # No input-text, personal-store or conversation argument exists.
    document=build_document(epoch,language_source(directory));raw=canonical(document)
    if epoch<len(chain):
        if canonical(registry.document(chain[epoch]))!=raw:raise ValueError('This public epoch already has another accepted source')
        return {'manifest':chain[epoch],'already_published':True,'private_inputs':False,'new_model_published':False}
    if epoch!=len(chain):raise ValueError('Publication must append the next public memory epoch')
    if managed_public_bytes(directory)+len(raw)*2+8192>controls.policy.max_disk_gb*1024**3:
        raise ValueError('Public memory publication exceeds local storage consent')
    manifest=sign_document(document,chain,key)
    temporary=registry.store.incoming/(uuid.uuid4().hex+'.json');temporary.write_bytes(raw)
    artifact=registry.store.commit_generated(temporary,kind='dataset',media_type='application/json')
    if artifact.artifact_id!=manifest['content_id']:raise ValueError('Public source changed during publication')
    registry.install(chain+[manifest])
    return {'manifest':manifest,'records':len(document['cards']),'private_inputs':False,'new_model_published':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--publisher-key',type=Path,required=True)
    parser.add_argument('--epoch',type=int,choices=[0,1],required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(repo) for path in [args.directory,args.publisher_key]):
        parser.error('Identity and public node storage must remain outside the checkout')
    key=LSPIdentity.load(str(args.publisher_key))
    if key is None:raise ValueError('Use an existing authorized public publisher')
    print(json.dumps(publish(args.directory,args.epoch,key),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
