"""Acquire signed memory from a replica and use its model over public LSP.

Run on the producer's host during a separately managed service outage. This
probe never stops services, reads personal state, loads tensors or signs data.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from src.contribution import ContributionPolicy,ResourceRequirements
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import PublicKnowledgeService,KnowledgeSettings
from src.network.public_memory import PublicMemoryService,select_cards,augment_messages
from src.network.telemetry import atomic_json
from src.network.work_protocol import ResultReceipt
from src.network.language_compute import LANGUAGE_OPERATION
from src.network.report_privacy import public_route


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--provider',required=True)
    parser.add_argument('--expected-release',required=True)
    parser.add_argument('--expected-model',required=True)
    args=parser.parse_args()
    if (args.directory.exists() or args.directory.resolve().is_relative_to(Path.home()/'Dropbox')
            or any(not re.fullmatch('[0-9a-f]{64}',v) for v in [args.provider,args.expected_release,args.expected_model])):
        parser.error('Use fresh private test storage and exact public content identities')
    def producer_inactive():
        value=subprocess.run(['systemctl','--user','is-active','lixyswarm-node.service'],capture_output=True,text=True)
        return value.stdout.strip() not in {'active','activating','reloading'}
    if not producer_inactive():raise ValueError('This assessment requires the separately stopped local producer service')
    args.directory.mkdir(parents=True,mode=0o700)
    report={'success':False,'scope':'Known physical replica, with producer node service stopped; no physical host power-off claim',
        'producer_inactive_at_start':True,'replica_physical_id':args.provider,'expected_memory_release_id':args.expected_release,
        'requester_is_additional_physical_worker':False,'private_inputs':False,'new_model_trained':False,'started_at':time.time()}
    identity=LSPIdentity.generate()
    node=LSPNodeV3(identity,host='0.0.0.0',port=0,seeds=get_seed_endpoints(),address_book_path=args.directory/'peers.json',allow_private=False)
    settings=ClientSettings(policy=ContributionPolicy.for_mode('relay'))
    knowledge=PublicKnowledgeService(node,args.directory,{'cpu_cores':1,'ram_gb':1,'has_gpu':False,'gpu_vram_gb':0})
    knowledge.update(settings,KnowledgeSettings())
    memory=PublicMemoryService(knowledge,args.directory);memory.update(settings)
    try:
        node.start();until=time.monotonic()+60
        while args.provider not in node.connected_node_ids:
            if time.monotonic()>until:raise TimeoutError('Public replica discovery expired')
            node.connect_node(args.provider,timeout=10)
        route=next(p for p in node.peers() if p['node_id']==args.provider)
        if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
            raise ValueError('Replica must use encrypted public Internet transport')
        head=memory.acquire(args.provider,threading.Event())
        if head['release_id']!=args.expected_release:raise ValueError('Replica has another memory version')
        document=memory.registry.document()
        if document['sources']['accepted_language']['chain'][-1]['release']['model_artifact_id']!=args.expected_model:
            raise ValueError('Replica memory describes another accepted model')
        question='According to the public catalog, what is the accepted qwen3-1.7b adapter hash? Reply only with the hash.'
        cards=select_cards(document,question)
        messages,records=augment_messages([{'role':'user','content':question}],cards)
        result=knowledge.work.submit(LANGUAGE_OPERATION,{'model_artifact_id':args.expected_model,'messages':messages},
            ResourceRequirements(kind='inference',gpu_required=True,ram_gb=5),peer_id=args.provider,timeout_s=90)
        if (result.status!='ok' or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=args.provider,expected_requester_id=identity.node_id_hex)
                or result.output.get('model_id')!=args.expected_model
                or not re.fullmatch(re.escape(args.expected_model)+r'[.!]?',result.output.get('answer','').strip(),flags=re.I)):
            raise ValueError('Replica did not answer from the exact accepted public knowledge')
        if not producer_inactive():raise ValueError('Producer recovered before the outage assessment ended')
        assert 'torch' not in __import__('sys').modules
        report.update(success=True,producer_inactive_at_end=True,head=head,documents=memory.registry.chain(),
            content_bytes_received=memory.registry.store.total_bytes(),public_records=len(document['cards']),record_ids=records,
            signed_result=result.to_dict(),requester_id=identity.node_id_hex,
            route=public_route({k:route.get(k) for k in ['transport','host','relay_host','encrypted']}),finished_at=time.time(),weights_fetched=False)
    finally:
        atomic_json(args.directory/'report.json',report)
        memory.close();knowledge.close();node.stop()
    print(json.dumps({'success':report['success'],'replica':args.provider,'public_records':report.get('public_records')}))


if __name__=='__main__':main()
