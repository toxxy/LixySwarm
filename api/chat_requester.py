"""Transport-only requester; no models, personal memory or public telemetry."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from src.contribution import ContributionPolicy
from src.network.bootstrap import get_seed_endpoints
from src.network.chat_service import ChatBroker,LocalChatAPI
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.language_compute import LanguageComputeService
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import PublicKnowledgeService,KnowledgeSettings
from src.network.telemetry import atomic_json,build_status,sign_status,verify_status
from src.version import VERSION


def run(directory):
    directory.mkdir(mode=0o700,parents=True,exist_ok=True);directory.chmod(0o700)
    with NodeStateLock(directory):
        identity=LSPIdentity.load(str(directory/'identity.key'))
        if identity is None:identity=LSPIdentity.generate();identity.save(str(directory/'identity.key'))
        policy=ContributionPolicy.for_mode('relay');policy.max_ram_gb=.5;policy.max_disk_gb=.0625;policy.max_bandwidth_mbps=1
        settings=ClientSettings(policy=policy);settings.save(directory)
        resources={'cpu_cores':1,'ram_gb':.5,'gpu_vram_gb':0.,'has_gpu':False}
        node=LSPNodeV3(identity,host='0.0.0.0',port=0,seeds=get_seed_endpoints(),
            address_book_path=directory/'peers.json',allow_private=False,
            capabilities={'peer_exchange':True,'training':False,'inference':False,'worker_ready':False},
            resource_profile={**resources,'mode':'relay','worker_ready':False})
        knowledge=PublicKnowledgeService(node,directory,resources)
        language=LanguageComputeService(knowledge,directory,resources)
        chat=ChatBroker(node,knowledge.work,language);api=LocalChatAPI(chat,directory)
        stop=threading.Event()
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.set())
        sequence=0;status_path=directory/'status.json'
        if status_path.exists():sequence=verify_status(json.loads(status_path.read_text()))['sequence']
        started=time.monotonic()
        try:
            if 'torch' in sys.modules:raise RuntimeError('A web requester must not load a tensor engine')
            node.start();api.start()
            while not stop.is_set():
                knowledge.update(settings,KnowledgeSettings(enabled=False));language.update(settings);chat.update(settings)
                sequence+=1
                status=build_status(node,sequence=sequence,role='participant',state='connected' if node.peer_count else 'connecting',
                    uptime_s=time.monotonic()-started,resources=resources)
                # Only owner IPC reads this status; never send it as a new PC.
                atomic_json(status_path,sign_status(identity,status))
                atomic_json(directory/'diagnostics.json',{'pid':os.getpid(),'runtime_version':VERSION,
                    'timestamp_ms':status['timestamp_ms'],'chat':chat.snapshot(),'requester_only':True,
                    'public_telemetry_sent':False,'tensor_engine_loaded':False})
                stop.wait(1)
        finally:api.close();knowledge.close();node.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',action='store_true',required=True)
    parser.add_argument('--state-dir',type=Path,required=True)
    run(parser.parse_args().state_dir)


if __name__=='__main__':main()
