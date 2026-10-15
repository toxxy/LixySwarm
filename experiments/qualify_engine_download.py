"""Anonymous public code-package installation in bounded developer scratch.

Uses the consenting physical machine's bandwidth and an 8-GiB scratch bound.
No node identity, model, learned state or extra physical participant is created.
This is engine-only qualification, not a clean end-to-end app installation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import ssl
import urllib.request
import certifi

from src.network.client_state import ClientSettings
from src.network.engine_catalog import language_engine
from src.network.engine_download import prepare_engine
from src.network.engine_package import verify_package,digest_file
from src.network.engine_download import HTTPSRedirect
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from .prepare_public_math import outside_checkout


def run(args,cancel):
    directory=args.directory.resolve();physical=args.physical_node_directory.resolve()
    if not outside_checkout(directory) or directory.is_relative_to(physical) or physical.is_relative_to(directory):
        raise ValueError('Use separate external developer scratch and existing physical controls')
    entry=language_engine('cuda')
    if entry is None:raise ValueError('No pinned CUDA package for this machine')
    protocol=json.loads(json.dumps(asdict(entry)))
    if directory.exists():
        if not args.resume or json.loads((directory/'entry.json').read_text())!=protocol:
            raise ValueError('Resume requires the same pinned code package and explicit resume flag')
    else:
        directory.mkdir(mode=0o700,parents=True);atomic_json(directory/'entry.json',protocol)
    initial=ClientSettings.load(physical);policy=asdict(initial.policy)
    limit=min(8*1024**3,int(initial.policy.max_disk_gb*1024**3));peak=0;last=0.;last_phase=None
    began=time.monotonic()
    def sample(force=False):
        nonlocal peak,last
        if force or time.monotonic()-last>=2:
            current=managed_public_bytes(directory);peak=max(peak,current);last=time.monotonic()
            if current>limit:raise ValueError('Code-package scratch exceeded its physical-policy bound')
        return peak
    def allowed():
        now=ClientSettings.load(physical);sample()
        return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and now.policy.max_bandwidth_mbps>0)
    def progress(phase,done,total):
        nonlocal last_phase
        changed=phase!=last_phase;sample(force=changed or done==total);last_phase=phase
        atomic_json(directory/'status.json',{'phase':phase,'done':done,'total':total,'peak_managed_bytes':peak,'success':False,'timestamp':time.time()})
    try:
        if not allowed():raise ValueError('Physical permissions do not allow this code download')
        connection=urllib.request.build_opener(HTTPSRedirect(),urllib.request.HTTPSHandler(
            context=ssl.create_default_context(cafile=certifi.where()))).open
        def opener(request,**kwargs):
            response=connection(request,**kwargs)
            with (directory/'http-ranges.jsonl').open('a') as stream:
                stream.write(json.dumps({'timestamp':time.time(),'request_range':request.headers.get('Range'),
                    'response_status':response.status,'response_range':response.headers.get('Content-Range'),
                    'response_bytes':response.headers.get('Content-Length')})+'\n')
            return response
        command=prepare_engine(entry,directory,allowed=allowed,
            rate_mbps=lambda:ClientSettings.load(physical).policy.max_bandwidth_mbps/2,
            cancel_event=cancel,progress=progress,opener=opener)
        metadata=verify_package(Path(command).parent,entry.manifest_sha256,profile=entry.profile)
        sample(force=True)
        completed=subprocess.run([command,'--self-test'],capture_output=True,timeout=120,check=True)
        result=json.loads(completed.stdout)
        if not result['success'] or not result['frozen']:raise ValueError('Downloaded code package failed its frozen self-test')
        if not allowed():raise ValueError('Physical permissions changed before code qualification completed')
        report={'success':True,'profile':entry.profile,'archive_id':entry.sha256,'manifest_id':entry.manifest_sha256,
            'regular_file_bytes':entry.regular_file_bytes,'logical_installed_bytes':entry.installed_bytes,
            'managed_installed_bytes':managed_public_bytes(directory),'peak_managed_bytes':peak,'scratch_limit_bytes':limit,
            'bandwidth_mbps':initial.policy.max_bandwidth_mbps/2,'self_test':result,'declared_files':len(metadata['files']),
            'anonymous_public_https':True,'model_downloaded':False,'node_identity_created':False,
            'extra_physical_participant':False,'end_to_end_clean_app_proven':False,'seconds':time.monotonic()-began}
        atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
        return report
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024],
            'peak_managed_bytes':peak,'timestamp':time.time()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--physical-node-directory',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    cancel=threading.Event()
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
