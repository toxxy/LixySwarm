"""Concurrent reception must preserve hashes, a contiguous prefix and consent."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random
import threading
import time
from types import SimpleNamespace

import pytest

from src.network.artifact_store import ArtifactStore,ArtifactService,ArtifactError,ARTIFACT_CHUNK_BYTES
from src.network.work_protocol import WorkCoordinator


class Coordinator:
    artifact_request_slot=WorkCoordinator.artifact_request_slot
    max_offers_per_peer=4
    _max_workers=2
    def __init__(self,store):
        self._lock=threading.RLock();self._closed=False;self._artifact_fetch_slots={}
        self.node=SimpleNamespace(capabilities={},peers=lambda:[{'node_id':'peer','capabilities':
            {} if self.limit is None else {'artifact_read_window_v1':self.limit}}])
        self.server=ArtifactService(self,store,register_handlers=False);self.active=0;self.peak=0;self.calls=[]
        self.on_read=lambda payload,cancel:None;self.limit=2
    def submit(self,operation,payload,*a,cancel_event=None,**kw):
        if operation.endswith('describe.v1'):
            output=self.server._describe(payload,None)
            return SimpleNamespace(status='ok',output=output)
        with self._lock:self.active+=1;self.peak=max(self.peak,self.active);self.calls.append(payload['offset'])
        try:
            changed=self.on_read(payload,cancel_event)
            if changed is not None:return changed
            output=self.server._read_encoded_chunk(payload,None) if operation.endswith('.v2') else self.server._read_chunk(payload,None)
            return SimpleNamespace(status='ok',output=output)
        finally:
            with self._lock:self.active-=1


def fixture(tmp_path):
    source=ArtifactStore(tmp_path/'source');content=random.Random(1842).randbytes(3*ARTIFACT_CHUNK_BYTES+5)
    path=tmp_path/'public';path.write_bytes(content);manifest=source.import_file(path,kind='model')
    return source,content,manifest


def test_out_of_order_responses_resume_an_unaligned_prefix_without_gaps(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);second=threading.Event();finished=[]
    def read(payload,cancel):
        if payload['offset']==3:assert second.wait(2)
        elif payload['offset']==3+ARTIFACT_CHUNK_BYTES:second.set()
        finished.append(payload['offset'])
    coordinator.on_read=read;target=ArtifactStore(tmp_path/'target');part,_=target.begin_receive(m);part.write_bytes(content[:3])
    stats=[];path=ArtifactService(coordinator,target,register_handlers=False).fetch(m.artifact_id,peer_id='peer',observer=stats.append)
    assert path.read_bytes()==content and coordinator.peak==2 and finished[0]!=3
    assert [r['offset'] for r in stats]==sorted(r['offset'] for r in stats)
    assert sum(r['bytes'] for r in stats)==len(content)-3


@pytest.mark.parametrize('limit,window',[(None,2),(2,1),(1,2)])
def test_legacy_or_serial_receivers_keep_one_request(limit,window,tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);coordinator.limit=limit
    coordinator.on_read=lambda *_:time.sleep(.005)
    target=ArtifactStore(tmp_path/'target');stats=[]
    ArtifactService(coordinator,target,register_handlers=False,fetch_window=window).fetch(m.artifact_id,peer_id='peer',observer=stats.append)
    assert coordinator.peak==1 and {r['fetch_window'] for r in stats}=={1}


def test_corruption_preserves_only_the_contiguous_verified_prefix_and_retry_finishes(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);committed=threading.Event()
    def bad(payload,cancel):
        if payload['offset']==ARTIFACT_CHUNK_BYTES:
            assert committed.wait(2)
            return SimpleNamespace(status='ok',output={'offset':payload['offset'],'encoding':'raw','data':'AA==','chunk_sha256':'0'*64})
    coordinator.on_read=bad;target=ArtifactStore(tmp_path/'target');client=ArtifactService(coordinator,target,register_handlers=False)
    with pytest.raises(ArtifactError):client.fetch(m.artifact_id,peer_id='peer',progress=lambda done,total:committed.set() if done>=ARTIFACT_CHUNK_BYTES else None)
    partial=target.incoming/(m.artifact_id+'.part')
    assert partial.read_bytes()==content[:ARTIFACT_CHUNK_BYTES] and not target.has(m.artifact_id)
    coordinator.on_read=lambda *_:None
    assert client.fetch(m.artifact_id,peer_id='peer').read_bytes()==content


def test_cancel_stops_prefetched_requests_and_never_writes_a_late_block(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);started=threading.Event();cancel=threading.Event();errors=[]
    def blocked(payload,event):
        with coordinator._lock:
            if coordinator.active==2:started.set()
        assert event.wait(2)
        return SimpleNamespace(status='error',error='work_cancelled')
    coordinator.on_read=blocked;target=ArtifactStore(tmp_path/'target')
    def receive():
        try:ArtifactService(coordinator,target,register_handlers=False).fetch(m.artifact_id,peer_id='peer',cancel_event=cancel)
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=receive);thread.start();assert started.wait(2);cancel.set();thread.join(2)
    assert not thread.is_alive() and len(coordinator.calls)==2 and coordinator.active==0
    assert errors and isinstance(errors[0],ArtifactError) and not target.has(m.artifact_id)
    assert (target.incoming/(m.artifact_id+'.part')).stat().st_size==0


def test_chunk_window_is_shared_across_services_and_namespaces(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);coordinator.on_read=lambda *_:time.sleep(.01)
    clients=[ArtifactService(coordinator,ArtifactStore(tmp_path/str(i)),namespace=name,register_handlers=False)
        for i,name in enumerate(['artifact','language.artifact','public-memory.artifact'])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda client:client.fetch(m.artifact_id,peer_id='peer'),clients))
    assert all(path.read_bytes()==content for path in results) and coordinator.peak==2


def test_same_object_from_two_store_instances_has_only_one_partial_writer(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);coordinator.on_read=lambda *_:time.sleep(.01)
    clients=[ArtifactService(coordinator,ArtifactStore(tmp_path/'shared'),register_handlers=False) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda c:c.fetch(m.artifact_id,peer_id='peer'),clients))
    assert all(path.read_bytes()==content for path in results)
    assert len(coordinator.calls)==4 and len(set(coordinator.calls))==4


def test_worker_timeout_is_not_mistaken_for_a_future_poll_timeout(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);cancel=threading.Event();done=threading.Event();errors=[]
    def timeout(*_):raise TimeoutError('synthetic RPC failure')
    coordinator.on_read=timeout
    def receive():
        try:ArtifactService(coordinator,ArtifactStore(tmp_path/'target'),register_handlers=False).fetch(m.artifact_id,peer_id='peer',cancel_event=cancel)
        except Exception as error:errors.append(error)
        finally:done.set()
    thread=threading.Thread(target=receive);thread.start();quick=done.wait(1)
    if not quick:cancel.set()
    thread.join(2)
    assert quick and isinstance(errors[0],TimeoutError)


def test_received_prefetch_is_charged_even_when_an_earlier_block_fails(tmp_path):
    source,content,m=fixture(tmp_path);coordinator=Coordinator(source);charged=threading.Event();sizes=[]
    class Budget:
        def wait(self,size,**kw):sizes.append(size);charged.set()
    def fail_first(payload,event):
        if payload['offset']==0:
            assert charged.wait(2)
            raise ArtifactError('first block failed after another response arrived')
    coordinator.on_read=fail_first;target=ArtifactStore(tmp_path/'target')
    with pytest.raises(ArtifactError,match='first block failed'):
        ArtifactService(coordinator,target,receive_budget=Budget(),register_handlers=False).fetch(m.artifact_id,peer_id='peer')
    assert sizes==[4*((ARTIFACT_CHUNK_BYTES+2)//3)+2048]
    assert (target.incoming/(m.artifact_id+'.part')).stat().st_size==0


@pytest.mark.parametrize('limit',[True,0,-1,1.5,65,'2'])
def test_bad_provider_window_is_rejected_before_any_body_request(tmp_path,limit):
    source,_,m=fixture(tmp_path);coordinator=Coordinator(source);coordinator.limit=limit
    with pytest.raises(ArtifactError,match='provider artifact receive window'):
        ArtifactService(coordinator,ArtifactStore(tmp_path/'target'),register_handlers=False).fetch(m.artifact_id,peer_id='peer')
    assert not coordinator.calls
