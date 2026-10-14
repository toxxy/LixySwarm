"""Observe fresh signed telemetry of explicit physical nodes over public HTTP.

This measures Explorer availability, not model quality or relay redundancy.
No operator addresses, chat contents or private state are recorded.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import signal
import time
import urllib.parse
import urllib.request
from pathlib import Path

from src.network.telemetry import atomic_json


class Continuity:
    def __init__(self,node_ids,required_seconds):
        self.node_ids=set(node_ids);self.required_seconds=required_seconds
        self.last_sample=None;self.healthy_since=None;self.previous={}
        self.samples=0;self.failures=0;self.longest_seconds=0.0

    def observe(self,data,now):
        healthy=isinstance(data,dict) and data.get("data_fresh") is True
        rows={};observed=[]
        for row in data.get("nodes",[]) if isinstance(data,dict) else []:
            if not isinstance(row,dict) or row.get("node_id") not in self.node_ids:continue
            identity=row["node_id"]
            if identity in rows:healthy=False
            rows[identity]=row
        for identity in sorted(self.node_ids):
            row=rows.get(identity,{})
            seq=row.get("sequence");stamp=row.get("timestamp_ms")
            valid=(row.get("active") is True and row.get("stale") is False
                and row.get("signature_verified") is True
                and row.get("state") in {"connected","preparing","working","paused"}
                and type(seq) is int and seq>0 and type(stamp) is int
                and 0<=now-stamp/1000<=60)
            previous=self.previous.get(identity)
            if valid and previous:
                valid=seq>=previous[0] and stamp>=previous[1] and (seq!=previous[0] or stamp==previous[1])
            healthy=healthy and valid
            if valid:self.previous[identity]=(seq,stamp)
            observed.append({"node_id":identity,"healthy":valid,"sequence":seq,
                             "timestamp_ms":stamp,"state":row.get("state"),"uptime_s":row.get("uptime_s")})
        gap=self.last_sample is not None and not 0<=now-self.last_sample<=60
        if gap or not healthy:self.healthy_since=None
        if healthy and self.healthy_since is None:self.healthy_since=now
        seconds=max(0.0,now-self.healthy_since) if self.healthy_since is not None else 0.0
        self.last_sample=now;self.samples+=1;self.failures+=int(not healthy or gap)
        self.longest_seconds=max(self.longest_seconds,seconds)
        return {"timestamp":now,"healthy":healthy,"sampling_gap":gap,"nodes":observed,
                "consecutive_healthy_seconds":round(seconds,2),
                "required_seconds":self.required_seconds,"satisfied":seconds>=self.required_seconds,
                "samples":self.samples,"failures":self.failures,
                "longest_healthy_seconds":round(self.longest_seconds,2)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--endpoint",default="http://31.97.9.54:8080/api/swarm/network")
    parser.add_argument("--node",action="append",required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--required-seconds",type=float,default=86400)
    parser.add_argument("--interval",type=float,default=20)
    parser.add_argument("--source-revision")
    args=parser.parse_args()
    address=urllib.parse.urlsplit(args.endpoint)
    if address.scheme not in {"http","https"} or address.username or address.password:
        parser.error("Observation requires a public endpoint without credentials")
    try:
        if not ipaddress.ip_address(address.hostname).is_global:parser.error("Private/VPN endpoints are not accepted")
    except ValueError:parser.error("Use the public IP of the pilot relay")
    if not 1<=args.interval<=30 or not 1<=args.required_seconds<=7*86400:
        parser.error("Observation interval/duration is outside its bound")
    if len(set(args.node))!=len(args.node) or any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in args.node):
        parser.error("Expected physical identities must be unique SHA-256 IDs")
    if args.source_revision is not None and (len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision)):
        parser.error("Source revision must be a full Git commit hash")
    args.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if any(args.directory.iterdir()):parser.error("Use an empty observation directory; do not replace previous evidence")
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,stopping)
    observer=Continuity(args.node,args.required_seconds);started=time.time();started_mono=time.monotonic()
    result=None;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (args.directory/"samples.jsonl").open("x",encoding="utf-8") as output:
        while not stop:
            sampled=time.monotonic();data={};error=None
            try:
                with opener.open(args.endpoint,timeout=10) as response:
                    raw=response.read(2*1024**2+1)
                    if len(raw)>2*1024**2:raise ValueError("Telemetry response exceeds its bound")
                    value=json.loads(raw)
                    if value.get("ok") is not True:raise ValueError("Unhealthy API")
                    data=value["data"]
            except Exception as exc:error=type(exc).__name__
            result=observer.observe(data,time.time());result["error_type"]=error
            output.write(json.dumps(result,allow_nan=False)+"\n");output.flush()
            atomic_json(args.directory/"summary.json",{
                **result,"started_at":started,"scope":"Fresh signed Explorer telemetry of explicit physical nodes",
                "source_revision":args.source_revision,
                "relay_redundancy_verified":False,"model_quality_verified":False})
            if result["satisfied"] or time.monotonic()-started_mono>2*args.required_seconds:break
            while not stop and time.monotonic()-sampled<args.interval:time.sleep(.25)
    return 0 if result and result["satisfied"] else 1


if __name__=="__main__":raise SystemExit(main())
