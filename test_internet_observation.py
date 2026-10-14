from experiments.observe_internet import Continuity


IDS=["a"*64,"b"*64,"c"*64]


def status(now,sequence):
    return {"data_fresh":True,"nodes":[{"node_id":identity,"active":True,"stale":False,
        "signature_verified":True,"state":"connected","sequence":sequence,"timestamp_ms":int(now*1000),
        "uptime_s":now} for identity in IDS]}


def test_all_three_physical_nodes_must_remain_fresh_for_the_whole_window():
    observer=Continuity(IDS,60)
    assert not observer.observe(status(1000,1),1000)["satisfied"]
    assert not observer.observe(status(1020,2),1020)["satisfied"]
    value=status(1040,3);value["nodes"][1]["stale"]=True
    assert not observer.observe(value,1040)["healthy"]
    assert observer.observe(status(1060,4),1060)["consecutive_healthy_seconds"]==0
    assert not observer.observe(status(1100,5),1100)["satisfied"]
    assert observer.observe(status(1120,6),1120)["satisfied"]


def test_sampling_gaps_duplicate_identities_unsigned_future_and_replayed_status_do_not_pass():
    for change in ['gap','duplicate','unsigned','future','replay']:
        observer=Continuity(IDS,60);observer.observe(status(1000,2),1000)
        now=1070 if change=='gap' else 1020;value=status(now,3)
        if change=='duplicate':value["nodes"].append(dict(value["nodes"][0]))
        if change=='unsigned':value["nodes"][0]["signature_verified"]=False
        if change=='future':value["nodes"][0]["timestamp_ms"]=int((now+1)*1000)
        if change=='replay':value["nodes"][0].update(sequence=1,timestamp_ms=999000)
        result=observer.observe(value,now)
        assert not result['satisfied'] and result['consecutive_healthy_seconds']==0
        assert result['failures']==1
