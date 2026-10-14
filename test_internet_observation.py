import json
from experiments.observe_internet import Continuity


IDS=["a"*64,"b"*64,"c"*64]


def status(now,sequence):
    return {"data_fresh":True,"nodes":[{"node_id":identity,"active":True,"stale":False,
        "signature_verified":True,"state":"connected","sequence":sequence,"timestamp_ms":int(now*1000),
        "uptime_s":now,'process_instance_id':identity[:32]} for identity in IDS]}


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


def test_restart_between_healthy_samples_restarts_the_window_and_then_recovers():
    observer=Continuity(IDS,60);observer.observe(status(1000,1),1000)
    reset=status(1020,2);reset['nodes'][0]['uptime_s']=5
    result=observer.observe(reset,1020)
    assert not result['healthy'] and result['consecutive_healthy_seconds']==0
    assert result['nodes'][0]['uptime_reset_or_clock_jump']
    for now,uptime in [(1040,25),(1100,85)]:
        value=status(now,3 if now==1040 else 4);value['nodes'][0]['uptime_s']=uptime
        result=observer.observe(value,now)
    assert result['healthy'] and result['satisfied'] and result['consecutive_healthy_seconds']==60


def test_a_restart_is_detected_even_when_new_uptime_exceeds_previous_uptime():
    observer=Continuity(IDS,60)
    first=status(1000,1);first['nodes'][0]['uptime_s']=2
    observer.observe(first,1000)
    later=status(1020,2);later['nodes'][0]['uptime_s']=10
    result=observer.observe(later,1020)
    assert not result['healthy'] and result['nodes'][0]['uptime_reset_or_clock_jump']


def test_missing_boolean_negative_or_nonfinite_uptime_cannot_certify_continuity():
    for value in [None,True,-1,float('nan'),float('inf'),10**400]:
        observer=Continuity(IDS,60);data=status(1000,1);data['nodes'][0]['uptime_s']=value
        result=observer.observe(data,1000)
        assert not result['healthy'] and not result['satisfied']
        json.dumps(result,allow_nan=False)


def test_process_nonce_change_breaks_continuity_even_if_uptime_and_counters_look_continuous():
    observer=Continuity(IDS,60);observer.observe(status(1000,1),1000)
    value=status(1020,2);value['nodes'][0]['process_instance_id']='d'*32
    result=observer.observe(value,1020)
    assert not result['healthy'] and result['consecutive_healthy_seconds']==0
    assert result['nodes'][0]['uptime_reset_or_clock_jump']
    missing=status(1040,3);missing['nodes'][0].pop('process_instance_id')
    assert not observer.observe(missing,1040)['healthy']
