import json
from dataclasses import replace
from pathlib import Path
import pytest

from src.network.lsp import LSPIdentity
from src.network.report_privacy import redact_report_routes
from src.network.work_protocol import WorkResult,ResultReceipt


def test_nested_routes_omit_addresses_without_changing_verified_receipts():
    worker=LSPIdentity.generate();requester=LSPIdentity.generate()
    work=WorkResult(job_id='a'*64,status='ok',output={'answer':'public fixture'})
    receipt=ResultReceipt.create(work,worker,requester.node_id_hex)
    work=replace(work,receipt=receipt.to_dict())
    raw={'source_result':work.to_dict(),'routes':[{'transport':'relay','encrypted':True,'host':'192.0.2.1','relay_host':'8.8.8.8','node_id':worker.node_id_hex}]}
    public=redact_report_routes(raw)
    assert public['routes']==[{'transport':'relay','encrypted':True,'public_internet':True,'node_id':worker.node_id_hex}]
    restored=WorkResult.from_dict(public['source_result'])
    assert receipt.verify(restored,expected_worker_id=worker.node_id_hex,expected_requester_id=requester.node_id_hex)
    assert public['source_result']==raw['source_result'] and raw['routes'][0]['host']=='192.0.2.1'


def test_signed_routes_are_rejected_instead_of_invalidating_the_signature():
    raw={'signature':'opaque signature','route':{'transport':'direct','encrypted':True,'host':'8.8.8.8'}}
    with pytest.raises(ValueError,match='signed route'):redact_report_routes(raw)
    offer={'job_id':'a'*64,'operation':'public.probe.v1','origin_node_id':'b'*64,'payload':raw['route']}
    with pytest.raises(ValueError,match='signed route'):redact_report_routes(offer)


def test_current_public_reports_have_no_observed_route_addresses():
    def check(value):
        if isinstance(value,list):
            for item in value:check(item)
        elif isinstance(value,dict):
            if 'transport' in value and 'encrypted' in value:
                # Older reports may identify the explicitly public relay;
                # participant addresses must never be published here.
                assert 'host' not in value
            for item in value.values():check(item)
    for path in (Path(__file__).parent/'experiments/reports').glob('*.json'):check(json.loads(path.read_text()))
