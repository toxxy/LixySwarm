import time
from dataclasses import replace

from src.network.lsp import LSPIdentity
from src.network.work_protocol import ResultReceipt, WorkResult
from src.network.useful_work import UsefulWorkCredit, UsefulWorkLedger


def test_validated_training_credit_is_signed_bound_and_persistent(tmp_path):
    issuer = LSPIdentity.generate()
    worker = LSPIdentity.generate()
    result = WorkResult(
        job_id="01" * 32,
        status="ok",
        output={"gradient_artifact_id": "03" * 32},
        finished_at=time.time(),
    )
    receipt = ResultReceipt.create(result, worker, issuer.node_id_hex).to_dict()
    credit = UsefulWorkCredit.issue(
        issuer,
        worker_node_id=worker.node_id_hex,
        receipt=receipt,
        gradient_artifact_id="03" * 32,
        aggregate_artifact_id="04" * 32,
        model_artifact_id="05" * 32,
        dataset_artifact_id="06" * 32,
        token_count=512,
    )
    assert credit.verify()
    assert not replace(credit, token_count=513).verify()
    path = tmp_path / "credits.json"
    ledger = UsefulWorkLedger(path, worker.node_id_hex)
    assert ledger.add(credit.to_dict())
    assert ledger.add(credit.to_dict())
    assert ledger.summary() == {
        "validated_training_credits": 1,
        "distinct_issuers": 1,
        "validated_tokens": 512,
    }
    assert UsefulWorkLedger(path, worker.node_id_hex).summary()[
        "validated_training_credits"
    ] == 1
    assert not UsefulWorkLedger(
        tmp_path / "other.json", LSPIdentity.generate().node_id_hex
    ).add(credit.to_dict())
    assert not replace(
        credit, receipt={**credit.receipt, "result_digest": "07" * 32}
    ).verify()
