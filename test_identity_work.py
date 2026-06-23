from src.network.identity_work import (
    load_or_mine_identity_work,
    verify_identity_work,
)
from src.network.lsp import LSPIdentity


def test_identity_work_is_bound_to_key_and_persists(tmp_path):
    identity = LSPIdentity.generate()
    path = tmp_path / "identity-work.json"
    proof = load_or_mine_identity_work(
        path, identity.node_id_hex, bits=12
    )
    assert verify_identity_work(
        identity.node_id_hex, proof, minimum_bits=12
    )
    assert load_or_mine_identity_work(
        path, identity.node_id_hex, bits=12
    ) == proof
    assert not verify_identity_work(
        LSPIdentity.generate().node_id_hex, proof, minimum_bits=12
    )
    assert identity.node_id_hex in path.read_text()


def test_identity_work_rejects_tampering_and_insufficient_cost(tmp_path):
    identity = LSPIdentity.generate()
    proof = load_or_mine_identity_work(
        tmp_path / "proof.json", identity.node_id_hex, bits=10
    )
    tampered = dict(proof, nonce=proof["nonce"] + 1)
    assert not verify_identity_work(
        identity.node_id_hex, tampered, minimum_bits=10
    )
    assert not verify_identity_work(
        identity.node_id_hex, proof, minimum_bits=11
    )
