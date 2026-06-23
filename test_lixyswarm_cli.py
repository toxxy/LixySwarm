import json

from lixyswarm_cli import main
from src.contribution import ContributionPolicy


def test_cli_init_persists_explicit_policy(tmp_path, capsys):
    assert main([
        "--home", str(tmp_path), "init", "--mode", "balanced", "--yes"
    ]) == 0
    policy = ContributionPolicy.load(tmp_path / "contribution.json")
    assert policy.mode == "balanced"
    assert policy.consented_at > 0
    assert "Contribution policy saved" in capsys.readouterr().out


def test_cli_status_is_machine_readable(tmp_path, capsys):
    ContributionPolicy.for_mode("relay").save(tmp_path / "contribution.json")
    assert main(["--home", str(tmp_path), "status"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["profile"]["mode"] == "relay"
    assert output["known_peers"] == 0


def test_cli_missing_policy_is_connectivity_only(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "status"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["profile"]["work"]["connectivity"]
    assert not output["profile"]["work"]["training"]


def test_cli_artifacts_never_publish_source_paths(tmp_path, capsys):
    ContributionPolicy.for_mode("relay").save(tmp_path / "contribution.json")
    source = tmp_path / "personal-filename.bin"
    source.write_bytes(b"public artifact content")
    assert main([
        "--home", str(tmp_path), "artifact-add", str(source), "--kind", "model"
    ]) == 0
    added = json.loads(capsys.readouterr().out)
    assert "personal-filename" not in json.dumps(added)
    assert main(["--home", str(tmp_path), "artifact-list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == [added]


def test_cli_threshold_release_workflow(tmp_path, capsys):
    ContributionPolicy.for_mode("relay").save(tmp_path / "contribution.json")
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"release model")
    assert main([
        "--home", str(tmp_path), "artifact-add", str(model_path),
        "--kind", "model",
    ]) == 0
    model = json.loads(capsys.readouterr().out)

    signer_ids = []
    keys = []
    for index in range(2):
        key = tmp_path / f"release-key-{index}.pem"
        keys.append(key)
        assert main(["--home", str(tmp_path), "release-keygen", str(key)]) == 0
        signer_ids.append(json.loads(capsys.readouterr().out)["signer_id"])
    trust_args = [
        "--home", str(tmp_path), "trust-init", "--threshold", "2"
    ]
    for signer_id in signer_ids:
        trust_args.extend(["--signer", signer_id])
    assert main(trust_args) == 0
    capsys.readouterr()

    manifest = tmp_path / "release.json"
    assert main([
        "--home", str(tmp_path), "release-create",
        "--model-id", model["artifact_id"],
        "--model-format", "pytorch-weights-only-v1",
        "--sequence", "0", "--output", str(manifest),
    ]) == 0
    capsys.readouterr()
    for key in keys:
        assert main([
            "--home", str(tmp_path), "release-sign", str(manifest),
            "--key", str(key),
        ]) == 0
        capsys.readouterr()
    assert main([
        "--home", str(tmp_path), "release-accept", str(manifest), "--activate"
    ]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["activated"]
    assert main(["--home", str(tmp_path), "release-status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["active_release_id"] == accepted["release_id"]
