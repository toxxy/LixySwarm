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
