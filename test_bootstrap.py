"""Bootstrap defaults and operator override behavior."""

from src.network import bootstrap


def test_public_seed_is_available_without_configuration(monkeypatch):
    monkeypatch.delenv("LIXYSWARM_BOOTSTRAP_SEEDS", raising=False)
    monkeypatch.setattr(bootstrap, "DNS_SEEDS", [])

    assert bootstrap.get_seed_endpoints() == bootstrap.PUBLIC_BOOTSTRAP_SEEDS
    assert bootstrap.get_bootstrap_addresses() == bootstrap.PUBLIC_BOOTSTRAP_SEEDS


def test_configured_seeds_replace_public_defaults(monkeypatch):
    monkeypatch.setenv(
        "LIXYSWARM_BOOTSTRAP_SEEDS",
        "seed-a.example:7444,seed-b.example,seed-a.example:7444",
    )

    assert bootstrap.get_seed_endpoints() == [
        ("seed-a.example", 7444),
        ("seed-b.example", 7338),
    ]
    assert bootstrap.get_bootstrap_addresses() == [
        ("seed-a.example", 7444),
        ("seed-b.example", 7338),
    ]


def test_empty_override_disables_public_bootstrap(monkeypatch):
    monkeypatch.setenv("LIXYSWARM_BOOTSTRAP_SEEDS", "")

    assert bootstrap.get_seed_endpoints() == []
    assert bootstrap.get_bootstrap_addresses() == []


def test_invalid_explicit_override_fails_closed(monkeypatch):
    monkeypatch.setenv("LIXYSWARM_BOOTSTRAP_SEEDS", "seed.example:not-a-port")

    assert bootstrap.get_seed_endpoints() == []
    assert bootstrap.get_bootstrap_addresses() == []
