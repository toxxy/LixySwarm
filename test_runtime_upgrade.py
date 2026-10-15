"""Upgrade only the owned node; retain persisted learning and consent."""
import json
from types import SimpleNamespace

import pytest

from desktop.backend import NodeController
from src.version import VERSION


@pytest.mark.parametrize('version,expected', [('0.4.0a18', True), (VERSION, False),
    ('0.4.0a21', False), ('unknown', False)])
def test_update_never_downgrades_a_newer_runtime(tmp_path, version, expected):
    controller = NodeController(tmp_path)
    assert controller.runtime_update_available({'running': True,
        'diagnostics': {'runtime_version': version}}) is expected


def owned_node(tmp_path, monkeypatch, *, managed):
    import desktop.backend as backend
    controller = NodeController(tmp_path)
    monkeypatch.setattr(controller, 'snapshot', lambda: {'running': True,
        'diagnostics': {'runtime_version': '0.4.0a18', 'pid': 1234}})
    events = []
    process = SimpleNamespace(cmdline=lambda: ['LixySwarm', '--node', '--state-dir', str(tmp_path)],
        is_running=lambda: True, terminate=lambda: events.append('terminate'),
        wait=lambda **kw: events.append('wait'))
    monkeypatch.setattr(backend.psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(backend.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='1234\n' if managed else '0\n'))
    monkeypatch.setattr(backend.sys, 'platform', 'linux')
    monkeypatch.setattr(controller, '_launch_node', lambda **kw: events.append('launch'))
    return controller, process, events


@pytest.mark.parametrize('managed', [True, False])
def test_upgrade_preserves_all_existing_controls_and_learning(tmp_path, monkeypatch, managed):
    import client_daemon
    for name, value in [('controls.json', {'paused': True, 'disk': 8}),
                        ('language-learning-state.json', {'round': 5, 'cursor': 96}),
                        ('language-training.json', {'enabled': False})]:
        (tmp_path / name).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    controller, _, events = owned_node(tmp_path, monkeypatch, managed=managed)
    monkeypatch.setattr(client_daemon, 'install_user_service', lambda *a, **kw: events.append(('service', kw)))
    controller.update_runtime()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    if managed:
        assert events == [('service', {'port': 7338, 'executable': None, 'restart': True})]
    else:
        assert events == ['terminate', 'wait', 'launch']


def test_upgrade_refuses_an_unrelated_reused_pid(tmp_path, monkeypatch):
    controller, process, events = owned_node(tmp_path, monkeypatch, managed=False)
    process.cmdline = lambda: ['unrelated', '--state-dir', str(tmp_path)]
    with pytest.raises(RuntimeError, match='proceso'):
        controller.update_runtime()
    assert not events


def test_old_runtime_cannot_silently_prepare_the_previous_catalog(tmp_path, monkeypatch):
    controller, _, _ = owned_node(tmp_path, monkeypatch, managed=False)
    monkeypatch.setattr(controller, '_chat_call', lambda *a, **kw: pytest.fail('Old preparation was called'))
    with pytest.raises(RuntimeError, match='Actualiza'):
        controller.prepare_language()


def test_persistent_upgrade_retargets_installed_binary_and_restarts(tmp_path, monkeypatch):
    import client_daemon
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setattr(client_daemon.sys, 'platform', 'linux')
    monkeypatch.setattr(client_daemon.sys, 'frozen', True, raising=False)
    calls = []
    monkeypatch.setattr(client_daemon.subprocess, 'run', lambda command, **kw: calls.append(command))
    installed = tmp_path / 'new bundle' / 'LixySwarm'
    client_daemon.install_user_service(tmp_path / 'state', executable=installed, restart=True)
    unit = (tmp_path / 'config/systemd/user/lixyswarm-node.service').read_text()
    assert f'"{installed}" "--node" "--state-dir" "{tmp_path / "state"}"' in unit
    assert calls[-1] == ['systemctl', '--user', 'restart', 'lixyswarm-node.service']
    assert not list((tmp_path / 'config/systemd/user').glob('.lixyswarm-node-*'))
