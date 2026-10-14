"""Local controller; settings never pass through an external web API."""
from __future__ import annotations
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from client_daemon import node_command
from src.network.client_state import ClientSettings
from src.network.telemetry import verify_status


class NodeController:
    def __init__(self, directory: Path, *, port=7338, allow_private=False):
        self.directory = directory
        self.port = port
        self.allow_private = allow_private
        self.process = None

    def _read(self, name):
        try:
            path = self.directory / name
            if path.stat().st_size > 256 * 1024:
                return {}
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}

    def settings(self):
        return ClientSettings.load(self.directory)

    def snapshot(self):
        diagnostics = self._read("diagnostics.json")
        envelope = self._read("status.json")
        try:
            status = verify_status(envelope)
        except (ValueError, TypeError):
            status = {}
        pid = diagnostics.get("pid")
        running = False
        if isinstance(pid, int) and pid > 0:
            try:
                process = psutil.Process(pid)
                command = process.cmdline()
                # PID reuse must not turn an unrelated process into our node.
                running = process.is_running() and str(self.directory.resolve()) in command and (
                    "--node" in command or any(Path(arg).name == "client_daemon.py" for arg in command)
                )
            except (psutil.Error, OSError):
                pass
        elif self.process is not None:
            running = self.process.poll() is None
        recent = bool(status) and abs(time.time() - status["timestamp_ms"] / 1000) <= 60
        return {"status": status, "diagnostics": diagnostics,
                "running": bool(running), "fresh": recent and running}

    def start(self, settings: ClientSettings):
        settings.connected = True
        settings.save(self.directory)
        if self.snapshot()["running"]:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        output = (self.directory / "client.log").open("ab")
        try:
            kwargs = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            }
            self.process = subprocess.Popen(node_command(self.directory, port=self.port, allow_private=self.allow_private),
                                            stdin=subprocess.DEVNULL, stdout=output, stderr=output, **kwargs)
        finally:
            output.close()

    def pause(self, paused: bool):
        settings = self.settings()
        settings.paused = bool(paused)
        settings.save(self.directory)

    def disconnect(self):
        settings = self.settings()
        settings.connected = False
        settings.save(self.directory)

    def save_policy(self, policy):
        settings = self.settings()
        settings.policy = policy
        settings.save(self.directory)

    def confirmed_settings(self):
        snapshot = self.snapshot()
        return snapshot["fresh"] and snapshot["diagnostics"].get("settings") == asdict(self.settings())

    def log_tail(self):
        try:
            with (self.directory / "client.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 16000))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return "El nodo guardará aquí su actividad cuando se conecte."
