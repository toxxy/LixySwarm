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
from src.network.public_knowledge import KnowledgeSettings
from src.network.learning_coordinator import LearningSettings


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

    def share_knowledge(self, enabled):
        KnowledgeSettings(enabled=bool(enabled), consented_at=time.time() if enabled else 0).save(self.directory)

    def knowledge_settings(self):
        try:
            return KnowledgeSettings.load(self.directory)
        except (OSError, ValueError, TypeError):
            return KnowledgeSettings()

    def learning_settings(self):
        try: return LearningSettings.load(self.directory)
        except (OSError,ValueError,TypeError): return LearningSettings()

    def start_learning(self):
        settings=self.learning_settings()
        completed=self.snapshot()["diagnostics"].get("learning",{}).get("round") or 0
        if completed>=10: raise ValueError("La ventana inicial de diez rondas terminó")
        settings.cycles=completed+1; settings.enabled=True; settings.save(self.directory)

    def stop_learning(self):
        settings=self.learning_settings(); settings.enabled=False; settings.save(self.directory)

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
            snapshot = self.snapshot()
            if snapshot["fresh"]:
                status = snapshot["status"]
                diagnostics=snapshot["diagnostics"]
                knowledge=diagnostics.get("knowledge",{}); compute=diagnostics.get("compute",{}); learning=diagnostics.get("learning",{})
                states={"connected":"Conectado","preparing":"Preparando modelo","working":"Calculando","paused":"Pausado","connecting":"Conectando","disconnected":"Desconectado"}
                lines=[f"Nodo {status['node_id'][:16]}",f"Estado: {states.get(status['state'],status['state'])}",
                    f"Conexiones cifradas: {len(diagnostics.get('peers',[]))}"]
                if knowledge.get("sequence") is not None:
                    lines.append(f"Modelo público conservado: versión {knowledge['sequence']} · {knowledge.get('stored_bytes',0)/1024**2:.1f} MiB")
                lines.append(f"Cálculos completados: {compute.get('completed',0)} · activos: {compute.get('active',0)}")
                if learning.get("round"):
                    lines.append(f"Rondas de aprendizaje terminadas: {learning['round']}")
                if learning.get("accepted_sequence") is not None:
                    lines.append(f"Última versión aceptada: {learning['accepted_sequence']}")
                if learning.get("error"):
                    lines.append(f"Trabajo pendiente de recuperar: {learning['error']}")
                return "\n".join(lines)
            return "El nodo guardará aquí su actividad cuando se conecte."
