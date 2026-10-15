"""Local controller; settings never pass through an external web API."""
from __future__ import annotations
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import asdict
from pathlib import Path

import psutil

from client_daemon import node_command
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.telemetry import verify_status
from src.network.public_knowledge import KnowledgeSettings
from src.network.learning_coordinator import LearningSettings
from src.network.language_learning import LanguageLearningSettings
from src.network.engine_catalog import cpu_engine
from src.network.engine_download import prepare_engine


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

    def learning_channel(self):
        return 'language_learning' if (self.directory/'language-learning-controls.json').exists() else 'learning'

    def learning_settings(self):
        kind=LanguageLearningSettings if self.learning_channel()=='language_learning' else LearningSettings
        try: return kind.load(self.directory)
        except (OSError,ValueError,TypeError): return kind()

    def start_learning(self):
        settings=self.learning_settings()
        completed=self.snapshot()["diagnostics"].get(self.learning_channel(),{}).get("round") or 0
        if completed>=10: raise ValueError("La ventana inicial de diez rondas terminó")
        if isinstance(settings,LanguageLearningSettings):
            diagnostic=self.snapshot()['diagnostics']
            if diagnostic.get('language_learning',{}).get('required_training_policy_version')==2:
                if not diagnostic.get('language_training',{}).get('batching_enabled'):
                    raise ValueError('Este equipo necesita habilitar el aprendizaje por lotes antes de continuar')
                settings.training_policy_version=2
            settings.generation+=1
        settings.cycles=completed+1; settings.enabled=True; settings.save(self.directory)

    def stop_learning(self):
        settings=self.learning_settings(); settings.enabled=False; settings.save(self.directory)

    def prepare_cpu_engine(self, *, progress=lambda *_:None, cancel_event=None):
        with NodeStateLock(self.directory/'preparation-lock'):
            return self._prepare_cpu_engine(progress=progress,cancel_event=cancel_event)

    def _prepare_cpu_engine(self, *, progress=lambda *_:None, cancel_event=None):
        entry=cpu_engine()
        if entry is None: raise ValueError("Aún no hay un motor CPU publicado para esta plataforma")
        settings=self.settings(); policy=settings.policy
        if not settings.connected or settings.paused or policy.mode not in {"balanced","maximum"} or policy.cpu_percent<=0 or policy.max_cpu_jobs<=0 or policy.max_ram_gb<2:
            raise ValueError("Elige recursos de CPU y al menos 2 GB de RAM antes de preparar el motor")
        from src.network.public_knowledge import managed_public_bytes
        required=entry.bytes+2*entry.installed_bytes
        if managed_public_bytes(self.directory)+required>policy.max_disk_gb*1024**3:
            raise ValueError("El permiso de almacenamiento no alcanza para preparar este motor")
        import shutil
        if shutil.disk_usage(self.directory).free<required:
            raise ValueError("No hay suficiente espacio libre para preparar el motor")
        permission=asdict(policy)
        def allowed():
            current=self.settings()
            return current.connected and not current.paused and asdict(current.policy)==permission
        return prepare_engine(entry,self.directory,allowed=allowed,
                              rate_mbps=lambda:self.settings().policy.max_bandwidth_mbps/2,
                              progress=progress,cancel_event=cancel_event)

    def cpu_engine_prepared(self):
        entry=cpu_engine()
        if entry is None: return False
        from src.network.reference_compute import EngineSettings
        try:
            settings=EngineSettings.load(self.directory)
            command=settings.command_for("cpu")
            folder=(self.directory/"engines"/entry.manifest_sha256).resolve()
            return bool(settings.enabled and command and Path(command[0]).is_file()
                        and Path(command[0]).resolve().parent==folder)
        except (OSError,ValueError,TypeError): return False

    def _chat_call(self,path,value=None,timeout=2):
        info=self._read("local-api.json")
        snapshot=self.snapshot()
        if not snapshot["running"] or info.get("pid")!=snapshot["diagnostics"].get("pid") or type(info.get("port")) is not int or not 1<=info["port"]<=65535 or not isinstance(info.get("token"),str) or len(info["token"])!=64:
            raise RuntimeError("Actualiza o conecta el servicio local para usar el chat")
        data=json.dumps(value,ensure_ascii=False).encode() if value is not None else None
        request=urllib.request.Request(f"http://127.0.0.1:{info['port']}{path}",data=data,
            headers={"Authorization":"Bearer "+info["token"],"Content-Type":"application/json"})
        # Bypass proxy configuration for machine-local private IPC.
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request,timeout=timeout) as response:
                raw=response.read(64*1024+1)
        except urllib.error.HTTPError as error:
            raw=error.read(64*1024)
            try:message=json.loads(raw).get("error","Consulta no disponible")
            except (ValueError,TypeError):message="Consulta no disponible"
            raise RuntimeError(message) from None
        if len(raw)>64*1024:raise RuntimeError("La respuesta local supera su límite")
        return json.loads(raw)

    def chat_status(self):
        try:return self._chat_call("/status")
        except (OSError,ValueError,RuntimeError):return {"connected":False,"available":False,"local":False,"remote_providers":0}

    def ask_chat(self,messages,remote_context,request_id):
        return self._chat_call("/ask",{"request_id":request_id,"messages":messages,"remote_context":bool(remote_context)},timeout=75)

    def cancel_chat(self,request_id):
        return self._chat_call("/cancel",{"request_id":request_id})

    def prepare_language(self):
        return self._chat_call('/preparation/start',{},timeout=10)

    def stop_language_preparation(self):
        return self._chat_call('/preparation/stop',{})

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
                knowledge=diagnostics.get("knowledge",{}); compute=diagnostics.get("compute",{}); learning=diagnostics.get(self.learning_channel(),{})
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
