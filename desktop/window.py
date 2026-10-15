"""Simple participant UI. No model is loaded by the interface."""
from __future__ import annotations
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import sys
import threading
import time
import uuid

from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QProgressBar, QScrollArea,
)

from src.contribution.policy import ContributionPolicy
from src.network.client_state import ClientSettings, state_directory
from desktop.backend import NodeController
from src.network.engine_catalog import cpu_engine,language_engine
from src.network.telemetry import atomic_json
from desktop.chat_history import ChatHistory,bounded_context

STYLE = """
QWidget { background: #101713; color: #e9f1ec; font: 14px 'Noto Sans'; }
QLabel#title { font-size: 27px; font-weight: bold; }
QLabel#status { color: #74e0a2; font-size: 23px; font-weight: bold; }
QLabel#muted { color: #a3b6ab; }
QPushButton { background: #243d2e; border: 1px solid #426b51; border-radius: 7px; padding: 10px 18px; }
QPushButton:hover { background: #31533e; }
QPushButton:disabled { color: #7a8c81; background: #1b2820; }
QTabBar::tab { background: #1b2820; padding: 12px 24px; }
QTabBar::tab:selected { background: #2b4936; color: #8cf5b4; }
QTabWidget::pane { border: 1px solid #304438; padding: 16px; }
QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #18251c; border: 1px solid #405747; border-radius: 4px; padding: 6px; }
"""


class PersonalPreparation(QThread):
    prepared = Signal(str)
    history_ready=Signal(object)

    def __init__(self, directory, parent):
        super().__init__(parent); self.directory = directory

    def run(self):
        from desktop.personal_store import PersonalStore
        try:
            history=ChatHistory(self.directory)
            self.history_ready.emit(history)
            self.prepared.emit("Cifrado personal preparado en este equipo")
        except Exception:
            self.prepared.emit("Memoria personal pendiente: almacén protegido no disponible o bloqueado")


class Installation(QThread):
    finished_install = Signal(str)

    def run(self):
        from desktop.install import bundle_directory, install_user_app
        try:
            install_user_app(bundle_directory())
            self.finished_install.emit("App instalada. Puedes abrir LixySwarm desde tus aplicaciones.")
        except Exception:
            self.finished_install.emit("No se pudo instalar. La app puede seguir usándose desde esta carpeta.")


class EnginePreparation(QThread):
    advanced=Signal(str,int,int)
    completed=Signal(bool,str)

    def __init__(self,controller,parent=None):
        super().__init__(parent); self.controller=controller; self.cancel=threading.Event()

    def run(self):
        try:
            self.controller.prepare_cpu_engine(progress=self.advanced.emit,cancel_event=self.cancel)
            self.completed.emit(True,"Motor CPU preparado; el cálculo respeta tus permisos")
        except Exception as error:
            self.completed.emit(False,str(error))


class RuntimeUpdate(QThread):
    completed = Signal(bool, str)

    def __init__(self, controller, parent=None):
        super().__init__(parent); self.controller = controller

    def run(self):
        try:
            self.controller.update_runtime()
            self.completed.emit(True, 'Servicio actualizado; reconectando con tus permisos y conocimiento conservados.')
        except Exception as error:
            self.completed.emit(False, str(error))


class ChatRequest(QThread):
    completed=Signal(bool,object)

    def __init__(self,controller,messages,remote_context,parent=None):
        super().__init__(parent);self.controller=controller;self.messages=messages;self.remote_context=remote_context
        self.request_id=uuid.uuid4().hex

    def run(self):
        try:self.completed.emit(True,self.controller.ask_chat(self.messages,self.remote_context,self.request_id))
        except Exception as error:self.completed.emit(False,str(error))


class SwarmWindow(QMainWindow):
    def __init__(self, controller=None, *, initialize_personal=True, auto_prepare=True):
        super().__init__()
        self.controller = controller or NodeController(state_directory())
        self.auto_prepare=auto_prepare; self.engine_preparation=None; self.engine_phase=""
        self.preparation_requested=False
        self.chat_worker=None;self.chat_pending=False;self.chat_messages=[];self.chat_history=None;self.chat_last_result=None;self.close_pending=False
        self.chat_pending_memories=0;self.chat_last_memories=0
        self.setWindowTitle("LixySwarm")
        self.resize(900, 760)
        self.setMinimumSize(720, 680)
        self.setStyleSheet(STYLE)
        root = QWidget(); layout = QVBoxLayout(root); layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("LixySwarm"); title.setObjectName("title"); layout.addWidget(title)
        subtitle = QLabel("Tu equipo es una hormiga. Tú decides cuánto aporta.")
        subtitle.setObjectName("muted"); layout.addWidget(subtitle)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs)
        self.state_tab = self._page("Estado")
        self.status = QLabel("Listo para conectar"); self.status.setObjectName("status"); self.state_tab.addWidget(self.status)
        self.detail = QLabel(); self.detail.setWordWrap(True); self.state_tab.addWidget(self.detail)
        self.identity = QLabel(); self.state_tab.addWidget(self.identity)
        self.routes = QLabel(); self.state_tab.addWidget(self.routes)
        self.resources_label = QLabel(); self.state_tab.addWidget(self.resources_label)
        self.personal_label = QLabel("Memoria personal: preparando almacenamiento protegido")
        self.personal_label.setWordWrap(True); self.personal_label.setObjectName("muted"); self.state_tab.addWidget(self.personal_label)
        self.stage = QLabel("Puedes conservar el conocimiento público entre equipos. El chat y los procesos de cálculo están en preparación.")
        self.stage.setWordWrap(True); self.stage.setObjectName("muted"); self.state_tab.addWidget(self.stage)
        self.knowledge_label = QLabel(); self.knowledge_label.setWordWrap(True); self.state_tab.addWidget(self.knowledge_label)
        self.knowledge_button = QPushButton("Conservar conocimiento público")
        self.knowledge_button.clicked.connect(self.toggle_knowledge); self.state_tab.addWidget(self.knowledge_button)
        actions = QHBoxLayout()
        self.connect_button = QPushButton("Conectar"); self.connect_button.clicked.connect(self.connect_node); actions.addWidget(self.connect_button)
        self.pause_button = QPushButton("Pausar aporte"); self.pause_button.clicked.connect(self.pause_node); actions.addWidget(self.pause_button)
        self.disconnect_button = QPushButton("Desconectar"); self.disconnect_button.clicked.connect(self.disconnect_node); actions.addWidget(self.disconnect_button)
        self.state_tab.addLayout(actions)
        self.runtime_update = None
        self.update_button = QPushButton('Actualizar servicio')
        self.update_button.clicked.connect(self.update_runtime)
        self.state_tab.addWidget(self.update_button)
        self.update_result = QLabel(); self.update_result.setWordWrap(True); self.state_tab.addWidget(self.update_result)
        explorer = QPushButton("Ver la red en el Explorer")
        explorer.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("http://31.97.9.54:8080/swarm-explorer.html")))
        self.state_tab.addWidget(explorer); self.state_tab.addStretch()
        self.installation = None
        self.install_button = QPushButton("Instalar en este equipo")
        self.install_button.setVisible(bool(getattr(sys, "frozen", False)))
        self.install_button.clicked.connect(self.install_app); self.state_tab.addWidget(self.install_button)
        self.install_result = QLabel(); self.install_result.setWordWrap(True); self.state_tab.addWidget(self.install_result)

        chat = self._page("Chat")
        self.chat_status=QLabel("Esperando un experto de lenguaje disponible")
        self.chat_status.setWordWrap(True);chat.addWidget(self.chat_status)
        self.chat_transcript=QTextEdit();self.chat_transcript.setReadOnly(True);chat.addWidget(self.chat_transcript)
        self.chat_privacy=QLabel("El historial queda cifrado si el almacén protegido está disponible. La pregunta y la memoria pública que actives pueden enviarse al experto.")
        self.chat_privacy.setWordWrap(True);chat.addWidget(self.chat_privacy)
        memory_actions=QHBoxLayout()
        self.chat_memory=QCheckBox('Recuperar recuerdos de esta conversación')
        self.chat_memory.setEnabled(False);self.chat_memory.toggled.connect(self.save_chat_history)
        memory_actions.addWidget(self.chat_memory)
        self.chat_forget=QPushButton('Vaciar recuerdos');self.chat_forget.setEnabled(False)
        self.chat_forget.setToolTip('Vacía la memoria local y la desactiva. Conserva el historial visible.')
        self.chat_forget.clicked.connect(self.forget_chat_memory);memory_actions.addWidget(self.chat_forget)
        chat.addLayout(memory_actions)
        self.chat_public_memory=QCheckBox('Usar memoria pública verificada de la red')
        self.chat_public_memory.setChecked(self.controller.public_memory_settings().enabled)
        self.chat_public_memory.toggled.connect(lambda enabled:self._action(lambda:self.controller.set_public_memory(enabled)))
        chat.addWidget(self.chat_public_memory)
        self.chat_context=QCheckBox("Enviar también contexto y recuerdos personales al experto de la red")
        self.chat_context.toggled.connect(self.save_chat_history);chat.addWidget(self.chat_context)
        self.chat_input = QTextEdit(); self.chat_input.setPlaceholderText("Pregunta al enjambre…")
        self.chat_input.setMaximumHeight(110);chat.addWidget(self.chat_input)
        chat_actions=QHBoxLayout()
        self.chat_send=QPushButton("Preguntar");self.chat_send.clicked.connect(self.send_chat);chat_actions.addWidget(self.chat_send)
        self.chat_cancel=QPushButton("Cancelar consulta");self.chat_cancel.clicked.connect(self.cancel_chat);self.chat_cancel.setEnabled(False);chat_actions.addWidget(self.chat_cancel)
        chat.addLayout(chat_actions)

        resources = self._page("Recursos",scrollable=True); form = QFormLayout()
        self.profile = QComboBox(); self.profile.addItems(["Equilibrado", "Máximo", "Solo con equipo disponible", "Solo conectividad"])
        self.profile.currentIndexChanged.connect(self.select_profile); form.addRow("Cómo colaborar", self.profile)
        self.cpu = QSpinBox(); self.cpu.setRange(0, 100); self.cpu.setSuffix(" %"); form.addRow("Presupuesto de CPU", self.cpu)
        self.gpu = QSpinBox(); self.gpu.setRange(0, 100); self.gpu.setSuffix(" %"); form.addRow("Presupuesto de GPU", self.gpu)
        self.ram = QDoubleSpinBox(); self.ram.setRange(0.1, 1024); self.ram.setSuffix(" GB"); form.addRow("RAM para colaborar", self.ram)
        self.disk = QDoubleSpinBox(); self.disk.setRange(0, 10000); self.disk.setSuffix(" GB"); form.addRow("Almacenamiento compartido", self.disk)
        self.bandwidth = QDoubleSpinBox(); self.bandwidth.setRange(0, 10000); self.bandwidth.setSuffix(" Mbps"); form.addRow("Presupuesto de conexión", self.bandwidth)
        self.battery = QCheckBox("Pausar cálculo cuando esté usando batería"); form.addRow(self.battery)
        resources.addLayout(form)
        note = QLabel("Al autorizar CPU se preparan el motor y el modelo público para colaborar. El cálculo respeta tus límites. La preparación reserva la mitad del presupuesto de conexión.")
        note.setWordWrap(True); note.setObjectName("muted"); resources.addWidget(note)
        save = QPushButton("Guardar límites"); save.clicked.connect(self.save_resources); resources.addWidget(save)
        self.applied = QLabel(); resources.addWidget(self.applied)
        self.engine_label=QLabel("Motor CPU: pendiente de preparación")
        self.engine_label.setWordWrap(True); resources.addWidget(self.engine_label)
        self.engine_progress=QProgressBar(); self.engine_progress.setRange(0,100); self.engine_progress.setVisible(False)
        resources.addWidget(self.engine_progress)
        self.engine_button=QPushButton("Preparar motor CPU")
        self.engine_button.clicked.connect(self.prepare_engine); resources.addWidget(self.engine_button)
        self.language_label=QLabel('Puedes usar el chat de la red sin cargar un modelo en este equipo.')
        self.language_label.setWordWrap(True);resources.addWidget(self.language_label)
        self.language_progress=QProgressBar();self.language_progress.setVisible(False);resources.addWidget(self.language_progress)
        language_actions=QHBoxLayout()
        self.language_button=QPushButton('Preparar experto de lenguaje')
        self.language_button.clicked.connect(lambda:self._action(self.controller.prepare_language));language_actions.addWidget(self.language_button)
        self.language_stop=QPushButton('Detener preparación')
        self.language_stop.clicked.connect(lambda:self._action(self.controller.stop_language_preparation));language_actions.addWidget(self.language_stop)
        resources.addLayout(language_actions);resources.addStretch()

        activity = self._page("Actividad")
        self.learning_label=QLabel("Aprendizaje distribuido: preparando coordinación")
        self.learning_label.setWordWrap(True); activity.addWidget(self.learning_label)
        self.follow_label=QLabel('');self.follow_label.setWordWrap(True);self.follow_label.hide();activity.addWidget(self.follow_label)
        learning_actions=QHBoxLayout()
        self.learning_button=QPushButton("Aprender una ronda")
        self.learning_button.clicked.connect(lambda:self._action(self.controller.start_learning))
        learning_actions.addWidget(self.learning_button)
        self.learning_stop=QPushButton("Detener aprendizaje")
        self.learning_stop.clicked.connect(lambda:self._action(self.controller.stop_learning))
        learning_actions.addWidget(self.learning_stop); activity.addLayout(learning_actions)
        self.activity = QTextEdit(); self.activity.setReadOnly(True); activity.addWidget(self.activity)
        self.setCentralWidget(root)
        self.select_profile(0)
        self._load_resources()
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(1000)
        self.refresh()
        self.personal_preparation = None
        if initialize_personal:
            self.personal_preparation = PersonalPreparation(self.controller.directory, self)
            self.personal_preparation.prepared.connect(self.personal_label.setText)
            self.personal_preparation.history_ready.connect(self.history_ready)
            self.personal_preparation.start()
        else:
            self.personal_label.setText("El contenido personal permanece en tu equipo")
            self.chat_privacy.setText("Esta prueba usa una conversación temporal en memoria")

    def _page(self, name,scrollable=False):
        page = QWidget(); layout = QVBoxLayout(page); layout.setSpacing(16)
        if scrollable:
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page);self.tabs.addTab(scroll,name)
        else:self.tabs.addTab(page, name)
        return layout

    def select_profile(self, index):
        policy = ContributionPolicy.for_mode("maximum" if index == 1 else "relay" if index == 3 else "balanced")
        if index == 2:
            policy.cpu_percent = 15; policy.gpu_percent = 15
        for field, value in ((self.cpu, policy.cpu_percent), (self.gpu, policy.gpu_percent),
                             (self.ram, policy.max_ram_gb), (self.disk, policy.max_disk_gb),
                             (self.bandwidth, policy.max_bandwidth_mbps)):
            field.setValue(value)
        self.battery.setChecked(policy.pause_on_battery)

    def chosen_policy(self):
        index = self.profile.currentIndex()
        mode="maximum" if index==1 else "relay" if index==3 else "balanced"
        current=self.controller.settings().policy
        changed=mode!=current.mode or (index==2)!=current.only_when_idle
        policy=ContributionPolicy.for_mode(mode) if changed else deepcopy(current)
        policy.only_when_idle = index == 2
        if policy.only_when_idle:
            policy.pause_above_cpu_percent = 35
        policy.cpu_percent = self.cpu.value(); policy.gpu_percent = self.gpu.value()
        policy.max_ram_gb = self.ram.value(); policy.max_disk_gb = self.disk.value()
        policy.max_bandwidth_mbps = self.bandwidth.value(); policy.pause_on_battery = self.battery.isChecked()
        if changed and policy.max_gpu_jobs and policy.gpu_percent:
            resources=self.controller.resources()
            available=resources.get('gpu_vram_gb',0)*policy.gpu_percent/100
            # A Large public training task needs 6 GiB. Splitting an 8-GiB
            # allocation into two inference-sized slots prevents collaboration.
            policy.max_gpu_jobs=min(policy.max_gpu_jobs,max(1,int(available//6)))
        if index == 3:
            policy.gpu_percent = 0
        policy.validate()
        return policy

    def _load_resources(self):
        try:
            policy = self.controller.settings().policy
            self.profile.blockSignals(True)
            self.profile.setCurrentIndex(3 if policy.mode == "relay" else 1 if policy.mode == "maximum" else 2 if policy.only_when_idle else 0)
            self.profile.blockSignals(False)
            for field, value in ((self.cpu, policy.cpu_percent), (self.gpu, policy.gpu_percent),
                                 (self.ram, policy.max_ram_gb), (self.disk, policy.max_disk_gb),
                                 (self.bandwidth, policy.max_bandwidth_mbps)):
                field.setValue(value)
            self.battery.setChecked(policy.pause_on_battery)
        except (ValueError, TypeError, OSError):
            self.applied.setText("Revisa y guarda los permisos antes de conectar.")

    def _action(self, function):
        try:
            function(); self.refresh()
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            QMessageBox.warning(self, "No se pudo completar", str(exc))

    def connect_node(self):
        def action():
            settings = self.controller.settings()
            settings.policy = self.chosen_policy(); settings.paused = False
            self.controller.start(settings)
        self._action(action)
        self.preparation_requested=True
        self.maybe_prepare_engine()

    def pause_node(self):
        self._action(lambda: self.controller.pause(not self.controller.settings().paused))

    def update_runtime(self):
        if self.runtime_update is not None and self.runtime_update.isRunning(): return
        self.update_result.setText('Actualizando el servicio; la conexión se interrumpirá brevemente.')
        self.runtime_update = RuntimeUpdate(self.controller, self)
        self.runtime_update.completed.connect(lambda ok, message: self.update_result.setText(message))
        self.runtime_update.start()

    def disconnect_node(self):
        self._action(self.controller.disconnect)

    def save_resources(self):
        self._action(lambda: self.controller.save_policy(self.chosen_policy()))
        self.preparation_requested=True
        self.maybe_prepare_engine()

    def maybe_prepare_engine(self):
        if not self.auto_prepare:return
        if self.controller.runtime_update_available(): return
        try: settings=self.controller.settings()
        except (OSError,ValueError,TypeError): return
        if not self.controller.snapshot()['fresh']:return
        self.preparation_requested=False
        target=self.controller.automatic_preparation_target()
        if target is None:return
        if not (self.controller.directory/"public-knowledge.json").exists():
            self.controller.share_knowledge(True)
        if target=='language':
            if not self.controller.knowledge_settings().enabled:return
            if self.controller.snapshot()['diagnostics'].get('knowledge',{}).get('sharing') is not True:
                self.preparation_requested=True;return
            self.maybe_prepare_language();return
        policy=settings.policy
        if self.auto_prepare and settings.connected and not settings.paused and policy.mode in {"balanced","maximum"} and policy.cpu_percent>0 and policy.max_ram_gb>=2 and not self.controller.cpu_engine_prepared() and cpu_engine() is not None:
            self.prepare_engine()
        self.maybe_prepare_language()

    def maybe_prepare_language(self):
        if self.controller.runtime_update_available(): return
        if not self.auto_prepare or self.engine_preparation is not None and self.engine_preparation.isRunning():return
        diagnostic=self.controller.snapshot()['diagnostics']
        preparation=diagnostic.get('language_preparation',{})
        policy=self.controller.settings().policy
        if (diagnostic.get('language',{}).get('installed') or preparation.get('state') not in {None,'idle'}
                or policy.mode not in {'balanced','maximum'} or policy.max_ram_gb<5 or policy.gpu_percent<=0
                or language_engine('cuda') is None or not self.controller.knowledge_settings().enabled):return
        try:self.controller.prepare_language()
        except (OSError,ValueError,RuntimeError):pass

    def toggle_knowledge(self):
        self._action(lambda: self.controller.share_knowledge(not self.controller.knowledge_settings().enabled))

    def history_ready(self,history):
        current=self.chat_messages;self.chat_history=history
        self.chat_messages=list(history.messages)+current
        self.chat_context.blockSignals(True);self.chat_context.setChecked(history.share_context);self.chat_context.blockSignals(False)
        self.chat_memory.blockSignals(True);self.chat_memory.setChecked(history.memory_enabled);self.chat_memory.blockSignals(False)
        self.chat_memory.setEnabled(True);self.chat_forget.setEnabled(True)
        self.render_chat()
        self.chat_privacy.setText("Historial y recuerdos personales cifrados en este equipo. La pregunta y la memoria pública activada pueden enviarse al experto; el contexto personal requiere tu autorización.")
        if current:self.save_chat_history()

    def render_chat(self):
        self.chat_transcript.setPlainText("\n\n".join(("Tú: " if row["role"]=="user" else "Enjambre: ")+row["content"] for row in self.chat_messages))
        self.chat_transcript.verticalScrollBar().setValue(self.chat_transcript.verticalScrollBar().maximum())

    def save_chat_history(self):
        self.chat_messages=self.chat_messages[-200:]
        if self.chat_history is None:
            self.chat_privacy.setText("Historial temporal en memoria: el almacén protegido está pendiente. La pregunta actual puede enviarse a la red.")
            return
        try:self.chat_history.save(self.chat_messages,self.chat_context.isChecked(),memory_enabled=self.chat_memory.isChecked())
        except Exception:
            self.chat_privacy.setText("No se pudo guardar el historial cifrado; la conversación actual sigue en memoria")

    def forget_chat_memory(self):
        if self.chat_history is None:return
        try:
            self.chat_history.forget_memory()
            self.chat_memory.blockSignals(True);self.chat_memory.setChecked(False);self.chat_memory.blockSignals(False)
            self.chat_privacy.setText('Recuerdos vaciados y memoria desactivada. El historial de esta conversación sigue cifrado en este equipo.')
        except Exception:self.chat_privacy.setText('No se pudieron vaciar los recuerdos cifrados')

    def send_chat(self):
        if self.chat_pending or self.chat_worker is not None and self.chat_worker.isRunning():return
        try:
            question=self.chat_input.toPlainText().strip()
            if self.chat_history is not None:
                messages,recalled=self.chat_history.context(self.chat_messages,question,enabled=self.chat_memory.isChecked())
            else:messages,recalled=bounded_context(self.chat_messages,question),[]
            self.chat_pending_memories=len(recalled);self.chat_last_memories=0
            self.chat_messages.append({"role":"user","content":question});self.render_chat();self.save_chat_history()
            self.chat_last_result=None
            self.chat_input.clear();self.chat_status.setText("Consultando al experto…")
            if self.chat_worker is not None:self.chat_worker.deleteLater()
            self.chat_worker=ChatRequest(self.controller,messages,self.chat_context.isChecked(),self)
            self.chat_worker.completed.connect(self.chat_completed);self.chat_worker.finished.connect(self.refresh)
            self.chat_pending=True
            self.chat_worker.start();self.refresh()
        except (ValueError,RuntimeError,OSError) as error:QMessageBox.warning(self,"No se pudo preguntar",str(error))

    def cancel_chat(self):
        if self.chat_worker is not None and self.chat_worker.isRunning():
            self._action(lambda:self.controller.cancel_chat(self.chat_worker.request_id))

    def chat_completed(self,success,result):
        self.chat_pending=False
        if success:
            self.chat_last_result=result
            self.chat_last_memories=self.chat_pending_memories if result['route']=='local' or result['context_sent'] else 0
            self.chat_messages.append({"role":"assistant","content":result["answer"]});self.render_chat();self.save_chat_history()
            self.chat_status.setText(("Respuesta en este equipo" if result["route"]=="local" else "Respuesta de la red")+" · recibo verificado")
            if self.chat_last_memories:self.chat_status.setText(self.chat_status.text()+f' · {self.chat_last_memories} recuerdo(s) recuperado(s)')
            if result.get('public_memory',{}).get('used'):
                self.chat_status.setText(self.chat_status.text()+f" · memoria pública v{result['public_memory']['release']['sequence']}")
        else:self.chat_status.setText(str(result))
        self.chat_cancel.setEnabled(False)
        if self.close_pending:self.close()

    def prepare_engine(self):
        if self.engine_preparation is not None and self.engine_preparation.isRunning(): return
        self.engine_phase="preparing"; self.engine_progress.setValue(0); self.engine_progress.setVisible(True)
        self.engine_preparation=EnginePreparation(self.controller,self)
        self.engine_preparation.advanced.connect(self.engine_advanced)
        self.engine_preparation.completed.connect(self.engine_completed)
        self.engine_preparation.start(); self.refresh()

    def engine_advanced(self,phase,done,total):
        self.engine_phase=phase
        labels={"downloading":"Descargando motor CPU", "verifying":"Verificando descarga",
                "extracting":"Preparando archivos", "installing":"Verificando instalación", "complete":"Motor CPU preparado"}
        self.engine_label.setText(labels.get(phase,"Preparando motor CPU")
                                  +(f" · {done/1024**2:.1f} de {total/1024**2:.1f} MB" if total else ""))
        self.engine_progress.setValue(int(100*done/max(1,total)))
        self.engine_heartbeat()

    def engine_heartbeat(self):
        atomic_json(self.controller.directory/"engine-preparation.json",
                    {"state":self.engine_phase,"updated_at":time.time()})

    def engine_completed(self,success,message):
        self.engine_phase="complete" if success else "stopped"; self.engine_heartbeat()
        self.engine_label.setText(message); self.engine_progress.setVisible(False)
        self.refresh()
        self.maybe_prepare_language()

    def refresh(self):
        preparing=self.engine_preparation is not None and self.engine_preparation.isRunning()
        if preparing: self.engine_heartbeat()
        entry=cpu_engine()
        try: policy=self.controller.settings().policy
        except (OSError,ValueError,TypeError): policy=ClientSettings().policy
        prepared=self.controller.cpu_engine_prepared()
        if prepared and not preparing:
            self.engine_label.setText("Motor CPU preparado; el cálculo respeta tus permisos")
        self.engine_button.setEnabled(not preparing and not prepared and entry is not None
            and policy.mode in {"balanced","maximum"} and policy.cpu_percent>0 and policy.max_ram_gb>=2)
        self.engine_button.setText("Motor CPU preparado" if prepared else "Reanudar preparación" if self.engine_phase=="stopped" else "Preparar motor CPU")
        snapshot = self.controller.snapshot(); status = snapshot["status"]
        diagnostic = snapshot["diagnostics"]
        needs_update = self.controller.runtime_update_available(snapshot)
        updating = self.runtime_update is not None and self.runtime_update.isRunning()
        self.update_button.setVisible(needs_update or updating)
        self.update_button.setEnabled(needs_update and not updating)
        preparation=diagnostic.get('language_preparation',{}) if snapshot['fresh'] else {}
        phase=preparation.get('state','idle')
        language_busy=phase not in {'idle','waiting','stopped','complete'}
        installed=diagnostic.get('language',{}).get('installed',False)
        language_ready=diagnostic.get('language',{}).get('ready',False) if snapshot['fresh'] else False
        repair=diagnostic.get('language_cache_repair',{}) if snapshot['fresh'] else {}
        repairing=repair.get('state') in {'starting','receiving','waiting'}
        progress=repair if repairing else preparation
        if repairing:
            message=repair.get('message') or 'Recuperando los archivos del conocimiento público'
        elif language_busy or phase in {'waiting','stopped'}:
            message=preparation.get('message') or 'Preparación del experto pendiente'
        elif language_ready:
            message='Experto de lenguaje disponible en este equipo'
        elif installed:
            message='Experto configurado; todavía no está disponible dentro de tus permisos o faltan archivos.'
        else:
            message='Puedes usar el chat de la red. Para alojar un experto se necesitan 5 GB de RAM y 4 GB de GPU dentro de tus límites.'
        self.language_label.setText(message)
        self.language_progress.setVisible((language_busy or repairing) and progress.get('total',0)>0)
        self.language_progress.setValue(int(100*progress.get('done',0)/max(1,progress.get('total',0))))
        self.language_button.setEnabled(snapshot['fresh'] and not needs_update and not preparing and not language_busy and not repairing and language_engine('cuda') is not None
            and policy.mode in {'balanced','maximum'} and policy.max_ram_gb>=5 and policy.gpu_percent>0)
        self.language_button.setText('Reanudar preparación del experto' if phase in {'waiting','stopped'} else 'Preparar experto de lenguaje')
        self.language_stop.setEnabled((language_busy or phase=='waiting') and not repairing)
        chat=diagnostic.get("chat",{}) if snapshot["fresh"] else {}
        public_memory=chat.get('public_memory') or {}
        self.chat_public_memory.setText('Usar memoria pública verificada de la red'+
            (f" ({public_memory['records']} registros)" if public_memory.get('available') else ''))
        asking=self.chat_pending or self.chat_worker is not None and self.chat_worker.isRunning()
        self.chat_send.setEnabled(bool(chat.get("available")) and not asking)
        self.chat_cancel.setEnabled(asking)
        self.chat_input.setEnabled(snapshot["running"])
        if not asking and not self.chat_messages:
            self.chat_status.setText("Experto disponible en este equipo" if chat.get("available") and chat.get("local") else "Experto de la red disponible" if chat.get("available") else "Esperando un experto de lenguaje disponible")
            if chat.get('catalog_pending'):
                self.chat_status.setText(chat.get('catalog_error') or 'Verificando la versión de lenguaje disponible…')
        state = status.get("state") if snapshot["fresh"] else "stopped"
        labels = {"stopped":"Listo para conectar", "connecting":"Conectando", "connected":"Conectado a la red", "preparing":"Preparando conocimiento público", "paused":"Aporte pausado", "disconnected":"Desconectado"}
        self.status.setText(labels.get(state, state))
        peers = diagnostic.get("peers", []) if snapshot["fresh"] else []
        self.detail.setText(f"{len(peers)} conexiones cifradas.")
        self.identity.setText("Identidad: " + status.get("node_id", "se crea al conectar")[:16])
        self.routes.setText(f"Rutas: {sum(p.get('transport') == 'direct' for p in peers)} directas · {sum(p.get('transport') == 'relay' for p in peers)} por relay")
        r = status.get("resources", {})
        self.resources_label.setText(f"Tu equipo: {r.get('cpu_cores', '—')} hilos CPU · {r.get('ram_gb', '—')} GB RAM · {r.get('gpu_vram_gb', '—')} GB VRAM")
        knowledge = diagnostic.get("knowledge", {}) if snapshot["fresh"] else {}
        computation=diagnostic.get("compute",{}) if snapshot["fresh"] else {}
        channel=self.controller.learning_channel()
        learning=diagnostic.get(channel,{}) if snapshot["fresh"] else {}
        learning_states={"idle":"Listo para una ronda", "collecting":"Repartiendo y recuperando fragmentos", "evaluating":"Evaluando la versión candidata", "promoting":"Conservando y publicando la mejora", "complete":"Ronda finalizada", "interrupted":"Trabajo conservado; esperando recuperación", "no_improvement":"En pausa: las últimas rondas no mejoraron el modelo", "needs_attention":"En pausa: revisa el error antes de reintentar", "waiting_replicas":"Esperando que los equipos compartan la misma versión", "waiting_capacity":"Esperando recursos disponibles para aprender"}
        learning_states['waiting_policy']='Esperando permiso para continuar el aprendizaje'
        learning_states['waiting_program']='El programa anterior conserva trabajo pendiente'
        self.learning_label.setText(learning_states.get(learning.get("status"),"Preparando coordinación")
            + f" · {learning.get('round') or 0} rondas terminadas"
            + (f" · {learning['historical_rounds']} rondas conservadas de otro programa" if learning.get('historical_rounds') else "")
            + (f" · versión {learning['accepted_sequence']}" if learning.get("accepted_sequence") is not None else ""))
        if learning.get("paused"):
            self.learning_label.setText("Aprendizaje detenido; el trabajo pendiente se conserva.")
        prepared=self.controller.learning_settings()
        learning_ready=diagnostic.get('language_training',{}).get('ready') if channel=='language_learning' else computation.get('ready')
        self.learning_button.setEnabled(bool(learning_ready) and len(prepared.workers)==2 and bool(prepared.signing_key_file)
            and not (prepared.enabled and learning.get("status") in {"collecting","evaluating","promoting"}))
        self.learning_stop.setEnabled(prepared.enabled)
        following=diagnostic.get('language_follow',{}) if snapshot['fresh'] else {}
        follow_states={'checking':'Comprobando el conocimiento compartido','acquiring':'Adquiriendo conocimiento de la red',
            'current':'Conocimiento compartido actualizado','waiting':'Esperando un proveedor de conocimiento','paused':'Seguimiento de conocimiento en pausa'}
        self.follow_label.setVisible(bool(following.get('enabled')))
        self.follow_label.setText(follow_states.get(following.get('state'),'Preparando conocimiento compartido')
            +(f" · versión {following['sequence']}" if following.get('sequence') is not None else ''))
        if chat.get("available"):
            self.stage.setText("Puedes preguntar al experto disponible. Tus permisos controlan el aporte de este equipo.")
        elif computation.get("installed"):
            self.stage.setText("Motor de referencia preparado. Tus permisos controlan el cálculo; el modelo de chat sigue en preparación.")
        else:
            self.stage.setText("Puedes conservar el conocimiento público entre equipos. El motor de cálculo está en preparación.")
        knowledge_state = knowledge.get("state", "disabled")
        messages = {"disabled":"Puedes aportar almacenamiento para conservar versiones aprendidas.",
                    "discovering":"Buscando participantes que conservan el modelo público…",
                    "pending":"Esperando un proveedor compatible del conocimiento público.",
                    "downloading":"Adquiriendo y verificando las versiones del modelo…",
                    "paused":"El aporte de almacenamiento está pausado; las versiones guardadas se conservan.",
                    "available":"Modelo público conservado y disponible para otros participantes."}
        self.knowledge_label.setText(messages.get(knowledge_state, "Preparando conocimiento público")
            + (f" {knowledge.get('stored_bytes',0)/1024**2:.1f} MiB · versión {knowledge.get('sequence')}"
               if knowledge.get("sequence") is not None else ""))
        if computation.get("ready"):
            self.knowledge_label.setText(self.knowledge_label.text()+f" · {computation.get('active',0)} tareas activas · {computation.get('completed',0)} completadas")
        self.knowledge_button.setText("Dejar de compartir conocimiento" if self.controller.knowledge_settings().enabled else "Conservar conocimiento público")
        self.connect_button.setEnabled(state in {"stopped", "disconnected"})
        self.disconnect_button.setEnabled(state not in {"stopped", "disconnected"})
        self.pause_button.setEnabled(state not in {"stopped", "disconnected"})
        self.pause_button.setText("Reanudar aporte" if state == "paused" else "Pausar aporte")
        try:
            confirmed = self.controller.confirmed_settings()
            self.applied.setText("Permisos recibidos por el nodo" if confirmed else "Permisos guardados; se aplicarán al conectar o actualizar el nodo")
        except (ValueError, TypeError, OSError):
            self.applied.setText("La configuración necesita revisión")
        self.activity.setPlainText(self.controller.log_tail())
        if self.preparation_requested and snapshot['fresh'] and not needs_update:
            self.preparation_requested=False
            QTimer.singleShot(0,self.maybe_prepare_engine)

    def install_app(self):
        self.install_button.setEnabled(False)
        self.install_result.setText("Instalando la app para tu usuario…")
        self.installation = Installation(self)
        self.installation.finished_install.connect(self.install_result.setText)
        self.installation.start()

    def closeEvent(self, event):
        if self.runtime_update is not None and self.runtime_update.isRunning():
            event.ignore(); return
        if self.chat_worker is not None and self.chat_worker.isRunning():
            self.close_pending=True;self.cancel_chat();self.chat_worker.wait(100)
            if self.chat_worker.isRunning():event.ignore();return
        if self.engine_preparation is not None and self.engine_preparation.isRunning():
            self.engine_preparation.cancel.set(); self.engine_preparation.wait(100)
            if self.engine_preparation.isRunning(): event.ignore(); return
        if self.installation is not None and self.installation.isRunning():
            event.ignore(); return
        if self.personal_preparation is not None and self.personal_preparation.isRunning():
            self.personal_preparation.requestInterruption()
            self.personal_preparation.wait(2000)
            if self.personal_preparation.isRunning():
                event.ignore(); return
        self.timer.stop()
        event.accept()


def main():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("LixySwarm"); app.setOrganizationName("LixySwarm")
    window = SwarmWindow(); window.show()
    raise SystemExit(app.exec())
