"""Simple participant UI. No model is loaded by the interface."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import sys

from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from src.contribution.policy import ContributionPolicy
from src.network.client_state import ClientSettings, state_directory
from desktop.backend import NodeController

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

    def __init__(self, directory, parent):
        super().__init__(parent); self.directory = directory

    def run(self):
        from desktop.personal_store import PersonalStore
        try:
            store = PersonalStore(self.directory)
            store.load()
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


class SwarmWindow(QMainWindow):
    def __init__(self, controller=None, *, initialize_personal=True):
        super().__init__()
        self.controller = controller or NodeController(state_directory())
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
        explorer = QPushButton("Ver la red en el Explorer")
        explorer.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("http://31.97.9.54:8080/swarm-explorer.html")))
        self.state_tab.addWidget(explorer); self.state_tab.addStretch()
        self.installation = None
        self.install_button = QPushButton("Instalar en este equipo")
        self.install_button.setVisible(bool(getattr(sys, "frozen", False)))
        self.install_button.clicked.connect(self.install_app); self.state_tab.addWidget(self.install_button)
        self.install_result = QLabel(); self.install_result.setWordWrap(True); self.state_tab.addWidget(self.install_result)

        chat = self._page("Chat")
        text = QLabel("El chat utilizará conocimiento compartido por otros participantes. En esta versión todavía no hay un modelo de red preparado.")
        text.setWordWrap(True); chat.addWidget(text)
        self.chat_input = QTextEdit(); self.chat_input.setPlaceholderText("Aquí podrás preguntar al enjambre.")
        self.chat_input.setEnabled(False); chat.addWidget(self.chat_input)
        self.chat_send = QPushButton("Esperando modelo de la red"); self.chat_send.setEnabled(False); chat.addWidget(self.chat_send)

        resources = self._page("Recursos"); form = QFormLayout()
        self.profile = QComboBox(); self.profile.addItems(["Equilibrado", "Máximo", "Solo con equipo disponible", "Solo conectividad"])
        self.profile.currentIndexChanged.connect(self.select_profile); form.addRow("Cómo colaborar", self.profile)
        self.cpu = QSpinBox(); self.cpu.setRange(0, 100); self.cpu.setSuffix(" %"); form.addRow("Presupuesto de CPU", self.cpu)
        self.gpu = QSpinBox(); self.gpu.setRange(0, 100); self.gpu.setSuffix(" %"); form.addRow("Presupuesto de GPU", self.gpu)
        self.ram = QDoubleSpinBox(); self.ram.setRange(0.1, 1024); self.ram.setSuffix(" GB"); form.addRow("RAM para colaborar", self.ram)
        self.disk = QDoubleSpinBox(); self.disk.setRange(0, 10000); self.disk.setSuffix(" GB"); form.addRow("Almacenamiento compartido", self.disk)
        self.bandwidth = QDoubleSpinBox(); self.bandwidth.setRange(0, 10000); self.bandwidth.setSuffix(" Mbps"); form.addRow("Presupuesto de conexión", self.bandwidth)
        self.battery = QCheckBox("Pausar cálculo cuando esté usando batería"); form.addRow(self.battery)
        resources.addLayout(form)
        note = QLabel("Estos permisos quedan guardados en tu equipo. El cálculo comienza cuando hay trabajo compatible y un motor preparado.")
        note.setWordWrap(True); note.setObjectName("muted"); resources.addWidget(note)
        save = QPushButton("Guardar límites"); save.clicked.connect(self.save_resources); resources.addWidget(save)
        self.applied = QLabel(); resources.addWidget(self.applied); resources.addStretch()

        activity = self._page("Actividad")
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
            self.personal_preparation.start()
        else:
            self.personal_label.setText("El contenido personal permanece en tu equipo")

    def _page(self, name):
        page = QWidget(); layout = QVBoxLayout(page); layout.setSpacing(16); self.tabs.addTab(page, name)
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
        policy = ContributionPolicy.for_mode("maximum" if index == 1 else "relay" if index == 3 else "balanced")
        policy.only_when_idle = index == 2
        if policy.only_when_idle:
            policy.pause_above_cpu_percent = 35
        policy.cpu_percent = self.cpu.value(); policy.gpu_percent = self.gpu.value()
        policy.max_ram_gb = self.ram.value(); policy.max_disk_gb = self.disk.value()
        policy.max_bandwidth_mbps = self.bandwidth.value(); policy.pause_on_battery = self.battery.isChecked()
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

    def pause_node(self):
        self._action(lambda: self.controller.pause(not self.controller.settings().paused))

    def disconnect_node(self):
        self._action(self.controller.disconnect)

    def save_resources(self):
        self._action(lambda: self.controller.save_policy(self.chosen_policy()))

    def toggle_knowledge(self):
        self._action(lambda: self.controller.share_knowledge(not self.controller.knowledge_settings().enabled))

    def refresh(self):
        snapshot = self.controller.snapshot(); status = snapshot["status"]
        diagnostic = snapshot["diagnostics"]
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

    def install_app(self):
        self.install_button.setEnabled(False)
        self.install_result.setText("Instalando la app para tu usuario…")
        self.installation = Installation(self)
        self.installation.finished_install.connect(self.install_result.setText)
        self.installation.start()

    def closeEvent(self, event):
        if self.installation is not None and self.installation.isRunning():
            event.ignore(); return
        if self.personal_preparation is not None and self.personal_preparation.isRunning():
            self.personal_preparation.requestInterruption()
            self.personal_preparation.wait(2000)
            if self.personal_preparation.isRunning():
                event.ignore(); return
        event.accept()


def main():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("LixySwarm"); app.setOrganizationName("LixySwarm")
    window = SwarmWindow(); window.show()
    raise SystemExit(app.exec())
