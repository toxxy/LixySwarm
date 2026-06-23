"""
lsp_relay.py — Relay de chat via LSP al nodo local.

Cuando el VPS recibe POST /chat, no carga el modelo (sin GPU).
En cambio, abre una conexión LSP al nodo local y reenvía el mensaje.
El nodo local genera la respuesta y la devuelve.

Protocolo:
  VPS → LSP GOSSIP (kind="chat_request") → nodo local
  nodo local → LSP GOSSIP (kind="chat_response") → VPS
  VPS → respuesta al frontend
"""

import json
import time
import socket
import struct
import threading
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("lixy.relay")

# Config — en producción viene de env vars o config file
LOCAL_NODE_HOST = None   # Configure explicitly; never commit an operator address.
LOCAL_NODE_PORT = 7338   # TCP gossip port
RELAY_TIMEOUT   = 30     # segundos máximo para esperar respuesta


class LSPRelay:
    """
    Relay ligero que envía mensajes de chat al nodo local via TCP LSP.
    No requiere torch ni cargar el modelo en el VPS.
    """

    def __init__(self, local_host: Optional[str] = None, local_port: int = LOCAL_NODE_PORT):
        self.local_host = local_host
        self.local_port = local_port
        self._available = local_host is not None

    @property
    def available(self) -> bool:
        return self._available and self.local_host is not None

    def chat(self, message: str) -> dict:
        """
        Envía mensaje al nodo local y espera respuesta.
        Returns dict con 'response', 'agent_used', 'status'.
        """
        if not self.available:
            return {
                "response": None,
                "agent_used": None,
                "status": "relay_unavailable",
                "detail": "Nodo local no configurado. El chat requiere conexión directa al nodo con GPU.",
            }

        try:
            return self._send_via_tcp(message)
        except ConnectionRefusedError:
            return {"response": None, "status": "relay_offline",
                    "detail": f"Nodo local {self.local_host}:{self.local_port} no accesible"}
        except TimeoutError:
            return {"response": None, "status": "relay_timeout",
                    "detail": f"Timeout esperando respuesta del nodo local ({RELAY_TIMEOUT}s)"}
        except Exception as e:
            return {"response": None, "status": f"relay_error: {e}"}

    def _send_via_tcp(self, message: str) -> dict:
        """Envía chat_request via TCP y espera chat_response."""
        request = {
            "kind": "chat_request",
            "node_id": "vps-relay",
            "timestamp": time.time(),
            "payload": {"message": message, "source": "vps-frontend"},
        }
        data = json.dumps(request).encode("utf-8")
        framed = struct.pack("!I", len(data)) + data

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(RELAY_TIMEOUT)
            s.connect((self.local_host, self.local_port))
            s.sendall(framed)

            # Leer respuesta (length-prefixed)
            raw_len = s.recv(4)
            if len(raw_len) < 4:
                raise ConnectionError("Respuesta truncada del nodo local")
            length = struct.unpack("!I", raw_len)[0]
            raw_body = b""
            while len(raw_body) < length:
                chunk = s.recv(min(4096, length - len(raw_body)))
                if not chunk:
                    break
                raw_body += chunk

        resp = json.loads(raw_body.decode("utf-8"))
        payload = resp.get("payload", {})
        return {
            "response": payload.get("response", "Sin respuesta del nodo local"),
            "agent_used": payload.get("agent_used"),
            "status": "ok" if payload.get("response") else "empty",
        }

    def ping(self) -> bool:
        """Verifica si el nodo local está accesible."""
        if not self.available:
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((self.local_host, self.local_port))
            return True
        except Exception:
            return False


# Instancia global — se configura con la IP del nodo local cuando esté disponible
_relay = LSPRelay(local_host=LOCAL_NODE_HOST)


def get_relay() -> LSPRelay:
    return _relay


def configure_relay(host: str, port: int = LOCAL_NODE_PORT):
    """Configura el relay con la IP del nodo local."""
    global _relay
    _relay = LSPRelay(local_host=host, local_port=port)
    log.info(f"LSP Relay configurado → {host}:{port}")
