"""
chat_handler.py — Handler de chat con lazy loading del LixyOrchestrator.
El modelo solo se carga cuando hay actividad; se descarga tras 5 min de inactividad.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

_orchestrator = None
_last_used: float = 0.0
_lock = threading.Lock()
_loading = False
IDLE_TIMEOUT = 300  # 5 minutes


def _unload_loop():
    """Background thread: unload model after IDLE_TIMEOUT seconds of inactivity."""
    global _orchestrator, _last_used
    while True:
        time.sleep(30)
        with _lock:
            if _orchestrator is not None and (time.time() - _last_used) > IDLE_TIMEOUT:
                print("[chat_handler] Unloading model due to inactivity.")
                try:
                    del _orchestrator
                except Exception:
                    pass
                _orchestrator = None


_unload_thread = threading.Thread(target=_unload_loop, daemon=True)
_unload_thread.start()


def _get_or_load_orchestrator():
    global _orchestrator, _last_used, _loading
    with _lock:
        if _orchestrator is not None:
            _last_used = time.time()
            return _orchestrator, None
        if _loading:
            return None, "loading"
        _loading = True

    # Load outside lock to avoid blocking health checks
    try:
        from lixy_orchestrator import LixyOrchestrator
        orc = LixyOrchestrator()
        with _lock:
            _orchestrator = orc
            _last_used = time.time()
            _loading = False
        return orc, None
    except Exception as e:
        with _lock:
            _loading = False
        return None, f"error: {e}"


def chat(message: str) -> dict:
    """Send a message to the swarm. Returns response dict."""
    orc, status = _get_or_load_orchestrator()
    if orc is None:
        return {
            "response": None,
            "agent_used": None,
            "status": status or "unavailable",
        }

    try:
        result = orc.chat(message)
        # Result may be a string or dict depending on orchestrator version
        if isinstance(result, dict):
            response_text = result.get("response", result.get("text", str(result)))
            agent_used = result.get("agent_used") or result.get("agent") or result.get("active_agent")
        else:
            response_text = str(result)
            agent_used = None

        return {
            "response": response_text,
            "agent_used": agent_used,
            "status": "ok",
        }
    except Exception as e:
        return {
            "response": None,
            "agent_used": None,
            "status": f"error: {e}",
        }


def is_model_loaded() -> bool:
    return _orchestrator is not None
