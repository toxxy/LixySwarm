"""
main.py — FastAPI backend para LixySwarm
Expone estado del enjambre (solo lectura) y endpoint de chat.
Puerto: 8080
"""

from datetime import datetime, timezone
from typing import Any
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import api.swarm_state as state
import api.chat_handler as chat_handler

app = FastAPI(title="LixySwarm API", version="1.0")

# CORS — allow any origin for the Lixy frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history (last 20)
_chat_history: deque = deque(maxlen=20)


def _ok(data: Any) -> dict:
    return {
        "ok": True,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _err(msg: str, status_code: int = 500):
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


# ── Swarm endpoints ───────────────────────────────────────────────────────────

@app.get("/swarm/status")
def swarm_status():
    try:
        return _ok(state.get_full_swarm_status())
    except Exception as e:
        return _err(str(e))


@app.get("/swarm/agents")
def swarm_agents():
    try:
        return _ok(state.get_agents_state())
    except Exception as e:
        return _err(str(e))


@app.get("/swarm/matriarca")
def swarm_matriarca():
    try:
        return _ok(state.get_matriarca_state())
    except Exception as e:
        return _err(str(e))


@app.get("/swarm/dolphin")
def swarm_dolphin():
    try:
        return _ok(state.get_dolphin_state())
    except Exception as e:
        return _err(str(e))


@app.get("/swarm/network")
def swarm_network():
    try:
        return _ok(state.get_network_state())
    except Exception as e:
        return _err(str(e))


@app.get("/swarm/metrics")
def swarm_metrics():
    try:
        return _ok(state.get_metrics())
    except Exception as e:
        return _err(str(e))


# ── Chat endpoints ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = chat_handler.chat(req.message)
        if result["status"] == "loading":
            return JSONResponse(
                status_code=202,
                content=_ok({"status": "loading", "message": "Model is loading, please retry in a moment."}),
            )
        entry = {
            "role_user": req.message,
            "role_assistant": result.get("response"),
            "agent_used": result.get("agent_used"),
            "status": result.get("status"),
        }
        _chat_history.append(entry)
        return _ok({
            "response": result.get("response"),
            "agent_used": result.get("agent_used"),
            "status": result.get("status"),
        })
    except Exception as e:
        return _err(str(e))


@app.get("/chat/history")
def chat_history():
    return _ok(list(_chat_history))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=False)
