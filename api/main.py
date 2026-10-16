"""
main.py — FastAPI backend para LixySwarm
Expone estado del enjambre (solo lectura) y endpoint de chat.
Puerto: 8080
"""

import hmac
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

PUBLISH_TOKEN_ENV = "LIXYSWARM_PUBLISH_TOKEN"


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


def _publish_auth_error(request: Request):
    expected = os.environ.get(PUBLISH_TOKEN_ENV, "")
    if not expected:
        return _err(f"{PUBLISH_TOKEN_ENV} no está configurado en la API", 503)

    received = request.headers.get("x-lixyswarm-token", "")
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        received = auth_header[7:].strip()

    if not received or not hmac.compare_digest(received, expected):
        return _err("Token de publicación inválido", 401)
    return None


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


# ── Publish endpoint ──────────────────────────────────────────────────────────

@app.post("/swarm/publish")
async def swarm_publish(request: Request):
    auth_error = _publish_auth_error(request)
    if auth_error:
        return auth_error

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            return _err("El estado publicado debe ser un objeto JSON", 400)

        publisher_ip = request.client.host if request.client else None
        saved = state.save_published_status(payload, publisher_ip=publisher_ip)
        return _ok({
            "status": "saved",
            "source": saved.get("source"),
            "received_at": saved.get("_received_at"),
            "path": str(state.STATUS_FILE),
        })
    except Exception as e:
        return _err(str(e), 400)


# ── Chat endpoints ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4096)


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = chat_handler.chat(req.message)
        if result.get('status')=='invalid':return _err('La pregunta excede su límite o está vacía',400)
        if result.get('status')=='busy':return _err('Hay una consulta en curso; vuelve a intentar en un momento',429)
        if result.get('status')!='ok' or not isinstance(result.get('response'),str) or not result['response'].strip():
            return _err('No hay un experto de lenguaje disponible en este momento',503)
        return _ok({
            "response": result.get("response"),
            "agent_used": result.get("agent_used"),
            "status": result.get("status"),
            "model_id":result.get('model_id'),
            "receipt":result.get('receipt'),
        })
    except Exception as e:
        return _err('La consulta no pudo completarse',503)


@app.get("/chat/history")
def chat_history():
    # Never disclose one visitor's conversation to another visitor.
    return _ok([])


@app.get('/chat/status')
def chat_status():
    return _ok(chat_handler.status())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=False)
