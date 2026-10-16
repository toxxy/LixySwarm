"""Stateless web chat through the authenticated machine-local LSP requester."""
import os
from pathlib import Path
import threading
import uuid
from desktop.backend import NodeController

_slot=threading.BoundedSemaphore(1)


def chat(message):
    if not isinstance(message,str) or not message.strip() or len(message.encode())>4096:
        return {'response':None,'status':'invalid'}
    if not _slot.acquire(blocking=False):return {'response':None,'status':'busy'}
    try:
        directory=Path(os.environ.get('LIXYSWARM_WEB_CHAT_STATE','/var/lib/lixyswarm-webchat'))
        result=NodeController(directory).ask_chat([{'role':'user','content':message.strip()}],False,uuid.uuid4().hex)
        if not isinstance(result.get('answer'),str) or not result['answer'].strip():
            return {'response':None,'status':'unavailable'}
        return {'response':result['answer'],'agent_used':result['provider_id'],
            'model_id':result['model_id'],'receipt':result['receipt'],'status':'ok'}
    except (OSError,ValueError,RuntimeError,KeyError):
        return {'response':None,'status':'unavailable'}
    finally:_slot.release()


def is_model_loaded():
    return False


def status():
    try:
        directory=Path(os.environ.get('LIXYSWARM_WEB_CHAT_STATE','/var/lib/lixyswarm-webchat'))
        value=NodeController(directory)._chat_call('/status')
        return {'available':bool(value.get('available')),'model_id':value.get('model_id',''),
            'remote_providers':value.get('remote_providers',0)}
    except (OSError,ValueError,RuntimeError):return {'available':False,'model_id':'','remote_providers':0}
