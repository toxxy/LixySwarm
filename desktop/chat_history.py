"""Conversation persistence belongs only to the encrypted personal store."""
from __future__ import annotations
from pathlib import Path

from .personal_store import PersonalStore


class ChatHistory:
    def __init__(self,directory:Path,*,backend=None):
        self.store=PersonalStore(directory,backend=backend);self.value=self.store.load()
        history=self.value.get("chat",{"version":1,"messages":[],"share_context":False})
        if not isinstance(history,dict) or set(history)!={"version","messages","share_context"} or history["version"]!=1 or type(history["share_context"]) is not bool or not isinstance(history["messages"],list) or len(history["messages"])>200:
            raise ValueError("El historial personal no tiene un formato válido")
        for row in history["messages"]:
            if not isinstance(row,dict) or set(row)!={"role","content"} or row["role"] not in {"user","assistant"} or not isinstance(row["content"],str) or len(row["content"].encode())>8192:
                raise ValueError("Mensaje personal inválido")
        self.messages=history["messages"];self.share_context=history["share_context"]

    def save(self,messages,share_context):
        history={"version":1,"messages":messages[-200:],"share_context":bool(share_context)}
        # Preserve other personal banks; this adapter owns only the chat key.
        self.value=self.store.load();self.value["chat"]=history;self.store.save(self.value)
        self.messages=history["messages"];self.share_context=history["share_context"]


def bounded_context(messages,current):
    if not isinstance(current,str) or not 1<=len(current.encode())<=4096:raise ValueError("La pregunta debe tener entre 1 y 4096 bytes")
    selected=[];size=len(current.encode())
    for row in reversed(messages[-7:]):
        length=len(row["content"].encode())
        if size+length>4096:break
        selected.append(row);size+=length
    return list(reversed(selected))+[{"role":"user","content":current}]
