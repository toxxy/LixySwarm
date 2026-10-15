"""Conversation persistence belongs only to the encrypted personal store."""
from __future__ import annotations
from pathlib import Path

from .personal_store import PersonalStore
from src.matriarca.personal_context import PersonalContextMemory, episode_id


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
        self.memory=PersonalContextMemory(self.value.get('personal_context'))
        self.memory_enabled=self.value.get('personal_context_enabled',False)
        if type(self.memory_enabled) is not bool:raise ValueError('La preferencia de memoria personal no es válida')

    def save(self,messages,share_context,*,memory_enabled=None):
        history={"version":1,"messages":messages[-200:],"share_context":bool(share_context)}
        # Preserve other personal banks; this adapter owns only the chat key.
        enabled=self.memory_enabled if memory_enabled is None else memory_enabled
        if type(enabled) is not bool:raise ValueError('La preferencia de memoria personal no es válida')
        self.value=self.store.load()
        memory=PersonalContextMemory(self.value.get('personal_context'))
        if enabled:memory.observe(messages)
        self.value.update(chat=history,personal_context=memory.snapshot(),personal_context_enabled=enabled)
        self.store.save(self.value)
        self.memory=memory;self.memory_enabled=enabled
        self.messages=history["messages"];self.share_context=history["share_context"]

    def context(self,messages,current,*,enabled=None):
        enabled=self.memory_enabled if enabled is None else enabled
        if not enabled:return bounded_context(messages,current),[]
        return recalled_context(messages,current,self.memory)

    def forget_memory(self):
        # Turning the feature off retains encrypted memories; forgetting is
        # an explicit local action and does not erase the visible transcript.
        value=self.store.load();memory=PersonalContextMemory()
        value.update(personal_context=memory.snapshot(),personal_context_enabled=False)
        self.store.save(value)
        self.value=value;self.memory=memory;self.memory_enabled=False


def bounded_context(messages,current):
    if not isinstance(current,str) or not 1<=len(current.encode())<=4096:raise ValueError("La pregunta debe tener entre 1 y 4096 bytes")
    selected=[];size=len(current.encode())
    for row in reversed(messages[-7:]):
        length=len(row["content"].encode())
        if size+length>4096:break
        selected.append(row);size+=length
    return list(reversed(selected))+[{"role":"user","content":current}]


def recalled_context(messages,current,memory):
    """Same 4096-byte/eight-message budget, reserving room for recent turns."""
    baseline=bounded_context(messages,current)
    recent=baseline[:-1][-3:]
    excluded={episode_id([a,b]) for a,b in zip(baseline,baseline[1:]) if a['role']=='user' and b['role']=='assistant'}
    candidates=memory.retrieve(current,exclude=excluded)
    remaining=4096-len(current.encode())-sum(len(row['content'].encode()) for row in recent)
    chosen=[]
    for row in reversed(candidates):
        size=sum(len(m['content'].encode()) for m in row['messages'])
        if size<=remaining:
            chosen.append(row);remaining-=size
    if not chosen:return baseline,[]
    chosen.sort(key=lambda row:row['position'])
    context=[message for row in chosen for message in row['messages']]+recent+[{'role':'user','content':current}]
    return context,[row['episode_id'] for row in chosen]
