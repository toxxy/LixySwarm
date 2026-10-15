"""Authenticated machine-local chat bridge to the physical LSP node.

Questions and answers remain in memory here. The GUI owns encrypted history;
the node never reads personal storage or exports it as training/signals.
"""
from __future__ import annotations
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

from src.contribution import ResourceRequirements
from .language_compute import LANGUAGE_OPERATION,validate_messages
from .telemetry import atomic_json
from .work_protocol import ResultReceipt

MODEL_ID="badef4394af180ab75ca25f0d9d47f2406b92809b53b5845ab65d451cf9c4b3c"
PROVIDERS={"c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255",
           "ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0"}
REQUEST_ID=re.compile(r"^[0-9a-f]{32}$")


class ChatUnavailable(RuntimeError):pass


class ChatBroker:
    def __init__(self,node,work,language,public_memory=None):
        self.node,self.work,self.language=node,work,language
        self.public_memory=public_memory
        self.lock=threading.RLock();self.requests={};self.cancelled={};self.connected=False;self.closed=False
        self.discovery_thread=None;self.last_discovery=0.0
        self.catalog_error=None
        self.settings=None;self.catalog_cancel=threading.Event();self.catalog=None
        if getattr(language,'directory',None) is not None:
            from .chat_versions import ChatVersions
            self.catalog=ChatVersions(Path(language.directory)/'public/chat-language')

    def update(self,settings):
        self.settings=settings
        self.connected=settings.connected and not self.closed
        if not self.connected:
            self.catalog_cancel.set()
            with self.lock:
                for event in self.requests.values():event.set()
        elif hasattr(self.node,"connect_node") and (self.catalog is not None or not self.language.ready and not self.providers()):
            # A fresh requester must not wait behind obsolete test peers in
            # the general connection queue to reach the approved pilot expert.
            interval=15 if self._catalog_pending() or not self.providers() else 60
            if time.monotonic()-self.last_discovery>=interval and (self.discovery_thread is None or not self.discovery_thread.is_alive()):
                self.last_discovery=time.monotonic()
                self.catalog_cancel=threading.Event()
                self.discovery_thread=threading.Thread(target=self._discover,daemon=True,name="lixy-chat-discovery")
                self.discovery_thread.start()

    def _discover(self):
        for provider in sorted(PROVIDERS):
            if not self.connected or self.closed:return
            if provider==self.node.identity.node_id_hex:continue
            try:
                self.node.connect_node(provider,timeout=12)
                if self.catalog is not None and provider in self.node.connected_node_ids:
                    from .public_knowledge import managed_public_bytes
                    def preflight(extra):
                        if (not self.connected or self.closed or managed_public_bytes(self.language.directory)+extra
                                >self.settings.policy.max_disk_gb*1024**3):raise ChatUnavailable('No hay espacio autorizado para verificar versiones de chat')
                    def allowed():
                        return (self.connected and not self.closed and self.settings.policy.mode!='disabled'
                            and self.settings.policy.max_bandwidth_mbps>0 and self.settings.policy.max_disk_gb>0)
                    with self.language.knowledge.metadata_budget(allowed) as budget:
                        self.catalog.synchronize(self.work,provider,cancel_event=self.catalog_cancel,preflight=preflight,
                            receive_budget=budget)
                    self.catalog_error=None
            except Exception:
                self.catalog_error='No se pudo verificar el catálogo de lenguaje'

    def _catalog_pending(self):
        if self.catalog is None:return False
        from src.release.language_versions import LanguageVersionRegistry
        try:
            registry=LanguageVersionRegistry(Path(self.language.directory)/'public/language','qwen3-1.7b')
            chain=registry.history(registry.head()['release']['release_id'])
            known={row['release']['model_artifact_id'] for row in chain+self.catalog.read()['chain']}
            peers=[peer for peer in self.node.peers() if peer['node_id'] in PROVIDERS
                and peer.get('capabilities',{}).get('language_inference_v1')]
            if peers and not self.language.ready and not self.catalog.path.exists():return True
            return any(model not in known for peer in peers for model in
                peer.get('capabilities',{}).get('language_inference_models_v3',peer.get('capabilities',{}).get('language_inference_models',peer.get('resources',{}).get('models',[]))))
        except (OSError,ValueError,TypeError,KeyError):return True

    def _model_id(self):
        directory=getattr(self.language,'directory',None)
        if directory is None:return MODEL_ID
        from src.release.language_versions import LanguageVersionRegistry
        try:
            registry=LanguageVersionRegistry(Path(directory)/'public/language','qwen3-1.7b');local=registry.head()
            remote=self.catalog.head() if self.catalog is not None else local
            if self.catalog is not None:
                local_sequence=local['release']['sequence'];remote_sequence=remote['release']['sequence']
                newer=(registry.history(local['release']['release_id']) if local_sequence>remote_sequence else self.catalog.read()['chain'])
                older=remote if local_sequence>remote_sequence else local
                if newer[older['release']['sequence']]['release']['release_id']!=older['release']['release_id']:
                    raise ValueError('Stored chat and model histories disagree')
            chosen=local if local['release']['sequence']>=remote['release']['sequence'] else remote
            return chosen['release']['model_artifact_id']
        except (OSError,ValueError,TypeError,KeyError):return ''

    def providers(self,model_id=None):
        model_id=self._model_id() if model_id is None else model_id
        if not model_id:return []
        result=[]
        for peer in self.node.peers():
            resources=peer.get("resources",{})
            capabilities=peer.get('capabilities',{})
            models=capabilities.get('language_inference_models_v3',capabilities.get('language_inference_models',resources.get('models',[])))
            if peer["node_id"] in PROVIDERS and capabilities.get("language_inference_v1") and model_id in models and resources.get("has_gpu") and resources.get("ram_gb",0)>=5:
                result.append(peer["node_id"])
        return result

    def snapshot(self):
        pending=self._catalog_pending()
        model_id=self._model_id();providers=self.providers(model_id)
        local=bool(model_id and self.language.ready and self.language.engine.adapter_id==model_id)
        return {"connected":self.connected,"available":self.connected and not pending and (local or bool(providers)),
                "local":local,"remote_providers":len(providers),"model_id":model_id,"active":len(self.requests),
                'catalog_pending':pending,'catalog_error':self.catalog_error,
                'public_memory':self.public_memory.snapshot() if self.public_memory is not None else None}

    def cancel(self,request_id):
        if not isinstance(request_id,str) or not REQUEST_ID.fullmatch(request_id):raise ValueError("Invalid chat cancellation")
        with self.lock:
            event=self.requests.get(request_id)
            if event:event.set()
            self.cancelled[request_id]=time.monotonic()
            if len(self.cancelled)>128:self.cancelled.pop(next(iter(self.cancelled)))
        return bool(event)

    def ask(self,value):
        if not isinstance(value,dict) or set(value)!={"request_id","messages","remote_context"} or not isinstance(value["request_id"],str) or not REQUEST_ID.fullmatch(value["request_id"]) or type(value["remote_context"]) is not bool:
            raise ValueError("Invalid local chat request")
        validate_messages(value["messages"])
        if not self.connected:raise ChatUnavailable("Conecta el nodo antes de preguntar")
        event=threading.Event();request_id=value["request_id"]
        with self.lock:
            if request_id in self.cancelled:raise ChatUnavailable("Consulta cancelada")
            if self.requests:raise ChatUnavailable("Ya hay una consulta en curso")
            self.requests[request_id]=event
        try:
            if self._catalog_pending():raise ChatUnavailable('Espera a verificar la versión de lenguaje disponible')
            model_id=self._model_id()
            if not model_id:raise ChatUnavailable('No se pudo verificar la versión de lenguaje conservada')
            local=self.language.ready and self.language.engine.adapter_id==model_id
            messages=value["messages"] if local or value["remote_context"] else [value["messages"][-1]]
            memory_release=None;memory_records=[]
            if self.public_memory is not None:
                from .public_memory import augment_messages
                cards,memory_release=self.public_memory.retrieve(value['messages'][-1]['content'])
                messages,memory_records=augment_messages(messages,cards)
            validate_messages(messages)
            payload={"model_artifact_id":model_id,"messages":messages}
            requirements=ResourceRequirements(kind="inference",gpu_required=True,ram_gb=5)
            if local:
                provider=self.node.identity.node_id_hex
                result=self.work.execute_local(LANGUAGE_OPERATION,payload,requirements,timeout_s=60,cancel_event=event)
            else:
                providers=self.providers(model_id)
                if not providers:raise ChatUnavailable("No hay un experto de lenguaje disponible; puedes seguir conectado")
                provider=providers[0]
                result=self.work.submit(LANGUAGE_OPERATION,payload,requirements,peer_id=provider,timeout_s=60,cancel_event=event)
            if event.is_set() or not self.connected:raise ChatUnavailable("Consulta cancelada")
            if result.status!="ok":raise ChatUnavailable("El experto no pudo completar la consulta dentro de sus permisos")
            if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=provider,expected_requester_id=self.node.identity.node_id_hex):
                raise ChatUnavailable("No se pudo verificar el recibo de la respuesta")
            output=result.output
            if output.get("model_id")!=model_id or not isinstance(output.get("answer"),str) or not 1<=len(output["answer"].encode())<=8192:
                raise ChatUnavailable("El experto devolvió una respuesta inválida")
            return {"answer":output["answer"],"provider_id":provider,"model_id":model_id,
                    "route":"local" if local else "network","context_sent":not local and value["remote_context"],"receipt":result.receipt,
                    "signed_result":result.to_dict(),'context_bytes':sum(len(row['content'].encode()) for row in messages),
                    'context_messages':len(messages),'public_memory':{'release':memory_release if memory_records else None,
                        'record_ids':memory_records,'used':bool(memory_records)}}
        finally:
            with self.lock:self.requests.pop(request_id,None)

    def close(self):
        self.closed=True;self.connected=False
        self.catalog_cancel.set()
        with self.lock:
            for event in self.requests.values():event.set()
        if self.discovery_thread is not None:self.discovery_thread.join(timeout=1)


class LocalChatAPI:
    def __init__(self,broker,directory,preparation=None,language_work=None):
        self.broker=broker;self.path=Path(directory)/"local-api.json";self.token=secrets.token_hex(32)
        self.preparation=preparation
        self.language_work=language_work
        owner=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version="HTTP/1.0"
            def setup(self):
                super().setup();self.connection.settimeout(5)
            def log_message(self,*_):pass
            def reply(self,status,value):
                encoded=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
                self.send_response(status);self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(encoded)));self.send_header("Cache-Control","no-store")
                self.send_header("X-Content-Type-Options","nosniff");self.end_headers()
                try:self.wfile.write(encoded)
                except (OSError,TimeoutError):pass
            def authenticated(self):
                return (self.headers.get("Host")==f"127.0.0.1:{owner.server.server_port}" and not self.headers.get("Origin")
                        and secrets.compare_digest(self.headers.get("Authorization",""),"Bearer "+owner.token))
            def do_GET(self):
                if not self.authenticated():return self.reply(403,{"error":"Acceso local no autorizado"})
                if self.path=='/preparation/status' and owner.preparation is not None:return self.reply(200,owner.preparation.snapshot())
                if self.path.startswith('/language-work/') and owner.language_work is not None:
                    try:return self.reply(200,owner.language_work.status(self.path[len('/language-work/'):]))
                    except ValueError:return self.reply(404,{'error':'Solicitud desconocida'})
                if self.path!="/status":return self.reply(404,{"error":"Ruta desconocida"})
                self.reply(200,owner.broker.snapshot())
            def do_POST(self):
                if not self.authenticated():return self.reply(403,{"error":"Acceso local no autorizado"})
                if self.headers.get("Content-Type")!="application/json" or self.headers.get("Transfer-Encoding"):
                    return self.reply(400,{"error":"Solicitud inválida"})
                try:
                    length=int(self.headers.get("Content-Length","0"))
                    if not 0<length<=16*1024:raise ValueError()
                    raw=self.rfile.read(length)
                    if len(raw)!=length:raise ValueError()
                    value=json.loads(raw)
                    if self.path=='/language-work/start' and owner.language_work is not None:
                        try:return self.reply(200,owner.language_work.start(value))
                        except RuntimeError as error:return self.reply(409,{'error':str(error)})
                    if self.path=='/language-work/cancel' and owner.language_work is not None:
                        if not isinstance(value,dict) or set(value)!={'request_id'}:raise ValueError('Invalid cancellation')
                        return self.reply(200,{'cancelled':owner.language_work.cancel(value['request_id'])})
                    if self.path in {'/preparation/start','/preparation/stop'} and owner.preparation is not None:
                        if value!={}:raise ValueError('Preparation does not accept peer-selected paths or programs')
                        try:return self.reply(200,owner.preparation.start() if self.path.endswith('/start') else owner.preparation.stop())
                        except (ValueError,RuntimeError) as error:return self.reply(409,{'error':str(error)})
                    if self.path=="/ask":return self.reply(200,owner.broker.ask(value))
                    if self.path=="/cancel" and isinstance(value,dict) and set(value)=={"request_id"}:
                        return self.reply(200,{"cancelled":owner.broker.cancel(value["request_id"])})
                    return self.reply(404,{"error":"Ruta desconocida"})
                except ChatUnavailable as error:self.reply(409,{"error":str(error)})
                except (ValueError,TypeError,KeyError):self.reply(400,{"error":"Solicitud inválida"})
                except Exception:self.reply(503,{"error":"La consulta no pudo completarse"})
        class Server(ThreadingHTTPServer):
            def __init__(self,*args):
                self.slots=threading.BoundedSemaphore(8);super().__init__(*args)
            def process_request(self,request,address):
                if not self.slots.acquire(blocking=False):self.shutdown_request(request);return
                try:super().process_request(request,address)
                except Exception:self.slots.release();raise
            def process_request_thread(self,request,address):
                try:super().process_request_thread(request,address)
                finally:self.slots.release()
        self.server=Server(("127.0.0.1",0),Handler);self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True,name="lixy-local-chat")

    def start(self):
        import os
        atomic_json(self.path,{"version":1,"pid":os.getpid(),"port":self.server.server_port,"token":self.token})
        self.thread.start()

    def close(self):
        if self.language_work is not None:self.language_work.close()
        self.broker.close();self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        try:
            if json.loads(self.path.read_text()).get("token")==self.token:self.path.unlink()
        except (OSError,ValueError):pass
