"""
LixySwarm VPS Node — Primer nodo externo de la red P2P
Escucha en 0.0.0.0:7337 (UDP feromon) y 0.0.0.0:7338 (TCP gossip).
Conecta con el nodo local de Emmanuel cuando PEER_HOST esté configurado.

Uso en VPS:
    python3 node_daemon.py

Como servicio systemd:
    Ver VPS_SETUP.md para instrucciones completas.
"""
import sys, time, signal, logging, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VPS-Node] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("node.log"),
    ]
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
PEER_HOST    = None       # IP del nodo local de Emmanuel — None = solo escuchar
FEROMON_PORT = 7337       # UDP
GOSSIP_PORT  = 7338       # TCP

IDENTITY_PATH = "/opt/lixyswarm/.lixyswarm/identity.key"

def run():
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v2 import LSPNodeV2

    log.info("🐜 LixySwarm VPS Node arrancando (LSP v2)...")
    log.info(f"   Feromon UDP: 0.0.0.0:{FEROMON_PORT}")
    log.info(f"   Gossip  TCP: 0.0.0.0:{GOSSIP_PORT}")
    log.info("   Wire: LYSW binary · float16 feromonas · merge-on-transit")

    # Cargar o generar identidad Ed25519 persistente
    identity = LSPIdentity.load(IDENTITY_PATH)
    if identity is None:
        log.info("🔑 Generando nueva identidad Ed25519...")
        identity = LSPIdentity.generate()
        identity.save(IDENTITY_PATH)
        log.info(f"   Identidad guardada en {IDENTITY_PATH}")
    log.info(f"   Node ID: {identity.node_id_hex[:32]}...")

    node = LSPNodeV2(identity, feromon_port=FEROMON_PORT, gossip_port=GOSSIP_PORT)

    @node.on_feromon_received
    def on_feromon(feromon, from_node_id):
        try:
            norm = sum(x*x for x in (feromon if not hasattr(feromon, 'norm') else [])) ** 0.5
            if hasattr(feromon, 'norm'):
                norm = feromon.norm().item()
            log.info(f"🐜 Feromona recibida de {from_node_id[:16]}... norm={norm:.3f}")
        except Exception as e:
            log.info(f"🐜 Feromona recibida de {from_node_id[:16]}...")

    @node.on_peer_connected
    def on_peer(node_id, host, port):
        log.info(f"🔗 Peer conectado: {node_id[:16]}...@{host}:{port}")

    node.start()
    log.info("✅ LSPNodeV2 activo")

    # Conectar al nodo de Emmanuel si IP configurada
    if PEER_HOST:
        try:
            node.connect_peer(PEER_HOST, GOSSIP_PORT)
            log.info(f"✅ Handshake enviado a {PEER_HOST}:{GOSSIP_PORT}")
        except Exception as e:
            log.warning(f"⚠ No se pudo conectar al peer: {e}")
    else:
        log.info("ℹ️  PEER_HOST=None — esperando conexiones entrantes (modo relay)")

    def handle_signal(sig, frame):
        log.info("🛑 Señal recibida — cerrando limpiamente")
        node.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log.info("🟢 Nodo activo — listo para recibir feromonas del enjambre")
    tick = 0
    while True:
        time.sleep(60)
        tick += 1
        n_peers = len(node.peers())
        log.info(f"💓 tick={tick} peers={n_peers}")

        # Auto-descubrimiento: leer swarm_status.json para detectar el nodo local
        # El swarm_publisher en la PC local sube este archivo con su identidad LSP
        # El handshake real lo inicia el nodo local (outbound) → VPS (inbound) por NAT
        status_file = __import__("pathlib").Path(__file__).parent / "swarm_status.json"
        if status_file.exists() and n_peers == 0:
            try:
                data = json.loads(status_file.read_text())
                published_peers = data.get("peers", [])
                for p in published_peers:
                    p_host = p.get("host", "")
                    p_gossip = p.get("gossip_port", 7338)
                    p_node_id = p.get("id", "?")
                    if p_host and p_host not in ("127.0.0.1", "0.0.0.0", "localhost"):
                        log.info(f"📋 Nodo local registrado en swarm_status.json: {p_node_id[:16]}...@{p_host}:{p_gossip}")
                        log.info(f"   ⏳ Esperando handshake LSP v2 entrante del nodo local (NAT outbound)")
            except Exception as e:
                log.debug(f"Auto-descubrimiento vía swarm_status.json: {e}")

if __name__ == "__main__":
    run()
