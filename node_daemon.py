"""
LixySwarm VPS Node — Primer nodo externo de la red P2P
Escucha en 0.0.0.0:7337 (UDP feromon) y 0.0.0.0:7338 (TCP gossip).
Conecta con el nodo local de Emmanuel cuando PEER_HOST esté configurado.

Uso en VPS:
    python3 node_daemon.py

Como servicio systemd:
    Ver VPS_SETUP.md para instrucciones completas.
"""
import sys, time, signal, logging
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

def run():
    from src.network.node import NodeIdentity
    from src.network.swarm_network import SwarmNetwork

    log.info("🐜 LixySwarm VPS Node arrancando...")
    log.info(f"   Feromon UDP: 0.0.0.0:{FEROMON_PORT}")
    log.info(f"   Gossip  TCP: 0.0.0.0:{GOSSIP_PORT}")

    identity = NodeIdentity.generate_anonymous(host="0.0.0.0")
    log.info(f"   Node ID: {identity.node_id}")

    identity.feromon_port = FEROMON_PORT
    identity.gossip_port  = GOSSIP_PORT

    net = SwarmNetwork(identity=identity, mode="lan")
    net.start()
    log.info("✅ SwarmNetwork activo")

    # Conectar al nodo de Emmanuel si IP configurada
    if PEER_HOST:
        try:
            from src.network.node import Peer
            peer = Peer(node_id="emmanuel-local", host=PEER_HOST,
                        feromon_port=FEROMON_PORT, gossip_port=GOSSIP_PORT)
            net.peers.add(peer)
            log.info(f"✅ Peer configurado: {PEER_HOST}:{FEROMON_PORT}")
        except Exception as e:
            log.warning(f"⚠ No se pudo añadir peer: {e}")
    else:
        log.info("ℹ️  PEER_HOST=None — esperando conexiones entrantes (modo relay)")

    def handle_signal(sig, frame):
        log.info("🛑 Señal recibida — cerrando limpiamente")
        net.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log.info("🟢 Nodo activo — listo para recibir feromonas del enjambre")
    tick = 0
    while True:
        time.sleep(60)
        tick += 1
        n_peers = len(net.peers.alive_peers())
        q_size  = len(getattr(net, "_feromon_queue", []))
        log.info(f"💓 tick={tick} peers={n_peers} feromon_queue={q_size}")

if __name__ == "__main__":
    run()
