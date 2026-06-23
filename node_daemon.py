"""LixySwarm LSP v3 public seed node.

The seed accepts persistent sessions and introduces peers. It is not a central
relay: connected nodes establish their own sessions through peer exchange and
continue operating if this process disappears.
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_log_handlers = [logging.StreamHandler()]
if os.environ.get("LIXYSWARM_LOG_FILE"):
    _log_handlers.append(logging.FileHandler(os.environ["LIXYSWARM_LOG_FILE"]))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LSP-v3-seed] %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger(__name__)

LISTEN_HOST = os.environ.get("LIXYSWARM_LISTEN_HOST", "0.0.0.0")
GOSSIP_PORT = int(os.environ.get("LIXYSWARM_GOSSIP_PORT", "7338"))
PUBLIC_HOST = os.environ.get("LIXYSWARM_PUBLIC_HOST")
TARGET_OUTBOUND = int(os.environ.get("LIXYSWARM_TARGET_OUTBOUND", "0"))
ALLOW_PRIVATE = os.environ.get("LIXYSWARM_ALLOW_PRIVATE_PEERS", "").lower() in {
    "1", "true", "yes",
}
IDENTITY_PATH = Path(os.environ.get(
    "LIXYSWARM_IDENTITY_PATH", "/opt/lixyswarm/.lixyswarm/identity.key"
))
PEERS_PATH = Path(os.environ.get(
    "LIXYSWARM_PEERS_PATH", str(IDENTITY_PATH.parent / "peers_v3.json")
))


def run():
    from src.network.bootstrap import get_seed_endpoints
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3

    identity = LSPIdentity.load(str(IDENTITY_PATH))
    if identity is None:
        identity = LSPIdentity.generate()
        identity.save(str(IDENTITY_PATH))
        log.info("Generated persistent Ed25519 identity")

    node = LSPNodeV3(
        identity,
        host=LISTEN_HOST,
        port=GOSSIP_PORT,
        advertised_host=PUBLIC_HOST,
        seeds=get_seed_endpoints(),
        address_book_path=PEERS_PATH,
        target_outbound=TARGET_OUTBOUND,
        allow_private=ALLOW_PRIVATE,
        capabilities={
            "peer_exchange": True,
            "seed": True,
            "pheromone": False,
            "global_memory": False,
        },
        resource_profile={
            "mode": "relay",
            "cpu_cores": 1,
            "ram_gb": 0.0,
            "gpu_vram_gb": 0.0,
            "disk_gb": 0.0,
            "has_gpu": False,
        },
    )

    @node.on_peer_connected
    def peer_connected(node_id, host, port, hello):
        mode = hello.get("resources", {}).get("mode", "unknown")
        log.info("Peer connected id=%s host=%s port=%s mode=%s",
                 node_id[:16], host, port, mode)

    @node.on_peer_lost
    def peer_lost(node_id):
        log.info("Peer disconnected id=%s", node_id[:16])

    node.start()
    log.info(
        "Seed active id=%s TCP=%s known=%s target_outbound=%s",
        identity.node_id_hex[:16], node.port, node.address_book.count, TARGET_OUTBOUND,
    )

    stopping = False

    def handle_signal(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not stopping:
        time.sleep(30)
        log.info(
            "heartbeat connected=%s outbound=%s known=%s",
            node.peer_count,
            node.outbound_count,
            node.address_book.count,
        )

    node.stop()
    log.info("Seed stopped cleanly")


if __name__ == "__main__":
    run()
