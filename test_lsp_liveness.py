"""A TCP connection can stay open while every application packet is lost."""
import asyncio
import threading

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3


def test_blackholed_tcp_is_closed_and_recovered_without_restarting_the_node(tmp_path):
    async def scenario():
        target = LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
            address_book_path=tmp_path / "target.json", allow_private=True,
            target_outbound=0, heartbeat_interval=0.1, session_idle_timeout=0.6)
        source = None
        writers, proxy_tasks = [], []
        dropped = False
        lost = threading.Event()

        async def proxy(reader, writer):
            peer_reader, peer_writer = await asyncio.open_connection("127.0.0.1", target.port)
            writers.extend((writer, peer_writer))
            async def forward(origin, destination):
                while data := await origin.read(65536):
                    if not dropped:
                        destination.write(data)
                        await destination.drain()
                destination.close()
            task = asyncio.gather(forward(reader, peer_writer), forward(peer_reader, writer))
            proxy_tasks.append(task)
            await task

        async def wait(predicate, timeout=15):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return True
                await asyncio.sleep(0.01)
            return False

        await asyncio.to_thread(target.start)
        server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        proxy_port = server.sockets[0].getsockname()[1]
        try:
            source = LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
                address_book_path=tmp_path / "source.json", allow_private=True,
                seeds=[("127.0.0.1", proxy_port)], target_outbound=1,
                maintenance_interval=0.1, heartbeat_interval=0.1, session_idle_timeout=0.6)
            original_connect = source._connect_address
            async def only_proxy(host, port, **kwargs):
                # As with a NAT network, the discovered direct address fails.
                return await original_connect(host, port, **kwargs) if port == proxy_port else False
            source._connect_address = only_proxy
            source.on_peer_lost(lambda peer: lost.set())
            await asyncio.to_thread(source.start)
            assert await wait(lambda: target.identity.node_id_hex in source.connected_node_ids)
            first = source.peers()[0]["connected_at"]
            # An idle healthy session survives several full idle windows.
            await asyncio.sleep(1.3)
            assert not lost.is_set() and source.peers()[0]["connected_at"] == first
            dropped = True
            assert await wait(lost.is_set, timeout=3)
            dropped = False
            assert await wait(lambda: bool(source.peers()) and source.peers()[0]["connected_at"] > first)
            assert source.peers()[0]["encrypted"]
        finally:
            if source:
                await asyncio.to_thread(source.stop)
            await asyncio.to_thread(target.stop)
            server.close(); await server.wait_closed()
            for writer in writers:
                writer.close()
            for task in proxy_tasks:
                task.cancel()
            await asyncio.gather(*proxy_tasks, return_exceptions=True)
    asyncio.run(scenario())
