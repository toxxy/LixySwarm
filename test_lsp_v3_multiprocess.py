"""Process-isolated acceptance test for seed-independent LSP v3 routing."""

import multiprocessing
import time
from pathlib import Path

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3


def _run_isolated_node(control, events, state_path, seeds, target_outbound):
    """Own one LSP node in a separate interpreter and expose test-only IPC."""
    node = None
    try:
        node = LSPNodeV3(
            LSPIdentity.generate(),
            host="127.0.0.1",
            port=0,
            advertised_host="127.0.0.1",
            seeds=list(seeds),
            address_book_path=Path(state_path),
            target_outbound=target_outbound,
            allow_private=True,
            maintenance_interval=0.1,
            resource_profile={"mode": "relay", "cpu_cores": 1},
        )

        @node.on_gossip_delta_received
        def capture(delta, sender_id):
            events.put({
                "kind": "delta",
                "delta": delta,
                "sender_id": sender_id,
            })

        node.start()
        control.send({
            "kind": "started",
            "node_id": node.identity.node_id_hex,
            "port": node.port,
        })
        while True:
            command = control.recv()
            if command == "status":
                control.send({
                    "kind": "status",
                    "peers": node.peers(),
                    "outbound": node.outbound_count,
                })
            elif command == "broadcast":
                delta = {
                    "kind": "multiprocess_seed_shutdown",
                    "version": 1,
                    "value": "direct",
                }
                control.send({
                    "kind": "broadcast",
                    "count": node.broadcast_global_delta(delta),
                    "delta": delta,
                })
            elif command == "stop":
                node.stop()
                control.send({"kind": "stopped"})
                return
            else:
                raise RuntimeError("unknown test command")
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as exc:
        try:
            control.send({
                "kind": "error",
                "error_type": type(exc).__name__,
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if node is not None:
            try:
                node.stop()
            except Exception:
                pass
        control.close()


def _start_node(context, tmp_path, name, *, seeds=(), target_outbound=2):
    parent, child = context.Pipe(duplex=True)
    events = context.Queue()
    process = context.Process(
        target=_run_isolated_node,
        args=(
            child,
            events,
            str(tmp_path / f"{name}-peers.json"),
            tuple(seeds),
            target_outbound,
        ),
        name=f"lsp-v3-test-{name}",
    )
    process.start()
    child.close()
    try:
        assert parent.poll(15), f"{name} process did not start"
        started = parent.recv()
        assert started["kind"] == "started", started
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
        events.close()
        raise
    return process, parent, events, started


def _call(control, command, timeout=5.0):
    control.send(command)
    assert control.poll(timeout), f"node did not answer {command}"
    response = control.recv()
    assert response["kind"] != "error", response
    return response


def _stop_process(process, control, events):
    if process.is_alive():
        try:
            response = _call(control, "stop", timeout=5.0)
            assert response["kind"] == "stopped"
        except (AssertionError, BrokenPipeError, EOFError, OSError):
            process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    control.close()
    events.close()
    events.join_thread()


def test_separate_processes_continue_securely_after_seed_is_killed(tmp_path):
    context = multiprocessing.get_context("spawn")
    nodes = []
    seed = left = right = None
    try:
        seed = _start_node(
            context, tmp_path, "seed", target_outbound=0
        )
        nodes.append(seed)
        seed_endpoint = ("127.0.0.1", seed[3]["port"])
        left = _start_node(
            context, tmp_path, "left", seeds=[seed_endpoint]
        )
        nodes.append(left)
        right = _start_node(
            context, tmp_path, "right", seeds=[seed_endpoint]
        )
        nodes.append(right)

        left_id = left[3]["node_id"]
        right_id = right[3]["node_id"]
        deadline = time.monotonic() + 15
        direct_left = direct_right = None
        while time.monotonic() < deadline:
            left_status = _call(left[1], "status")
            right_status = _call(right[1], "status")
            direct_left = next(
                (peer for peer in left_status["peers"]
                 if peer["node_id"] == right_id),
                None,
            )
            direct_right = next(
                (peer for peer in right_status["peers"]
                 if peer["node_id"] == left_id),
                None,
            )
            if direct_left and direct_right:
                break
            time.sleep(0.1)

        assert direct_left is not None, "left did not discover right directly"
        assert direct_right is not None, "right did not discover left directly"
        assert direct_left["encrypted"] is True
        assert direct_right["encrypted"] is True

        # Abrupt process death is stronger than a graceful protocol shutdown.
        seed[0].terminate()
        seed[0].join(timeout=10)
        assert not seed[0].is_alive()
        assert seed[0].exitcode is not None
        seed[1].close()
        seed[2].close()
        seed[2].join_thread()
        nodes.remove(seed)

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            left_status = _call(left[1], "status")
            right_status = _call(right[1], "status")
            if (
                any(peer["node_id"] == right_id and peer["encrypted"]
                    for peer in left_status["peers"])
                and any(peer["node_id"] == left_id and peer["encrypted"]
                        for peer in right_status["peers"])
            ):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("direct encrypted route died with the seed")

        sent = _call(left[1], "broadcast")
        assert sent["count"] >= 1
        received = right[2].get(timeout=5)
        assert received == {
            "kind": "delta",
            "delta": sent["delta"],
            "sender_id": left_id,
        }
    finally:
        for process, control, events, _started in reversed(nodes):
            _stop_process(process, control, events)
