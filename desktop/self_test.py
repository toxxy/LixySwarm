"""Exercise the actual packaged UI and its child node in a clean directory."""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.knowledge and not args.internet:
        parser.error("The real catalog acquisition gate requires --internet")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from desktop.backend import NodeController
    from desktop.window import SwarmWindow
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3

    app = QApplication.instance() or QApplication([])
    report = {"success": False, "internet": args.internet}
    seed = None
    previous_seeds = os.environ.get("LIXYSWARM_BOOTSTRAP_SEEDS")
    with tempfile.TemporaryDirectory(prefix="lixyswarm-package-") as temporary:
        root = Path(temporary)
        controller = NodeController(root / "clean-node", port=0, allow_private=not args.internet)
        def wait(predicate, timeout=30):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                app.processEvents(); window.refresh()
                if predicate():
                    return
                time.sleep(0.05)
            raise RuntimeError("Packaged application did not reach the expected state")
        try:
            if not args.internet:
                seed = LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
                                 address_book_path=root / "seed-peers.json", target_outbound=0, allow_private=True)
                seed.start()
                os.environ["LIXYSWARM_BOOTSTRAP_SEEDS"] = f"127.0.0.1:{seed.port}"
            window = SwarmWindow(controller, initialize_personal=False); window.show(); app.processEvents()
            assert window.tabs.count() == 4
            started = time.monotonic()
            window.connect_button.click()
            wait(lambda: controller.snapshot()["status"].get("state") == "connected" and controller.snapshot()["fresh"])
            connected_s = round(time.monotonic() - started, 2)
            snapshot = controller.snapshot()
            assert connected_s < 30
            assert all(p["encrypted"] for p in snapshot["diagnostics"]["peers"])
            if args.internet:
                import ipaddress
                assert all(ipaddress.ip_address(p.get("relay_host") or p["host"]).is_global for p in snapshot["diagnostics"]["peers"])
            identity = snapshot["status"]["node_id"]
            if args.knowledge:
                window.knowledge_button.click()
                wait(lambda: controller.snapshot()["diagnostics"].get("knowledge", {}).get("state") == "available", timeout=180)
                value=controller.snapshot()["diagnostics"]["knowledge"]
                from src.release import ReleaseRegistry
                from src.release.reference_catalog import REFERENCE_CATALOG
                from src.network.artifact_store import ArtifactStore
                store=ArtifactStore(controller.directory / "public" / "artifacts")
                release=ReleaseRegistry(controller.directory / "public" / "releases").verify_active(REFERENCE_CATALOG.policy,store)
                assert release.release_id == REFERENCE_CATALOG.release_id and value["sharing"]
                report.update(public_knowledge_ready=True, public_release_id=release.release_id,
                              public_model_id=release.model_artifact_id, public_bytes=store.total_bytes())
            window.profile.setCurrentIndex(0); window.cpu.setValue(17); window.gpu.setValue(12)
            window.save_resources()
            wait(lambda: controller.confirmed_settings() and controller.snapshot()["diagnostics"]["settings"]["policy"]["cpu_percent"] == 17)
            window.pause_button.click()
            wait(lambda: controller.snapshot()["status"].get("state") == "paused")
            assert len(controller.snapshot()["diagnostics"]["peers"]) > 0
            if args.knowledge:
                wait(lambda: not controller.snapshot()["diagnostics"]["knowledge"]["sharing"])
                assert store.has(release.model_artifact_id)
                report["pause_preserves_public_weights"] = True
            window.pause_button.click()
            wait(lambda: controller.snapshot()["status"].get("state") == "connected")
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.screenshot))
            window.disconnect_button.click()
            wait(lambda: controller.snapshot()["status"].get("state") == "disconnected"
                 and controller.snapshot()["diagnostics"]["peers"] == [])
            assert controller.snapshot()["diagnostics"]["peers"] == []
            window.connect_button.click()
            wait(lambda: controller.snapshot()["status"].get("state") == "connected")
            assert controller.snapshot()["status"]["node_id"] == identity
            if args.knowledge:
                wait(lambda: controller.snapshot()["diagnostics"]["knowledge"]["state"] == "available")
                assert controller.snapshot()["diagnostics"]["knowledge"]["release_id"] == release.release_id
            assert "torch" not in sys.modules
            report.update(success=True, connection_s=connected_s, controls_confirmed=True,
                          pause_keeps_connection=True, reconnect_keeps_identity=True,
                          calculation_engines_loaded=False)
        except Exception as exc:
            import traceback
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["traceback"] = traceback.format_exc()
            report["node_log"] = controller.log_tail()
        finally:
            if controller.process is not None:
                controller.process.terminate()
                try:
                    controller.process.wait(timeout=15)
                except Exception:
                    controller.process.kill(); controller.process.wait(timeout=5)
            if seed is not None:
                seed.stop()
            if previous_seeds is None:
                os.environ.pop("LIXYSWARM_BOOTSTRAP_SEEDS", None)
            else:
                os.environ["LIXYSWARM_BOOTSTRAP_SEEDS"] = previous_seeds
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    if sys.stdout is not None:
        print(json.dumps(report))
    return 0 if report["success"] else 1
