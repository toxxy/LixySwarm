"""Exercise the actual packaged UI and its child node in a clean directory."""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import contextlib
import time
from pathlib import Path


def main():
    import faulthandler
    if sys.stderr is not None:
        faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--prepare-engine", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--work-directory", type=Path)
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
    if args.work_directory:
        args.work_directory.mkdir(parents=True,exist_ok=True)
        if any(args.work_directory.iterdir()): parser.error("Verification requires an empty work directory")
    workspace=contextlib.nullcontext(str(args.work_directory)) if args.work_directory else tempfile.TemporaryDirectory(prefix="lixyswarm-package-")
    with workspace as temporary:
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
        window=None
        try:
            if not args.internet:
                seed = LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
                                 address_book_path=root / "seed-peers.json", target_outbound=0, allow_private=True)
                seed.start()
                os.environ["LIXYSWARM_BOOTSTRAP_SEEDS"] = f"127.0.0.1:{seed.port}"
            window = SwarmWindow(controller, initialize_personal=False,auto_prepare=args.prepare_engine); window.show(); app.processEvents()
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
                acquiring_started=time.monotonic()
                window.knowledge_button.click()
                wait(lambda: controller.snapshot()["diagnostics"].get("knowledge", {}).get("state") == "available", timeout=180)
                value=controller.snapshot()["diagnostics"]["knowledge"]
                from src.release import ReleaseRegistry
                from src.release.reference_catalog import REFERENCE_CATALOG
                from src.network.artifact_store import ArtifactStore
                store=ArtifactStore(controller.directory / "public" / "artifacts")
                release=ReleaseRegistry(controller.directory / "public" / "releases").verify_active(REFERENCE_CATALOG.policy,store)
                assert release.sequence >= 4 and value["sharing"]
                assert (controller.directory/"public/releases/manifests"/f"{REFERENCE_CATALOG.release_id}.json").exists()
                report.update(public_knowledge_ready=True, public_release_id=release.release_id,
                              public_model_id=release.model_artifact_id, public_bytes=store.total_bytes(),
                              model_acquisition_s=round(time.monotonic()-acquiring_started,2))
            window.profile.setCurrentIndex(0); window.cpu.setValue(17); window.gpu.setValue(12)
            if args.prepare_engine: window.bandwidth.setValue(100)
            engine_started=time.monotonic()
            window.save_resources()
            wait(lambda: controller.confirmed_settings() and controller.snapshot()["diagnostics"]["settings"]["policy"]["cpu_percent"] == 17)
            if args.prepare_engine:
                wait(lambda: window.engine_preparation is not None and not window.engine_preparation.isRunning(), timeout=600)
                assert controller.cpu_engine_prepared(), window.engine_label.text()
                from src.network.reference_compute import EngineSettings
                import subprocess
                engine=EngineSettings.load(controller.directory)
                engine_result=subprocess.run([*engine.command_for("cpu"),"--self-test"],capture_output=True,text=True,timeout=60,check=True)
                checked=json.loads(engine_result.stdout)
                assert checked["success"] and checked["candidate_evaluated"] and not checked["candidate_signed_or_promoted"]
                report.update(engine_downloaded_and_verified=True,engine_frozen=checked["frozen"],
                              engine_candidate_evaluated=checked["candidate_evaluated"],engine_devices=engine.devices,
                              engine_preparation_s=round(time.monotonic()-engine_started,2))
                if args.knowledge:
                    wait(lambda: controller.snapshot()["diagnostics"].get("compute",{}).get("ready"))
                    from src.network.work_protocol import WorkCoordinator,ResultReceipt
                    from src.contribution import ContributionPolicy,ResourceGovernor,ResourceRequirements
                    from src.network.reference_compute import INFERENCE_OPERATION
                    from types import SimpleNamespace
                    from src.network.bootstrap import get_seed_endpoints
                    visitor=LSPNodeV3(LSPIdentity.generate(),host="0.0.0.0",port=0,
                                      seeds=get_seed_endpoints(),address_book_path=root/"visitor-peers.json",allow_private=False)
                    requests=WorkCoordinator(visitor,ResourceGovernor(ContributionPolicy.for_mode("relay"),
                        hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,has_gpu=False,gpu_vram_gb=0,disk_gb=1),
                        storage_path=root/"visitor"))
                    try:
                        visitor.start()
                        until=time.monotonic()+60
                        while identity not in visitor.connected_node_ids:
                            if time.monotonic()>until: raise RuntimeError("The app worker was not reachable over Internet")
                            visitor.connect_node(identity,timeout=12)
                            app.processEvents();time.sleep(0.1)
                        response=requests.submit(INFERENCE_OPERATION,
                            {"model_artifact_id":release.model_artifact_id,"prompt":"La memoria personal "},
                            ResourceRequirements(kind="inference",ram_gb=2),peer_id=identity,timeout_s=60)
                        assert response.status=="ok",response.error
                        assert ResultReceipt.from_dict(response.receipt).verify(response,expected_worker_id=identity,
                                                                             expected_requester_id=visitor.identity.node_id_hex)
                        assert response.output["model_id"]==release.model_artifact_id and len(response.output["next_byte_logits"])==256
                        report.update(app_worker_cpu_inference_verified=True,app_worker_receipt=response.receipt,
                                      app_worker_model_id=response.output["model_id"])
                    finally:
                        requests.close();visitor.stop()
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
            report["knowledge_diagnostics"]=controller.snapshot()["diagnostics"].get("knowledge",{})
        finally:
            if window is not None:
                # Dispose QObject connections while QApplication and Python
                # callbacks are still alive, before interpreter shutdown.
                from PySide6.QtCore import QCoreApplication,QEvent
                if window.engine_preparation is not None and window.engine_preparation.isRunning():
                    window.engine_preparation.cancel.set()
                    window.engine_preparation.wait(15000)
                window.timer.stop()
                window.close()
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
                app.processEvents()
                window=None
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
