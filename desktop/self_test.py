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
    parser.add_argument('--own-bootstrap',action='store_true')
    parser.add_argument("--chat",action="store_true")
    parser.add_argument('--memory',action='store_true')
    parser.add_argument('--expected-chat-model')
    parser.add_argument('--global-memory',action='store_true')
    parser.add_argument('--expected-memory-release')
    parser.add_argument("--report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument('--upgrade-from', type=Path)
    args = parser.parse_args()
    if args.own_bootstrap:
        if not args.internet or not args.work_directory or args.prepare_engine or args.chat or args.knowledge:
            parser.error('Own cold validation needs --internet --work-directory and its separate preparation path')
        from desktop.own_self_test import run
        return run(args)
    if args.upgrade_from and (not getattr(sys, 'frozen', False) or sys.platform != 'linux' or not args.upgrade_from.is_file()):
        parser.error('Runtime upgrade verification requires a Linux native app and an existing native executable')
    if args.knowledge and not args.internet:
        parser.error("The real catalog acquisition gate requires --internet")
    if args.chat and not args.internet:parser.error("The real chat assessment requires --internet")
    if args.memory and not args.chat:parser.error('Memory comparison requires the real chat assessment')
    if args.global_memory and not args.chat:parser.error('Public memory comparison requires chat')
    if args.expected_memory_release and (not args.global_memory or len(args.expected_memory_release)!=64
            or any(c not in '0123456789abcdef' for c in args.expected_memory_release)):
        parser.error('Expected public memory release is a content assertion')
    if args.expected_chat_model and (not args.chat or len(args.expected_chat_model)!=64
            or any(c not in '0123456789abcdef' for c in args.expected_chat_model)):
        parser.error('Expected chat model is an assertion against a public artifact hash')
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from desktop.backend import NodeController
    from desktop.window import SwarmWindow
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3

    app = QApplication.instance() or QApplication([])
    report = {"success": False, "internet": args.internet}
    if args.expected_chat_model:report['expected_chat_model_id']=args.expected_chat_model
    seed = None
    previous_seeds = os.environ.get("LIXYSWARM_BOOTSTRAP_SEEDS")
    previous_data_home = os.environ.get('XDG_DATA_HOME')
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
            window = SwarmWindow(controller, initialize_personal=args.chat,auto_prepare=args.prepare_engine); window.show(); app.processEvents()
            assert window.tabs.count() == 4
            if args.upgrade_from:
                # The fixture has no physical participant state or model. Keep
                # its installer and launcher outside the operator's real app.
                os.environ['XDG_DATA_HOME'] = str(root / 'installed-app')
                controller.settings().save(controller.directory)
                controller._launch_node(executable=args.upgrade_from.resolve())
                wait(lambda: controller.snapshot()['fresh'])
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
            if args.upgrade_from:
                from src.version import VERSION
                before = (controller.directory / 'controls.json').read_bytes()
                previous_version = snapshot['diagnostics']['runtime_version']
                previous_instance = snapshot['status']['process_instance_id']
                assert controller.runtime_update_available(snapshot)
                window.refresh(); assert window.update_button.isVisible()
                window.update_button.click()
                wait(lambda: window.runtime_update is not None and not window.runtime_update.isRunning(), timeout=60)
                wait(lambda: controller.snapshot()['fresh'] and controller.snapshot()['diagnostics'].get('runtime_version') == VERSION)
                upgraded = controller.snapshot()
                assert upgraded['status']['node_id'] == identity
                assert upgraded['status']['process_instance_id'] != previous_instance
                assert (controller.directory / 'controls.json').read_bytes() == before
                assert not controller.runtime_update_available(upgraded)
                report['runtime_upgrade'] = {'from': previous_version, 'to': VERSION,
                    'identity_preserved': True, 'controls_preserved': True, 'new_process': True,
                    'native_executable': True, 'physical_node': False}
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
            # The packaged local API must reject expert preparation outside
            # these deliberately small RAM/GPU permissions, without changing
            # the policy or writing a preparation intent.
            previous_policy=controller.settings().policy
            preparation_status=controller._chat_call('/preparation/status')
            assert preparation_status['state']=='idle'
            try:
                controller.prepare_language()
            except RuntimeError as error:
                assert str(error)!='Ruta desconocida'
                assert controller.settings().policy==previous_policy
                assert not (controller.directory/'language-preparation-intent.json').exists()
                report['language_preparation_permission_gate']=True
            else:
                raise AssertionError('Packaged expert preparation ignored resource limits')
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
            if args.chat:
                wait(lambda: window.chat_history is not None,timeout=20)
                wait(lambda: window.chat_send.isEnabled(),timeout=90)
                if args.expected_chat_model:
                    assert controller.chat_status()['model_id']==args.expected_chat_model,'The light chat selected another model version'
                window.tabs.setCurrentIndex(1)
                canary="synthetic-private-history-canary-3081"
                window.chat_messages=[{"role":"user","content":canary}];window.save_chat_history()
                window.chat_input.setPlainText("¿Qué es una red descentralizada? Explica brevemente en español.")
                window.chat_send.click()
                wait(lambda:window.chat_worker is not None and not window.chat_pending and not window.chat_worker.isRunning(),timeout=75)
                assert window.chat_last_result and window.chat_last_result["route"]=="network",window.chat_status.text()
                assert not window.chat_last_result["context_sent"]
                report['chat_model_id']=window.chat_last_result['model_id']
                report["chat_first_answer"]=window.chat_last_result["answer"]
                for path in controller.directory.rglob("*"):
                    if path.is_file() and path.name not in {"node.lock"}:
                        assert canary.encode() not in path.read_bytes(),path.name
                window.chat_messages=[];window.chat_context.setChecked(True);window.save_chat_history()
                window.chat_input.setPlainText("En este ejemplo ficticio, el equipo se llama Lavanda. Confirma su nombre.")
                window.chat_send.click();wait(lambda:not window.chat_pending and not window.chat_worker.isRunning(),timeout=75)
                assert window.chat_last_result,window.chat_status.text()
                assert window.chat_messages[-1]["role"]=="assistant"
                window.chat_input.setPlainText("¿Cómo se llama el equipo del ejemplo? Responde solo su nombre.")
                window.chat_send.click();wait(lambda:not window.chat_pending and not window.chat_worker.isRunning(),timeout=75)
                assert window.chat_last_result,window.chat_status.text()
                assert window.chat_messages[-1]["role"]=="assistant"
                assert "lavanda" in window.chat_messages[-1]["content"].lower(),window.chat_messages[-1]
                assert window.chat_last_result["context_sent"]
                previous_messages=list(window.chat_messages)
                last_chat_receipt=window.chat_last_result["receipt"]
                window.timer.stop();window.close()
                from PySide6.QtCore import QCoreApplication,QEvent
                window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
                window=SwarmWindow(controller,initialize_personal=True,auto_prepare=False);window.show()
                wait(lambda:window.chat_history is not None,timeout=20)
                assert window.chat_messages==previous_messages and window.chat_context.isChecked()
                window.tabs.setCurrentIndex(1)
                report.update(chat_ui_verified=True,chat_remote_context_opt_in=True,chat_history_survives_reopen=True,
                              chat_history_encrypted=True,chat_receipt=last_chat_receipt)
                if args.memory:
                    import subprocess
                    from src.learning.public_memory_gate import memory_cases,statement_messages,distractor_messages,verdict
                    comparisons=[]
                    def ask_memory(question):
                        wait(lambda:window.chat_send.isEnabled())
                        messages,recalled=window.chat_history.context(window.chat_messages,question,enabled=window.chat_memory.isChecked())
                        window.chat_input.setPlainText(question);window.chat_send.click()
                        wait(lambda:not window.chat_pending and not window.chat_worker.isRunning(),timeout=75)
                        assert window.chat_last_result,window.chat_status.text()
                        result=window.chat_last_result
                        assert result['context_sent'] and result['route']=='network'
                        if args.expected_chat_model:assert result['model_id']==args.expected_chat_model
                        return {**result,'recalled_episodes':window.chat_last_memories,
                            'context_bytes':sum(len(row['content'].encode()) for row in messages),'context_messages':len(messages)}
                    for case in memory_cases():
                        window.chat_forget.click();window.chat_messages=[]
                        window.chat_memory.setChecked(True);window.chat_context.setChecked(True)
                        window.chat_messages=statement_messages(case);window.save_chat_history()
                        window.chat_messages+=distractor_messages();window.save_chat_history()
                        assert case['expected'] not in json.dumps(window.chat_messages)
                        window.chat_memory.setChecked(False)
                        recent=ask_memory(case['question'])
                        assert recent['recalled_episodes']==0
                        # Restore the identical visible background. The previous
                        # answer cannot become a memory or satisfy the next ask.
                        window.chat_messages=distractor_messages()[-200:]
                        window.chat_memory.setChecked(True);window.save_chat_history()
                        command=([sys.executable] if getattr(sys,'frozen',False) else [sys.executable,str(Path(__file__).with_name('main.py'))])
                        restored=subprocess.run([*command,'--memory-fixture-check','--directory',str(controller.directory),'--case',case['case']],
                            capture_output=True,text=True,timeout=20,check=True)
                        fresh=json.loads(restored.stdout)
                        assert fresh['success'] and fresh['restored_in_fresh_process']
                        window.timer.stop();window.close();window.deleteLater()
                        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
                        window=SwarmWindow(controller,initialize_personal=True,auto_prepare=False);window.show()
                        wait(lambda:window.chat_history is not None,timeout=20)
                        assert window.chat_memory.isChecked() and window.chat_context.isChecked()
                        remembered=ask_memory(case['question'])
                        comparisons.append({'case':case['case'],'recent':recent,'memory':remembered,'fresh_process':fresh})
                    report['personal_memory_comparison']={'results':comparisons,'verdict':verdict(comparisons),
                        'private_inputs':False,'model_training_performed':False,'global_signals_from_personal':False}
                    assert report['personal_memory_comparison']['verdict']['eligible']
                if args.global_memory:
                    from src.network.public_memory import PublicMemorySettings
                    from src.release.public_memory import PublicMemoryRegistry
                    from src.learning.public_memory_gate import memory_cases
                    window.chat_context.setChecked(False);window.chat_memory.setChecked(False)
                    wait(lambda:controller.snapshot()['diagnostics'].get('public_memory',{}).get('sequence')==1,timeout=120)
                    registry=PublicMemoryRegistry(controller.directory);document=registry.document();head=registry.chain()[-1]
                    if args.expected_memory_release:assert head['release_id']==args.expected_memory_release
                    source_head=document['sources']['accepted_language']['chain'][-1]['release']
                    expected={row['case']:row['expected'] for row in memory_cases()}
                    expected.update({'language-sequence':str(source_head['sequence']),'language-adapter':source_head['model_artifact_id']})
                    comparisons=[]
                    for card in document['cards']:
                        row={'record_id':card['record_id'],'expected':expected[card['record_id']]}
                        for enabled,mode in [(False,'without_memory'),(True,'with_memory')]:
                            controller.set_public_memory(enabled)
                            wait(lambda:controller.snapshot()['diagnostics'].get('public_memory',{}).get('enabled')==enabled)
                            window.chat_messages=[];window.save_chat_history()
                            window.chat_input.setPlainText(card['example_question']);window.chat_send.click()
                            wait(lambda:not window.chat_pending and window.chat_worker is not None and not window.chat_worker.isRunning(),timeout=75)
                            result=window.chat_last_result;assert result,window.chat_status.text()
                            assert not result['context_sent'] and result['route']=='network'
                            assert result['public_memory']['used']==enabled
                            if args.expected_chat_model:assert result['model_id']==args.expected_chat_model
                            if enabled:assert result['public_memory']['release']['release_id']==head['release_id']
                            row[mode]=result
                        comparisons.append(row)
                    import re
                    before=after=0;regressions=[]
                    for row in comparisons:
                        scores=[bool(re.fullmatch(re.escape(row['expected'])+r'[.!]?',row[mode]['answer'].strip(),flags=re.I))
                            for mode in ['without_memory','with_memory']]
                        before+=scores[0];after+=scores[1]
                        if scores[0] and not scores[1]:regressions.append(row['record_id'])
                    report['public_memory_comparison']={'head':head,'results':comparisons,'without_correct':before,
                        'with_correct':after,'total':len(comparisons),'regressions':regressions,'private_inputs':False,
                        'new_model_trained':False,'eligible':after>=9 and after-before>=6 and not regressions}
                    assert report['public_memory_comparison']['eligible']
                    controller.set_public_memory(True)
                # Cancel at dispatch, including the race before the API records
                # the request. A previous response cannot satisfy this check.
                wait(lambda:window.chat_send.isEnabled())
                window.chat_input.setPlainText("Enumera cien ejemplos ficticios de colaboración.")
                window.chat_send.click();window.cancel_chat()
                wait(lambda:not window.chat_pending and not window.chat_worker.isRunning(),timeout=75)
                assert window.chat_last_result is None,window.chat_status.text()
                assert window.chat_messages[-1]["role"]=="user"
                report["chat_cancellation_verified"]=True
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
                if window.chat_worker is not None and window.chat_worker.isRunning():
                    window.cancel_chat();window.chat_worker.wait(15000)
                if window.runtime_update is not None and window.runtime_update.isRunning():
                    window.runtime_update.wait(60000)
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
            if previous_data_home is None:
                os.environ.pop('XDG_DATA_HOME', None)
            else:
                os.environ['XDG_DATA_HOME'] = previous_data_home
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    if sys.stdout is not None:
        print(json.dumps(report))
    return 0 if report["success"] else 1
