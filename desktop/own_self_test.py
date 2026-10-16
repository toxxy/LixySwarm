"""Native cold own preparation on a real Internet path, in a fresh local profile."""
import json
from pathlib import Path
import sys
import time


def run(args):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, QEvent
    from desktop.backend import NodeController
    from desktop.window import SwarmWindow
    from src.network import own_catalog as catalog
    from src.network.artifact_store import digest_file
    from src.network.engine_package import verify_package
    from src.network.telemetry import atomic_json
    from src.network.work_protocol import WorkResult, ResultReceipt
    root = args.work_directory.resolve(); root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise ValueError('Cold own validation needs a fresh external directory')
    if root.is_relative_to(Path.home()/'Dropbox') or root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Cold own state must remain outside the code checkout')
    app = QApplication.instance() or QApplication([])
    controller = NodeController(root/'clean-node', port=0, allow_private=False)
    window = SwarmWindow(controller, initialize_personal=False, auto_prepare=True)
    report = {'success': False, 'internet': True, 'fresh_profile': True,
        'extra_physical_pc': False, 'model_state_copied_by_admin': False, 'anchor_id': catalog.anchor_id()}
    def status(): return controller.snapshot()['diagnostics']
    def wait(predicate, timeout=60):
        until = time.monotonic()+timeout
        while time.monotonic()<until:
            app.processEvents(); window.refresh()
            d = status(); prep = d.get('own_preparation', {})
            atomic_json(root/'live.json', {'node_id': d.get('node_id'), 'pid': d.get('pid'),
                'preparation': prep, 'own_swarm': d.get('own_swarm', {})})
            if predicate(): return
            time.sleep(.1)
        raise TimeoutError('Cold own application did not reach its expected state: '+str(status().get('own_preparation')))
    try:
        window.show(); window.profile.setCurrentIndex(0)
        window.cpu.setValue(100); window.gpu.setValue(0); window.ram.setValue(4)
        window.disk.setValue(2); window.bandwidth.setValue(10); window.battery.setChecked(False)
        began = time.monotonic(); window.connect_button.click()
        wait(lambda: controller.snapshot()['fresh'] and controller.snapshot()['status'].get('state') in {'connected','preparing','working'})
        report['connection_seconds'] = time.monotonic()-began
        wait(lambda: status().get('own_preparation', {}).get('state') == 'engine-downloading'
            and status()['own_preparation'].get('done',0)>256*1024, timeout=180)
        identity = status()['node_id']; window.pause_button.click()
        wait(lambda: controller.snapshot()['status'].get('state') == 'paused'
            and status().get('own_preparation', {}).get('state') == 'stopped', timeout=40)
        partials = list((controller.directory/'engine-downloads').glob('*.part'))
        assert partials and not (controller.directory/'own-swarm.json').exists()
        report['pause_preserved_partial_bytes'] = sum(p.stat().st_size for p in partials)
        report['no_selection_before_validation'] = True
        window.pause_button.click()
        wait(lambda: status().get('own_swarm', {}).get('ready') is True, timeout=1800)
        snapshot = status(); own = snapshot['own_swarm']
        assert own['state_id'] == catalog.ANCHOR['state_id'] and own['optimizer_step'] == catalog.ANCHOR['step']
        result = json.loads((controller.directory/'own-preparation-result.json').read_text())
        assert result['success'] and result['audit']['optimizer_states'] == 175
        assert result['audit']['saved_state_unchanged'] and not result['audit']['new_gradient_computed']
        assert all(t['route']['encrypted'] and t['route']['public_internet'] for t in result['transfers'])
        selected = json.loads((controller.directory/'own-swarm.json').read_text())
        entry = catalog.own_engine(); engine = Path(selected['command'][0]).parent
        verify_package(engine, entry.manifest_sha256, profile=entry.profile)
        for identifier,size in catalog.dependencies():
            assert digest_file(controller.directory/'public/own/artifacts/objects'/identifier[:2]/identifier[2:]) == (identifier,size)
        atomic_json(root/'own-ready.json', {'node_id': identity, 'state_id': own['state_id'],
            'model_id': own['model_id'], 'optimizer_step': own['optimizer_step'], 'pid': snapshot['pid']})
        report.update(preparation_seconds=time.monotonic()-began, preparation=result,
            physical_profile_node_id=identity, cpu_engine_manifest_sha256=entry.manifest_sha256,
            optimizer_step=own['optimizer_step'], state_id=own['state_id'])
        # A controller on the existing coordinator requests native work. Its
        # acknowledgement contains public metadata only, after gradient fetch.
        wait(lambda: (root/'work-ack.json').is_file(), timeout=300)
        ack = json.loads((root/'work-ack.json').read_text()); work = WorkResult.from_dict(ack['result'])
        assert work.status == 'ok' and ResultReceipt.from_dict(work.receipt).verify(work,
            expected_worker_id=identity, expected_requester_id=catalog.COORDINATOR)
        assert work.output['state_id'] == own['state_id'] and work.output['model_id'] == own['model_id']
        assert ack['route']['encrypted'] and ack['route']['public_internet']
        report['native_gradient_work'] = ack
        window.tabs.setCurrentIndex(3)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True); window.grab().save(str(args.screenshot))
        window.disconnect_button.click()
        wait(lambda: controller.snapshot()['status'].get('state') == 'disconnected')
        window.connect_button.click()
        wait(lambda: status().get('own_swarm', {}).get('ready') is True)
        assert status()['node_id'] == identity and status()['own_swarm']['state_id'] == own['state_id']
        assert 'torch' not in sys.modules
        report.update(success=True, resumed_same_identity_and_state=True, calculation_engines_loaded=False,
            final_own_swarm=status()['own_swarm'])
    except Exception as error:
        import traceback
        report.update(error=type(error).__name__+': '+str(error), traceback=traceback.format_exc(),
            preparation=status().get('own_preparation'), node_log=controller.log_tail())
    finally:
        window.timer.stop(); window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete); app.processEvents()
        if controller.process is not None:
            controller.process.terminate()
            try: controller.process.wait(timeout=20)
            except Exception: controller.process.kill(); controller.process.wait(timeout=5)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.report, report)
    print(json.dumps(report), flush=True)
    return 0 if report['success'] else 1
