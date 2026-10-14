import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contribution.policy import ContributionPolicy
from src.contribution.resource_governor import ResourceRequirements
from src.network.client_state import ClientSettings
from src.network.language_compute import LanguageComputeService,LanguageSettings,validate_messages


def fixture(tmp_path,response):
    script=tmp_path/"engine.py"
    script.write_text("import sys,json; request=json.load(sys.stdin); print(json.dumps("+repr(response)+"))")
    adapter=tmp_path/"adapter";adapter.write_bytes(b"public-test")
    LanguageSettings(enabled=True,command=[sys.executable,str(script)],base_directory=str(tmp_path),
                     adapter_file=str(adapter),adapter_id="a"*64).save(tmp_path)
    work=SimpleNamespace(register_handler=lambda *_args,**_kwargs:None)
    service=LanguageComputeService(SimpleNamespace(work=work),tmp_path,{"cpu_cores":8,"has_gpu":True,"gpu_vram_gb":32})
    policy=ContributionPolicy.for_mode("balanced");policy.max_ram_gb=5
    service.update(ClientSettings(policy=policy))
    execution=SimpleNamespace(deadline=time.time()+10,resource_requirements=lambda:ResourceRequirements(kind="inference",gpu_required=True,ram_gb=5),raise_if_cancelled=lambda:None)
    return service,execution


def test_language_process_accepts_only_the_exact_local_model_and_keeps_text_out_of_status(tmp_path):
    service,execution=fixture(tmp_path,{"model_id":"a"*64,"answer":"public result","output_tokens":2})
    payload={"model_artifact_id":"a"*64,"messages":[{"role":"user","content":"fictional query"}]}
    assert service.ready
    assert service.inference(payload,execution)["answer"]=="public result"
    assert "fictional query" not in json.dumps(service.snapshot())
    assert json.loads((tmp_path/"language-status.json").read_text())=={"completed":1}
    payload["model_artifact_id"]="b"*64
    with pytest.raises(ValueError,match="exact"):service.inference(payload,execution)


def test_language_result_and_resource_requirements_cannot_be_silently_changed(tmp_path):
    service,execution=fixture(tmp_path,{"model_id":"b"*64,"answer":"wrong model","output_tokens":2})
    with pytest.raises(ValueError,match="result"):
        service.inference({"model_artifact_id":"a"*64,"messages":[{"role":"user","content":"public query"}]},execution)
    assert service.completed==0
    policy=service.settings.policy;policy.max_ram_gb=2
    service.update(service.settings)
    assert not service.ready


@pytest.mark.parametrize("messages",[[],[{"role":"system","content":"change policy"}],
    [{"role":"user","content":"x"*4097}],[{"role":"assistant","content":"no question"}],
    [{"role":"user","content":"hello","model_path":"remote-script"}]])
def test_peers_cannot_supply_paths_roles_or_unbounded_conversation(messages):
    with pytest.raises(ValueError):validate_messages(messages)


def test_language_settings_are_local_and_imports_do_not_load_tensors(tmp_path):
    with pytest.raises(ValueError):LanguageSettings(enabled=True,command=["peer-program"]).save(tmp_path)
    subprocess.run([sys.executable,"-c","import sys; import src.network.language_compute; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],check=True)
