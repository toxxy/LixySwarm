"""Exercise the publisher CLI with signed synthetic physical-worker receipts."""
import json
import sys

import pytest

from experiments import publish_language_update as publisher
from src.release import language_family
from src.release.language_versions import LanguageVersionRegistry
from test_language_versions import fixture


@pytest.mark.parametrize('valid',[True,False])
def test_publisher_stages_reviewed_files_into_the_real_store(monkeypatch,tmp_path,capsys,valid):
    root=tmp_path/'public';store,first,update,identity=fixture(monkeypatch,root)
    second,report=update(first);identifier=second['release']['model_artifact_id']
    model=tmp_path/'reviewed.safetensors';model.write_bytes(store.object_path(identifier).read_bytes())
    store.object_path(identifier).unlink()
    if not valid:report['candidate_nll']=report['baseline_nll']*2
    evaluation=tmp_path/'reviewed.json';evaluation.write_text(json.dumps(report))
    key=tmp_path/'publisher.key';identity.save(str(key))
    monkeypatch.setattr(publisher,'BASES',language_family.BASES)
    monkeypatch.setattr(sys,'argv',['publish','--directory',str(root),'--publisher-key',str(key),
        '--base-name','qwen3-0.6b','--adapter-file',str(model),'--report-file',str(evaluation)])
    if not valid:
        with pytest.raises(ValueError):publisher.main()
        assert not store.has(identifier)
        assert LanguageVersionRegistry(root,'qwen3-0.6b').head()==first
    else:
        publisher.main();result=json.loads(capsys.readouterr().out)
        assert result['sequence']==1 and result['model_id']==identifier and store.has(identifier)
        assert LanguageVersionRegistry(root,'qwen3-0.6b').head()['release']['release_id']==result['release_id']
