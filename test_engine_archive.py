import hashlib
import io
import json
from pathlib import Path
import platform
import stat
import sys
import zipfile

import pytest

from engines.archive import archive_package
from src.network.engine_catalog import EngineDownload
from src.network.engine_download import prepare_engine
from src.network.engine_profiles import engine_profile


@pytest.mark.parametrize('profile',['reference-cpu','language-cuda-math'])
def test_verified_profile_archive_round_trips_internal_alias_without_duplicate_language_library(tmp_path,profile):
    if sys.platform=='win32':pytest.skip('POSIX package alias fixture')
    selected=engine_profile(profile);root=tmp_path/'package';root.mkdir()
    name=selected['name'];(root/name).write_bytes(b'fixture only; must never execute')
    library=b'fixture-library'*1000;(root/'library').write_bytes(library)
    (root/'alias').symlink_to(root/'library')  # Absolute internal source paths must not leak into archives.
    files=[]
    for path in [root/name,root/'library',root/'alias']:
        files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={'version':1,'name':name,'platform':sys.platform,'machine':platform.machine(),'torch':'fixture',
        'operations':selected['operations'],'entrypoint':name,'files':files}
    (root/'engine-package.json').write_text(json.dumps(manifest))
    output=tmp_path/'engine.zip';result=archive_package(root,output,profile=profile)
    with zipfile.ZipFile(output) as zipped:
        alias=zipped.getinfo('alias')
        if selected['archive_links']:
            assert stat.S_ISLNK(alias.external_attr>>16) and zipped.read('alias')==b'library'
            assert result['preserved_file_aliases']==1
            assert result['installed_bytes']-result['regular_file_bytes']==len(library)
        else:
            assert not stat.S_ISLNK(alias.external_attr>>16) and zipped.read('alias')==library
            assert result['regular_file_bytes']==result['installed_bytes']
    data=output.read_bytes()
    class Response(io.BytesIO):status=200;headers={}
    entry=EngineDownload('https://example.invalid/engine.zip',result['sha256'],len(data),
        result['manifest_sha256'],result['installed_bytes'],profile=profile)
    command=prepare_engine(entry,tmp_path/'state',allowed=lambda:True,rate_mbps=lambda:1000000,
        opener=lambda *a,**kw:Response(data))
    installed=Path(command).parent
    assert (installed/'alias').read_bytes()==library
    assert (installed/'alias').is_symlink() is selected['archive_links']
    assert (installed/name).read_bytes()==(root/name).read_bytes()
    if profile!='reference-cpu':
        with pytest.raises(ValueError,match='incompatible'):archive_package(root,tmp_path/'wrong.zip')


def test_archive_rejects_output_inside_verified_source(tmp_path):
    with pytest.raises(ValueError,match='outside'):archive_package(tmp_path,tmp_path/'inside.zip')
