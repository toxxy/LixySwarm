"""Freeze the offline Qwen language worker; model files are acquired separately."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-policy',action='store_true');parser.add_argument('--half-policy',action='store_true')
    parser.add_argument('--math-policy',action='store_true')
    parser.add_argument('--eager-only',action='store_true',help='Exclude the optional Triton compiler from the fixed eager CUDA worker')
    args=parser.parse_args()
    if args.eager_only and args.device!='cuda':raise ValueError('The eager-only size profile is for the CUDA worker')
    if args.math_policy:args.half_policy=True
    if args.half_policy:args.batch_policy=True
    if args.batch_policy and args.device!='cuda':raise ValueError('The batch training profile requires CUDA')
    import torch
    if (torch.version.cuda is not None)!=(args.device=='cuda'):raise ValueError('Use the matching isolated CPU/CUDA runtime')
    repo=Path(__file__).resolve().parents[1];output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    licenses=output/'licenses';licenses.mkdir(exist_ok=True)
    for distribution in importlib.metadata.distributions(path=list({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})):
        for file in distribution.files or []:
            if any(word in str(file).lower() for word in ['license','copying','copyright']):
                source=Path(distribution.locate_file(file))
                if source.is_file():
                    target=licenses/distribution.metadata['Name']/str(file).replace('..','_')
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    name='LixySwarmLanguageCUDA' if args.device=='cuda' else 'LixySwarmLanguageCPU'
    command=[sys.executable,'-m','PyInstaller','--clean','--noconfirm','--onedir','--name',name,
        '--distpath',str(output/'dist'),'--workpath',str(output/'work'),'--specpath',str(output),'--paths',str(repo),
        '--add-data',f'{licenses}{os.pathsep}licenses']
    for module in ['transformers.models.qwen3.modeling_qwen3','transformers.models.qwen3.configuration_qwen3',
                   'transformers.models.qwen2.tokenization_qwen2','safetensors.torch','accelerate']:
        command+=['--hidden-import',module]
    for package in ['transformers','tokenizers','safetensors','accelerate']:
        command+=['--copy-metadata',package]
    for module in ['PySide6','torchvision','torchaudio','matplotlib','pandas','scipy','sklearn','IPython','tensorboard',
                   'src.network','src.matriarca','src.swarm','desktop']:
        command+=['--exclude-module',module]
    if args.eager_only:command+=['--exclude-module','triton']
    command.append(str(repo/'engines/language_main.py'))
    environment=dict(os.environ);environment.pop('PYTHONPATH',None)
    subprocess.run(command,cwd=repo,env=environment,check=True)
    package=output/'dist'/name;files=[]
    for path in sorted(package.rglob('*')):
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
            files.append({'path':path.relative_to(package).as_posix(),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    manifest={'version':1,'name':name,'platform':sys.platform,'machine':platform.machine(),'torch':torch.__version__,
        'operations':(['language.inference.v1','language.gradient.v2','language.candidate.v1','language.gradient.v3','language.candidate.v2'] if args.batch_policy else ['language.inference.v1','language.gradient.v2','language.candidate.v1'] if args.device=='cuda'
                      else ['language.inference.v1']),
        'entrypoint':name+('.exe' if sys.platform=='win32' else ''),'files':files}
    if args.half_policy:manifest['operations']+=['language.gradient.v4','language.candidate.v3']
    if args.math_policy:manifest['operations']+=['language.gradient.v5','language.candidate.v4']
    (package/'engine-package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'device':args.device,'eager_only':args.eager_only,'excluded_optional_compilers':['triton'] if args.eager_only else [],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'manifest_sha256':hashlib.sha256((package/'engine-package.json').read_bytes()).hexdigest(),
        'logical_payload_bytes':sum(row['bytes'] for row in files),
        'regular_payload_bytes':sum(row['bytes'] for row in files if not (package/row['path']).is_symlink())}
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(row['bytes'] for row in files)}))


if __name__=='__main__':main()
