"""Locally selected, fail-closed Linux CPU worker filesystem/network boundary.

This is an explicit supervisor option, not a peer request field. It does not
enable unknown workers, enforce disk/cgroup quotas, or sandbox CUDA jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys


class ComputeSandboxError(RuntimeError):
    pass


SYSTEM_ROOTS = (Path('/usr'), Path('/bin'), Path('/sbin'), Path('/lib'), Path('/lib64'))


def _canonical(path, *, directory):
    path = Path(path)
    if (not path.is_absolute() or path != path.resolve(strict=True)
            or not (path.is_dir() if directory else path.is_file())):
        raise ComputeSandboxError('Sandbox mounts require canonical existing local paths')
    # A caller may expose one specific package, input file or job, never a
    # broad collection of operator state. System runtime mounts are fixed below.
    if path in {Path('/'), Path('/home'), Path('/tmp'), Path('/var'), Path.home().resolve()}:
        raise ComputeSandboxError('Sandbox mount is too broad')
    return path


@dataclass(frozen=True)
class CPUComputeSandbox:
    runtime_roots: tuple[Path, ...]
    input_files: tuple[Path, ...]
    output_directory: Path

    def launch(self, command, request):
        if sys.platform != 'linux' or request.get('device') != 'cpu':
            raise ComputeSandboxError('This sandbox requires a Linux CPU job')
        executable = shutil.which('bwrap', path='/usr/bin:/bin')
        if executable is None:
            raise ComputeSandboxError('Required Linux worker isolation is unavailable')
        if (not isinstance(command, (list, tuple)) or not command
                or not all(isinstance(arg, str) and arg and '\x00' not in arg for arg in command)
                or not Path(command[0]).is_absolute()):
            raise ComputeSandboxError('Sandbox requires a locally selected absolute executable')
        if not 1 <= len(self.runtime_roots) <= 8 or len(self.input_files) > 64:
            raise ComputeSandboxError('Sandbox mounts exceed their local bound')
        roots = tuple(_canonical(p, directory=True) for p in self.runtime_roots)
        inputs = tuple(_canonical(p, directory=False) for p in self.input_files)
        output = _canonical(self.output_directory, directory=True)
        # RW must never cover a package/input. Conversely, hiding a RW job
        # inside a RO package changes the declared policy and is rejected.
        if any(p.is_relative_to(output) or output.is_relative_to(p) for p in (*roots, *inputs)):
            raise ComputeSandboxError('Sandbox read and write mounts overlap')
        program = Path(command[0]).resolve(strict=True)
        if not program.is_file() or not any(program.is_relative_to(p.resolve()) for p in (*SYSTEM_ROOTS, *roots)):
            raise ComputeSandboxError('Sandbox executable is outside its selected runtime')
        args = [executable, '--unshare-user', '--unshare-pid', '--unshare-net', '--unshare-ipc',
                '--unshare-uts', '--unshare-cgroup-try', '--disable-userns', '--assert-userns-disabled',
                '--cap-drop', 'ALL', '--die-with-parent', '--new-session', '--clearenv']
        for p in SYSTEM_ROOTS:
            if p.exists():
                args += ['--ro-bind', str(p), str(p)]
        loader_cache = Path('/etc/ld.so.cache')
        if loader_cache.is_file():
            args += ['--ro-bind', str(loader_cache), str(loader_cache)]
        args += ['--proc', '/proc', '--remount-ro', '/proc', '--dev', '/dev',
                 '--tmpfs', '/tmp', '--tmpfs', '/dev/shm', '--dir', '/nonexistent']
        for p in (*roots, *inputs):
            args += ['--ro-bind', str(p), str(p)]
        args += ['--bind', str(output), str(output), '--chdir', str(output), '--remount-ro', '/']
        environment = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'TMPDIR': '/tmp',
                       'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        for key, value in environment.items():
            args += ['--setenv', key, value]
        # Resolve the executable so a host-side symlink does not require exposing
        # another directory. Other arguments are data, never shell fragments.
        return [*args, '--', str(program), *command[1:]], environment
