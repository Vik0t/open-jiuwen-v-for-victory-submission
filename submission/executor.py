"""Bounded Python execution inside the contest's offline, read-only container.

The audit hook catches ordinary unsafe Python operations. It is not an OS
sandbox: the caller must provide container isolation and no mounted secrets.
"""
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MAX_OUTPUT_BYTES = 8192
MAX_CODE_BYTES = 100_000


def protect_parent():
    """On Linux, protect this trusted process before starting untrusted children.

    Requires the container's dropped capabilities. An installation or readback
    error stops the caller; Python's child audit is still not an OS sandbox.
    """
    if sys.platform != 'linux':
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE=0
        raise OSError(ctypes.get_errno(), 'Cannot protect parent process')
    if libc.prctl(3, 0, 0, 0, 0) != 0:  # PR_GET_DUMPABLE must confirm zero.
        raise RuntimeError('Parent process protection is not active')


RUNNER = r'''
def run():
    import os, pathlib, resource, sys, types, linecache, traceback
    root = pathlib.Path(sys.argv[1]).resolve()
    seconds = int(sys.argv[2])
    code = sys.stdin.read()
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stdlib = pathlib.Path(os.__file__).resolve().parent
    protected = frozenset({'test', 'tests', '__tests__', 'metadata', 'answers'})

    def check_path(name, writing=False):
        if isinstance(name, int):
            if name not in (0, 1, 2):
                raise PermissionError('File descriptors are restricted')
            return
        path = pathlib.Path(os.fsdecode(name)).resolve()
        inside = path.is_relative_to(root)
        if not writing:
            if not inside and not path.is_relative_to(stdlib):
                raise PermissionError('Read only workspace or standard library files')
            return
        if not inside:
            raise PermissionError('Write only inside workspace')
        parts = tuple(part.lower() for part in path.relative_to(root).parts)
        filename = parts[-1] if parts else ''
        if (not parts or any(part.startswith('.') or part in protected for part in parts)
                or filename.startswith('test') or filename.endswith('_test.py')
                or filename in {'conftest.py', 'task.json', 'answer.txt'}):
            raise PermissionError('Tests, answers and metadata are read-only')

    def audit(event, args):
        if (event.startswith(('socket.', 'subprocess.', 'ctypes.', 'os.exec', 'os.spawn',
                              'os.posix_spawn', 'os.fork'))
                or event in {'os.system', 'os.kill', 'os.killpg', 'os.chdir',
                             'os.link', 'os.symlink', 'os.putenv', 'os.unsetenv', 'os.chmod', 'os.chown'}):
            raise PermissionError('Network, subprocesses and unsafe operations are disabled')
        if event == 'import' and args[0] in {'ctypes', '_ctypes', 'cffi', '_cffi_backend'}:
            raise PermissionError('Native process access is disabled')
        if event == 'open':
            if args[1] is None:
                raise PermissionError('Use Python file objects, not raw file descriptors')
            check_path(args[0], bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                                              | os.O_TRUNC | os.O_APPEND)))
        if event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown',
                     'os.truncate', 'os.utime', 'os.rename'}:
            for name in args[:2] if event == 'os.rename' else args[:1]:
                check_path(name, True)

    sys.addaudithook(audit)
    sys.path.insert(0, str(root))
    module = types.ModuleType('__main__')
    module.__file__ = str(root / '<agent-python>')
    linecache.cache[module.__file__] = (len(code.encode('utf-8')), None,
                                      code.splitlines(keepends=True), module.__file__)
    sys.excepthook = traceback.print_exception
    sys.modules['__main__'] = module
    sys.argv = ['<agent-python>']
    exec(compile(code, module.__file__, 'exec'), module.__dict__)

run()
'''


def execute_python(workspace, code, timeout=10):
    """Return CompletedProcess; 126 refuses execution, 124 means timeout."""
    def result(returncode, stdout='', stderr=''):
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    if sys.platform != 'linux' or not Path('/.dockerenv').is_file():
        return result(126, stderr='Python execution requires the offline contest container')
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        return result(126, stderr='Workspace must be an existing directory, not a symlink')
    if not isinstance(code, str) or len(code.encode('utf-8')) > MAX_CODE_BYTES:
        return result(126, stderr='Code must be text within the byte limit')
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        return result(126, stderr='Timeout must be between zero and sixty seconds')
    command = [sys.executable, '-I', '-B', '-c', RUNNER, str(root.resolve()), str(math.ceil(timeout))]
    try:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.run(command, input=code.encode('utf-8'), cwd=root.resolve(),
                                         env={'PATH': os.defpath, 'LANG': 'C.UTF-8'},
                                         stdout=output, stderr=errors, timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            for stream in (output, errors):
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - MAX_OUTPUT_BYTES))
            stdout = output.read(MAX_OUTPUT_BYTES).decode('utf-8', errors='replace')
            stderr = errors.read(MAX_OUTPUT_BYTES).decode('utf-8', errors='replace')
            return result(returncode, stdout, stderr + ('\nExecution timeout' if returncode == 124 else ''))
    except OSError as error:
        return result(126, stderr=f'Execution failed: {error.strerror or type(error).__name__}')
