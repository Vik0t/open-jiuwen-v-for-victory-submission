"""Small, byte-preserving editors for Python source files in a task workspace.

These functions validate syntax but never execute the supplied source.  Answers
must go through the separate submission tool; tests and metadata are read-only.
"""

import os
from pathlib import Path
import stat
import tempfile


MAX_SOURCE_BYTES = 100_000
MAX_EDIT_SOURCE_BYTES = 1_048_576
MAX_EDIT_GROWTH_BYTES = 100_000
PROTECTED_DIRECTORIES = {"test", "tests", "__tests__", "answers", "metadata"}


def _source_path(workspace, name):
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Workspace must be an existing directory, not a symlink")
    root = root.resolve()
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Use a relative Python file path")
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Path must stay inside the workspace")
    if any(ord(char) < 32 for char in name):
        raise ValueError("Control characters are not allowed in paths")
    if relative.suffix != ".py":
        raise ValueError("Only Python source files can be edited")
    if any(part.startswith(".") or part.lower() in PROTECTED_DIRECTORIES
           for part in relative.parts[:-1]):
        raise ValueError("Tests and metadata are read-only")
    filename = relative.name.lower()
    if filename.startswith("test") or filename.endswith("_test.py") or filename == "conftest.py":
        raise ValueError("Tests are read-only")
    target = root
    for part in relative.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("Symlinks are not allowed in source paths")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path must stay inside the workspace")
    if target.exists() and not target.is_file():
        raise ValueError("Source path must name a regular file")
    return target


def _read_source(target):
    with target.open("rb") as stream:
        data = stream.read(MAX_EDIT_SOURCE_BYTES + 1)
    if len(data) > MAX_EDIT_SOURCE_BYTES:
        raise ValueError(f"Source exceeds {MAX_EDIT_SOURCE_BYTES} bytes")
    return data.decode("utf-8"), len(data)


def _store_source(target, name, content, *, byte_limit=MAX_SOURCE_BYTES):
    if not isinstance(content, str):
        raise ValueError("Source must be text")
    data = content.encode("utf-8")
    if len(data) > byte_limit:
        raise ValueError(f"Source exceeds {byte_limit} bytes")
    compile(data, name, "exec")
    existed = target.exists()
    if existed:
        with target.open("rb") as stream:
            if stream.read(byte_limit + 1) == data:
                return {"success": True, "changed": False, "path": name, "bytes": len(data)}
    mode = stat.S_IMODE(target.stat().st_mode) if existed else 0o644
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".source-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"success": True, "changed": True, "created": not existed,
            "path": name, "bytes": len(data)}


def _failure(name, error):
    if isinstance(error, SyntaxError):
        message = f"Invalid Python at line {error.lineno}: {error.msg}"
    elif isinstance(error, OSError):
        message = f"File operation failed: {error.strerror or type(error).__name__}"
    else:
        message = str(error)
    return {"success": False, "changed": False, "path": name, "error": message}


def apply_edit(workspace, path, before=None, after=None, *, changes=None):
    """Apply 1-8 exact replacements in memory, then store the file once."""
    try:
        target = _source_path(workspace, path)
        batch = changes is not None
        if not batch:
            changes = [{"before": before, "after": after}]
        if not isinstance(changes, list) or not 1 <= len(changes) <= 8:
            raise ValueError("changes must be a list of 1-8 replacements for one file")
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError("Each change must contain before and after text")
            if (not isinstance(change.get("before"), str) or not change["before"]
                    or not isinstance(change.get("after"), str)):
                raise ValueError("before must be nonempty text and after must be text")
        source, source_bytes = _read_source(target)
        byte_limit = min(MAX_EDIT_SOURCE_BYTES, source_bytes + MAX_EDIT_GROWTH_BYTES)
        for number, change in enumerate(changes, 1):
            before, after = change["before"], change["after"]
            prefix = f"Change {number}: " if batch else ""
            start = source.find(before)
            if start < 0:
                raise ValueError(prefix + "The before fragment was not found; read the file again")
            if source.find(before, start + 1) >= 0:
                raise ValueError(prefix + "The before fragment is ambiguous; include more surrounding text")
            if len(after) > byte_limit:
                raise ValueError(prefix + f"Patched source exceeds {byte_limit} bytes")
            source_bytes += len(after.encode("utf-8")) - len(before.encode("utf-8"))
            if source_bytes > byte_limit:
                raise ValueError(prefix + f"Patched source exceeds {byte_limit} bytes")
            source = source[:start] + after + source[start + len(before):]
        return _store_source(target, path, source, byte_limit=byte_limit)
    except (OSError, ValueError, UnicodeError, SyntaxError, RecursionError) as error:
        return _failure(path, error)


def write_source(workspace, path, content):
    """Create or replace a Python source file; do not alter tests or metadata."""
    try:
        target = _source_path(workspace, path)
        if target.exists() and target.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"Source exceeds {MAX_SOURCE_BYTES} bytes")
        return _store_source(target, path, content)
    except (OSError, ValueError, UnicodeError, SyntaxError, RecursionError) as error:
        return _failure(path, error)
