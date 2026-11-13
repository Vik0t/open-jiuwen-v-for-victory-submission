"""Locate a single explicitly named Python function in the public source."""
import ast
from pathlib import Path
import re


def initial_range(root, files, instruction):
    """Return (relative file, character start, length) for an unambiguous anchor."""
    words = {name for name in re.findall(r"\b[A-Za-z_]\w*\b", instruction)
             if len(name) >= 6 and ("_" in name or re.search("[a-z][A-Z]", name))}
    # Plain lowercase names need an explicit repair target and file, or an
    # explicit function declaration. Ordinary prose must not become an anchor.
    target = r"([a-z][a-z0-9]{5,})"
    verb = r"(?i:repair|fix|modify|update|implement)"
    plain_paths = {}
    for match in re.finditer(r"\b" + verb + r"\s+`?" + target
                             + r"`?\s+(?i:in)\s+`?([\w./-]+\.py)`?(?!\w)", instruction):
        plain_paths.setdefault(match.group(1), set()).add(match.group(2))
    plain_words = set(re.findall(r"\b" + verb + r"\s+(?i:(?:the\s+)?function)\s+`?"
                                 + target + r"`?(?!\w|\.py\b)", instruction))
    words.update(plain_paths)
    words.update(plain_words)
    if not words:
        return None
    paths = []
    total_bytes = 0
    for name in files:
        relative = Path(name)
        if (relative.suffix != ".py" or relative.name.startswith("test")
                or relative.name.endswith("_test.py") or "tests" in relative.parts
                or relative.name == "conftest.py"):
            continue
        path = root / relative
        size = path.stat().st_size
        total_bytes += size
        paths.append((name, path, size))
    if total_bytes > 512_000 or len(paths) > 32 or not any(size >= 20_000 for _, _, size in paths):
        return None

    matches = []
    for name, path, size in paths:
        text = path.read_bytes().decode("utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in words:
                allowed = (node.name not in plain_paths or node.name in plain_words
                           or name in plain_paths[node.name])
                matches.append((name, node, text) if size >= 20_000 and allowed else None)
                if len(matches) > 1:
                    return None
    if len(matches) != 1 or matches[0] is None:
        return None
    name, node, text = matches[0]
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    length = sum(map(len, lines[start:node.end_lineno]))
    if length > 4500 or start < 1:
        return None
    return name, sum(map(len, lines[:start])), length
