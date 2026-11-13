"""Avoid ctypes just to obtain C long's size; keep the executor guard intact.

This is a local, explicit patch to SymPy 1.14.0, not an upstream release.
Native struct 'l' uses the interpreter's C long size, as ctypes.c_long does.
"""
import hashlib
from pathlib import Path
import struct

path = Path('/usr/local/lib/python3.12/site-packages/sympy/external/gmpy.py')
source = path.read_bytes()
expected = 'f3dbd7735283e53890093c252fef7caf4d4a76ab0eb09a7373c5cdd51ed3c0ce'
if hashlib.sha256(source).hexdigest() != expected:
    raise ValueError('Unexpected SymPy source; review this patch for the new version')
text = source.decode()
assert text.count('from ctypes import c_long, sizeof') == 1
assert text.count('sizeof(c_long)') == 1
text = text.replace('from ctypes import c_long, sizeof', 'from struct import calcsize')
text = text.replace('sizeof(c_long)', 'calcsize("l")')
path.write_text(text)
print('SymPy local patch applied: native C long bytes =', struct.calcsize('l'))
