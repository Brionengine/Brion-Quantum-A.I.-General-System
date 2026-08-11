# Optional dependencies

**You can clone, read, import and test this repository without installing the
heavy scientific stack.** numpy, torch, tensorflow, qiskit, cirq, Bio and the
rest are large, slow to build and frequently awkward to install on a fresh
machine, so nothing here requires them merely to import.

## How it works

Every heavy third-party import is wrapped:

```python
try:
    import numpy as np
except ImportError:  # optional dependency: pip install numpy
    np = None
```

The name is always bound, so the module imports either way. Code that genuinely
needs the package fails at the point of use, not at import time.

Modules that use an optional package in a type annotation also carry:

```python
from __future__ import annotations
```

Annotations then stay unevaluated strings, so a signature such as
`def f() -> np.ndarray:` does not blow up at definition time when numpy is
absent. Without it, a guarded import that binds `np = None` still raises
`AttributeError: 'NoneType' object has no attribute 'ndarray'` the moment the
`def` is executed.

## What this means for a fork

- `git clone` then `python -c "import <module>"` works with **no** third-party
  installs.
- Test suites that exercise pure logic run on a bare checkout.
- Install only the packages the parts you actually use require.

## Checking a package is present

```python
if np is not None:
    result = np.mean(values)
else:
    result = sum(values) / len(values)   # fallback path
```

## Adding new code

Two rules keep a bare checkout working:

1. **Guard the import.** Wrap any new heavy dependency in the `try` / `except
   ImportError` form above, and bind the name to `None` in the fallback.
2. **Do not use it at import time.** Anything that runs while the module is
   being imported — a module-level constant, a class attribute, a default
   argument, a base class — must not touch an optional package. Move that work
   inside a function, a `@property`, or a lazily-populated cache.

Rule 2 is the one that bites. A class body executes on import, so

```python
class Gates:
    X = np.array([[0, 1], [1, 0]])       # runs at import: breaks a bare checkout
```

needs to become lazy — build it on first access and cache it.

## Known limitations

A few modules still need their dependency in order to import, because they
execute dependency-requiring code at import time — most often a class that
inherits from `torch.nn.Module`, since a base class is evaluated when the class
statement runs and cannot be deferred. These raise a clear error naming the
missing package. Everything else in the repository imports without it.
