"""Backward-compatible shim for the legacy `graphify` package.

The project has been renamed to **Enterpriphy** (PyPI `enterpriphy`).
Existing code importing `graphify` or `graphify.<submodule>` continues to
work transparently — every attribute and submodule access is routed to the
corresponding `enterpriphy` symbol. A `DeprecationWarning` is emitted once
per process to signal the migration.

This shim is scheduled for removal in Enterpriphy 2.0.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import enterpriphy as _enterpriphy

_DEPRECATION_MESSAGE = (
    "The `graphify` package has been renamed to `enterpriphy`. "
    "Update your imports (`from enterpriphy import ...`) — the `graphify` "
    "name will be removed in Enterpriphy 2.0. "
    "See https://github.com/DanieleGVA/enterpriphy/blob/main/MIGRATION_PLAN.md"
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


class _GraphifyShimLoader(importlib.abc.Loader):
    """Loader that returns an enterpriphy submodule under the graphify name."""

    def create_module(self, spec):
        suffix = spec.name[len("graphify."):]
        target = f"enterpriphy.{suffix}"
        return importlib.import_module(target)

    def exec_module(self, module):  # already executed during create_module
        return None


class _GraphifyShimFinder(importlib.abc.MetaPathFinder):
    """Routes `import graphify.X` / `from graphify.X import Y` to enterpriphy.X."""

    _loader = _GraphifyShimLoader()

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("graphify."):
            return None
        suffix = fullname[len("graphify."):]
        if not suffix or "." in suffix:
            # Nested submodules: defer to the standard machinery operating on
            # the underlying enterpriphy package once the parent has been
            # aliased; this finder only intervenes for first-level submodules.
            return None
        target_name = f"enterpriphy.{suffix}"
        try:
            importlib.util.find_spec(target_name)
        except (ImportError, ValueError):
            return None
        return importlib.util.spec_from_loader(fullname, self._loader)


if not any(isinstance(f, _GraphifyShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _GraphifyShimFinder())


def __getattr__(name: str):
    # Attribute access on the package object: try a submodule first, then a
    # top-level attribute on enterpriphy (covers its lazy __getattr__ exports).
    try:
        return importlib.import_module(f"graphify.{name}")
    except ModuleNotFoundError:
        try:
            return getattr(_enterpriphy, name)
        except AttributeError as exc:
            raise AttributeError(
                f"module 'graphify' has no attribute {name!r} "
                f"(forwarded to 'enterpriphy')"
            ) from exc


try:
    from enterpriphy import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"
