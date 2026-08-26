"""Where the encoder runs, and why it is not left to chance.

Embedding 53k nodes took an hour on CPU and minutes on the platform's
accelerator — long enough that "re-embed after a model change" stopped being a
step anyone would take casually, which is how a stale index survives a model
swap. The selection is therefore explicit, overridable, and must never raise:
an environment where torch is missing or broken has to degrade to the library
default, not fail the import.
"""

from __future__ import annotations

import builtins

import pytest

from graphify_ent.embed import _device


class TestDeviceSelection:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("EMBED_DEVICE", "cpu")
        assert _device() == "cpu"

    def test_returns_a_supported_value(self, monkeypatch):
        monkeypatch.delenv("EMBED_DEVICE", raising=False)
        assert _device() in {None, "cpu", "cuda", "mps"}

    def test_missing_torch_degrades_to_library_default(self, monkeypatch):
        """No torch must mean 'let sentence-transformers decide', not a crash."""
        monkeypatch.delenv("EMBED_DEVICE", raising=False)
        real_import = builtins.__import__

        def no_torch(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        assert _device() is None

    def test_broken_accelerator_probe_does_not_propagate(self, monkeypatch):
        """Some torch builds raise from the availability probe itself."""
        monkeypatch.delenv("EMBED_DEVICE", raising=False)
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert _device() is None
