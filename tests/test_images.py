"""Phase 1.3 — image resize-not-drop + pHash dedup (TDD: written before implementation).

Acceptance (execution plan §1.3): on the pilot, 0 images silently unseen;
measured vision-call reduction from dedup reported.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL", reason="Pillow required")
pytest.importorskip("imagehash", reason="imagehash required")

from PIL import Image, ImageDraw  # noqa: E402

from graphify_ent.images import (  # noqa: E402
    DEFAULT_MAX_BYTES,
    dedup_clusters,
    fit_image_bytes,
    plan_vision_calls,
)


def _jpeg(path: Path, size=(4000, 3000), color=(180, 60, 40), noise: int = 0) -> Path:
    """A textured image. `noise` seeds the texture so two calls with the same
    seed produce identical content (duplicates) and different seeds produce
    different content."""
    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for i in range(0, size[0], max(1, size[0] // noise)):
            for j in range(0, size[1], max(1, size[1] // noise)):
                px[i, j] = ((i * 7 + j * 13) % 255, (j * 5) % 255, (i * 3) % 255)
    img.save(path, "JPEG", quality=95)
    return path


def _structured(path: Path, seed: int, size=(600, 450)) -> Path:
    """A structurally distinct image.

    dHash encodes *gradient structure*, deliberately ignoring colour — so a
    flat colour swatch is perceptually identical to any other swatch. Distinct
    fixtures must therefore differ in layout, the way real photographs do.
    """
    img = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    w, h = size
    rng = seed * 2654435761 % 2**32
    for k in range(6 + seed):
        rng = (rng * 1103515245 + 12345) % 2**31
        x0 = (rng % w) * 0.8
        rng = (rng * 1103515245 + 12345) % 2**31
        y0 = (rng % h) * 0.8
        box = [x0, y0, x0 + w * (0.1 + 0.05 * (k % 4)), y0 + h * (0.1 + 0.07 * (k % 3))]
        shade = (k * 37 % 200, (seed * 53 + k * 11) % 200, (k * 91) % 200)
        (d.ellipse if (k + seed) % 2 else d.rectangle)(box, fill=shade)
    img.save(path, "JPEG", quality=92)
    return path


class TestResizeNotDrop:
    def test_large_image_is_resized_not_dropped(self, tmp_path):
        """Upstream sets raw=None for >5MB images; they must be re-encoded instead."""
        big = _jpeg(tmp_path / "big.jpg", size=(6000, 4500), noise=400)
        assert big.stat().st_size > DEFAULT_MAX_BYTES

        data, was_resized = fit_image_bytes(big, max_bytes=DEFAULT_MAX_BYTES)
        assert data is not None, "image must never be silently dropped"
        assert was_resized is True
        assert len(data) <= DEFAULT_MAX_BYTES
        # The result is still a decodable image, not a truncated byte string.
        img = Image.open(io.BytesIO(data))
        img.verify()

    def test_small_image_passes_through_unmodified(self, tmp_path):
        small = _jpeg(tmp_path / "small.jpg", size=(320, 240))
        data, was_resized = fit_image_bytes(small, max_bytes=DEFAULT_MAX_BYTES)
        assert was_resized is False
        assert data == small.read_bytes()

    def test_max_side_is_respected(self, tmp_path):
        big = _jpeg(tmp_path / "big.jpg", size=(6000, 4500), noise=400)
        data, _ = fit_image_bytes(big, max_bytes=DEFAULT_MAX_BYTES, max_side=2048)
        img = Image.open(io.BytesIO(data))
        assert max(img.size) <= 2048

    def test_pathological_image_still_returns_bytes(self, tmp_path):
        """Even with a tiny cap, we return the best effort rather than None."""
        big = _jpeg(tmp_path / "big.jpg", size=(6000, 4500), noise=400)
        data, was_resized = fit_image_bytes(big, max_bytes=5_000, max_side=2048)
        assert data is not None and was_resized is True


class TestPhashDedup:
    def test_identical_images_cluster_together(self, tmp_path):
        a = _jpeg(tmp_path / "a.jpg", noise=50)
        b = _jpeg(tmp_path / "b.jpg", noise=50)  # byte-identical content
        clusters = dedup_clusters([a, b], hamming_threshold=6)
        assert len(clusters) == 1
        assert set(clusters[0].members) == {a, b}
        assert clusters[0].representative in (a, b)

    def test_near_duplicates_cluster_together(self, tmp_path):
        a = _jpeg(tmp_path / "a.jpg", size=(1200, 900), noise=30)
        # Same image re-encoded at a different quality/size: a near-duplicate.
        img = Image.open(a).resize((1100, 825))
        b = tmp_path / "b.jpg"
        img.save(b, "JPEG", quality=70)
        clusters = dedup_clusters([a, b], hamming_threshold=6)
        assert len(clusters) == 1, "near-duplicates must share a cluster"

    def test_distinct_images_do_not_cluster(self, tmp_path):
        a = _structured(tmp_path / "a.jpg", seed=1)
        b = _structured(tmp_path / "b.jpg", seed=9)
        clusters = dedup_clusters([a, b], hamming_threshold=6)
        assert len(clusters) == 2

    def test_every_image_appears_in_exactly_one_cluster(self, tmp_path):
        paths = [_structured(tmp_path / f"i{i}.jpg", seed=i + 1) for i in range(6)]
        clusters = dedup_clusters(paths, hamming_threshold=6)
        seen = [p for c in clusters for p in c.members]
        assert sorted(seen) == sorted(paths), "no image may be silently unseen"
        assert len(seen) == len(set(seen)), "no image may appear twice"


class TestVisionCallPlanning:
    def test_duplicates_cost_zero_vision_calls(self, tmp_path):
        a = _jpeg(tmp_path / "a.jpg", noise=50)
        b = _jpeg(tmp_path / "b.jpg", noise=50)
        c = _structured(tmp_path / "c.jpg", seed=7)

        plan = plan_vision_calls([a, b, c], hamming_threshold=6)
        assert plan.total_images == 3
        assert plan.vision_calls == 2, "one call per cluster, not per image"
        assert plan.duplicates_avoided == 1
        assert 0 < plan.reduction_pct < 100

        # Every non-representative carries a duplicate_of edge target, and a
        # duplicate is never itself a representative (the two sets are disjoint).
        assert not (set(plan.duplicate_of) & set(plan.representatives))
        for dup, rep in plan.duplicate_of.items():
            assert dup != rep
            assert rep in plan.representatives

    def test_no_duplicates_means_no_reduction(self, tmp_path):
        paths = [_structured(tmp_path / f"d{i}.jpg", seed=(i + 1) * 3) for i in range(3)]
        plan = plan_vision_calls(paths, hamming_threshold=6)
        assert plan.vision_calls == plan.total_images
        assert plan.duplicates_avoided == 0
        assert plan.reduction_pct == 0.0

    def test_empty_input_is_safe(self):
        plan = plan_vision_calls([], hamming_threshold=6)
        assert plan.total_images == 0 and plan.vision_calls == 0
        assert plan.reduction_pct == 0.0
