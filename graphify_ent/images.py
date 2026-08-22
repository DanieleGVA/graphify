"""Phase 1.3 — image pipeline: resize instead of drop, pHash dedup.

Two upstream defects this fixes (architecture doc §1, findings 4):

1. `graphify/llm.py` sets the image payload to None above `_MAX_IMAGE_BYTES`
   (5 MB) and falls back to a bare text reference — the vision model never sees
   the image. Here we re-encode (max side 2048 px, JPEG quality ladder) and only
   fall back if a re-encode still cannot fit.
2. No perceptual-hash dedup: photo-heavy corpora are 30–60 % near-duplicates,
   each paying a full vision call. Here a dHash + Hamming-distance clustering
   pass extracts one representative per cluster; the rest become nodes with a
   `duplicate_of` edge at zero vision cost.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_SIDE",
    "Cluster",
    "VisionPlan",
    "dedup_clusters",
    "fit_image_bytes",
    "plan_vision_calls",
]

#: Mirrors upstream `_MAX_IMAGE_BYTES` (5 MB).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
#: Execution plan §1.3: max side 2048 px.
DEFAULT_MAX_SIDE = 2048
#: JPEG quality ladder walked down until the payload fits.
_QUALITY_LADDER = (85, 75, 65, 55, 45, 35)
#: dHash Hamming distance at or below which two images are near-duplicates.
DEFAULT_HAMMING = 6


def fit_image_bytes(
    path: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[bytes, bool]:
    """Return (payload, was_resized) for an image that fits `max_bytes`.

    Never returns None: an image that cannot be made to fit is returned at the
    smallest attempted encoding rather than dropped, so it is still *seen*.
    """
    from PIL import Image

    raw = path.read_bytes()
    if len(raw) <= max_bytes:
        img = Image.open(io.BytesIO(raw))
        if max(img.size) <= max_side:
            return raw, False

    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB") if img.mode not in ("RGB", "L") else img

    # Step the long side down, and within each size walk the quality ladder.
    side = min(max(img.size), max_side)
    best: bytes | None = None
    while side >= 256:
        scaled = img.copy()
        scaled.thumbnail((side, side), Image.LANCZOS)
        for q in _QUALITY_LADDER:
            buf = io.BytesIO()
            scaled.save(buf, "JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            best = data if best is None or len(data) < len(best) else best
            if len(data) <= max_bytes:
                return data, True
        side //= 2

    return best or raw, True


@dataclass
class Cluster:
    """A set of perceptually near-identical images."""

    representative: Path
    members: list[Path] = field(default_factory=list)

    @property
    def duplicates(self) -> list[Path]:
        return [m for m in self.members if m != self.representative]


def _dhash(path: Path):
    from PIL import Image
    import imagehash

    with Image.open(path) as img:
        return imagehash.dhash(img)


def dedup_clusters(
    paths: list[Path], hamming_threshold: int = DEFAULT_HAMMING
) -> list[Cluster]:
    """Group images into near-duplicate clusters by dHash Hamming distance.

    Greedy single-pass clustering against cluster representatives: O(n·k) with
    k = number of clusters, which on a duplicate-heavy corpus is far below n.
    Every input appears in exactly one cluster — nothing is dropped.
    """
    clusters: list[Cluster] = []
    hashes: list = []

    for p in paths:
        try:
            h = _dhash(p)
        except Exception:
            # Unreadable image: it still gets its own cluster so it is never
            # silently unseen; the caller will attempt a vision call on it.
            clusters.append(Cluster(representative=p, members=[p]))
            hashes.append(None)
            continue

        placed = False
        for cluster, ch in zip(clusters, hashes):
            if ch is None:
                continue
            if (h - ch) <= hamming_threshold:
                cluster.members.append(p)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(representative=p, members=[p]))
            hashes.append(h)
    return clusters


@dataclass
class VisionPlan:
    """What the extraction pass should actually send to the vision model."""

    total_images: int
    representatives: list[Path]
    duplicate_of: dict[Path, Path]

    @property
    def vision_calls(self) -> int:
        return len(self.representatives)

    @property
    def duplicates_avoided(self) -> int:
        return len(self.duplicate_of)

    @property
    def reduction_pct(self) -> float:
        if not self.total_images:
            return 0.0
        return round(100 * self.duplicates_avoided / self.total_images, 2)

    def as_report(self) -> dict:
        return {
            "total_images": self.total_images,
            "vision_calls": self.vision_calls,
            "duplicates_avoided": self.duplicates_avoided,
            "reduction_pct": self.reduction_pct,
        }


def plan_vision_calls(
    paths: list[Path], hamming_threshold: int = DEFAULT_HAMMING
) -> VisionPlan:
    """Decide which images get a vision call and which become `duplicate_of` nodes."""
    clusters = dedup_clusters(list(paths), hamming_threshold)
    reps = [c.representative for c in clusters]
    dup_of = {d: c.representative for c in clusters for d in c.duplicates}
    return VisionPlan(total_images=len(paths), representatives=reps, duplicate_of=dup_of)


def env_max_bytes(default: int = DEFAULT_MAX_BYTES) -> int:
    raw = os.environ.get("GRAPHIFY_ENT_MAX_IMAGE_BYTES")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def extract_pdf_images(path: Path, out_dir: Path, min_bytes: int = 8_000) -> list[Path]:
    """Export embedded raster images from a PDF (the pilot's image source).

    The pilot corpus is PDF-only, so the image pipeline's real input is the
    photography embedded in the cookbooks rather than standalone files.
    """
    try:
        import pymupdf as _fitz
    except ImportError:  # pragma: no cover
        import fitz as _fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with _fitz.open(str(path)) as doc:
        seen: set[int] = set()
        for pno in range(doc.page_count):
            for info in doc.get_page_images(pno, full=True):
                xref = info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    img = doc.extract_image(xref)
                except Exception:
                    continue
                data = img.get("image") or b""
                if len(data) < min_bytes:
                    continue
                dest = out_dir / f"{path.stem[:40]}_p{pno:04d}_x{xref}.{img.get('ext', 'png')}"
                dest.write_bytes(data)
                written.append(dest)
    return written
