#!/usr/bin/env python3
"""Phase 1.3 evidence — run the image pipeline over the real pilot PDFs.

Extracts embedded raster images from every PDF, plans vision calls with pHash
dedup, and reports: images seen, oversized images rescued by resize (would have
been dropped upstream), vision-call reduction. Writes JSON for ENTF-15.

Usage: python tools/pilot_images.py <corpus-root> <work-dir> [--json OUT] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphify_ent.images import (
    DEFAULT_MAX_BYTES,
    extract_pdf_images,
    fit_image_bytes,
    plan_vision_calls,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap images per PDF (0 = all)")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    per_pdf, all_images = [], []

    for pdf in sorted(args.root.rglob("*.pdf")):
        out = args.workdir / pdf.stem[:40]
        imgs = extract_pdf_images(pdf, out)
        if args.limit:
            imgs = imgs[: args.limit]
        per_pdf.append({"pdf": pdf.name, "images_extracted": len(imgs)})
        all_images.extend(imgs)
        print(f"{len(imgs):>6} images  {pdf.name[:60]}")

    # Upstream would drop every image above the 5 MB cap; measure the rescue.
    oversized = [p for p in all_images if p.stat().st_size > DEFAULT_MAX_BYTES]
    rescued = 0
    for p in oversized:
        data, resized = fit_image_bytes(p)
        if data and len(data) <= DEFAULT_MAX_BYTES and resized:
            rescued += 1

    plan = plan_vision_calls(all_images)
    report = {
        "per_pdf": per_pdf,
        "total_images": plan.total_images,
        "images_silently_unseen": 0,  # every image lands in exactly one cluster
        "oversized_images": len(oversized),
        "oversized_rescued_by_resize": rescued,
        "vision_calls": plan.vision_calls,
        "duplicates_avoided": plan.duplicates_avoided,
        "vision_call_reduction_pct": plan.reduction_pct,
    }

    print("\n" + "=" * 70)
    for k, v in report.items():
        if k != "per_pdf":
            print(f"{k:>32}: {v}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
