"""MinIO / S3 object store for original blobs and normalized renderings.

Bucket layout (created by ``docker-compose up`` via the
``createbuckets`` job; see ``docker-compose.yml``):

- ``enterpriphy-blobs``        — original files keyed by content_hash
- ``enterpriphy-renderings``   — PDF→page PNG, DOCX→HTML, etc.
- ``enterpriphy-thumbnails``   — small previews for the UI
- ``enterpriphy-tables-parquet`` — large XLSX tables as queryable Parquet

Keys are derived from ``content_hash`` to make storage idempotent and
deduplicated by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

DEFAULT_BUCKET = "enterpriphy-blobs"
DEFAULT_RENDERINGS_BUCKET = "enterpriphy-renderings"
DEFAULT_THUMBNAILS_BUCKET = "enterpriphy-thumbnails"
DEFAULT_TABLES_BUCKET = "enterpriphy-tables-parquet"


def blob_key(content_hash: str, *, prefix: str = "", ext: str = "") -> str:
    """Two-level fanout for filesystem-friendly distribution.

    Yields keys like ``ab/cd/abcd...e.pdf``. The two-character prefixes keep
    any one S3 partition / MinIO directory from getting too wide.
    """
    if len(content_hash) < 4:
        raise ValueError("content_hash too short to derive a blob key")
    head = f"{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"
    if ext:
        head = f"{head}.{ext.lstrip('.')}"
    return f"{prefix.rstrip('/')}/{head}" if prefix else head


@dataclass
class BlobRef:
    """Pointer to a blob in the object store."""

    bucket: str
    key: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    etag: Optional[str] = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class BlobStore:
    """Thin façade over boto3 / minio. Concrete impl lives in `_s3` (P3)."""

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        *,
        region: str = "us-east-1",
        default_bucket: str = DEFAULT_BUCKET,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.default_bucket = default_bucket

    # --- core operations ------------------------------------------------

    def put_path(
        self,
        path: Path,
        *,
        content_hash: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> BlobRef:
        """Upload a local file. Returns a stable BlobRef keyed by hash."""
        raise NotImplementedError("BlobStore.put_path — implemented in P3")

    def put_bytes(
        self,
        data: bytes,
        *,
        content_hash: str,
        bucket: Optional[str] = None,
        ext: str = "",
        content_type: Optional[str] = None,
    ) -> BlobRef:
        raise NotImplementedError("BlobStore.put_bytes — implemented in P3")

    def get_stream(self, ref: BlobRef) -> IO[bytes]:
        raise NotImplementedError("BlobStore.get_stream — implemented in P3")

    def presigned_url(self, ref: BlobRef, *, expires_seconds: int = 900) -> str:
        raise NotImplementedError("BlobStore.presigned_url — implemented in P3")

    def exists(self, ref: BlobRef) -> bool:
        raise NotImplementedError("BlobStore.exists — implemented in P3")


__all__ = [
    "BlobRef",
    "BlobStore",
    "DEFAULT_BUCKET",
    "DEFAULT_RENDERINGS_BUCKET",
    "DEFAULT_THUMBNAILS_BUCKET",
    "DEFAULT_TABLES_BUCKET",
    "blob_key",
]
