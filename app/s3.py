"""boto3 S3 manager — singleton, presigned URLs, mirrors djangoblunderfolder style."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config as BotoConfig

from app.config import CFG
from app.logging_config import get_logger

log = get_logger("s3")


class S3Manager:
    def __init__(self) -> None:
        cfg = CFG.s3
        self.bucket = cfg.bucket
        self.prefix = cfg.key_prefix.strip("/")
        self.expiry = cfg.presign_expiry

        self.client = boto3.client(
            "s3",
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            region_name=cfg.region,
            config=BotoConfig(signature_version="s3v4"),
        )
        log.info("🪣 S3Manager ready bucket=%s prefix=%s region=%s", self.bucket, self.prefix, cfg.region)

    def _key(self, subpath: str, filename: str) -> str:
        safe = f"{uuid.uuid4().hex[:12]}_{Path(filename).name}"
        return f"{self.prefix}/{subpath.strip('/')}/{safe}"

    def upload_bytes(self, data: bytes, filename: str, subpath: str = "", content_type: str = "image/png") -> str:
        key = self._key(subpath or "designs", filename)
        log.info("⬆️  upload_bytes bucket=%s key=%s bytes=%d", self.bucket, key, len(data))
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def presigned_download(self, key: str, expiry: Optional[int] = None) -> str:
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry or self.expiry,
        )
        log.info("🔗 presigned_download key=%s expiry=%ds", key, expiry or self.expiry)
        return url


_manager: Optional[S3Manager] = None


def get_s3() -> S3Manager:
    global _manager
    if _manager is None:
        _manager = S3Manager()
    return _manager
