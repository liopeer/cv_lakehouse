#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A multipart upload to S3 that a later run finds again, and continues.

S3 keeps every part of an upload that is not completed, and lists them. A run that dies
loses only the part that it was sending. The next run lists the parts, and continues at
the byte after the last one. No state is kept outside of S3, so a new pod resumes the
upload of the one before.

Every part but the last has the same size, so the parts alone give the offset. An
upload whose parts do not have that size is not ours to continue, and is aborted.
"""

from typing import Any, cast

from s3fs import S3FileSystem
from upath import UPath

MIB = 1024 * 1024
MIN_PART_SIZE = 64 * MIB
# S3 allows 10,000 parts. One stays free for the last part, which is shorter.
MAX_FULL_PARTS = 9_999


def choose_part_size(size: int) -> int:
    """Return the part size for `size` bytes: whole MiB, and few enough parts."""
    needed = -(-size // MAX_FULL_PARTS)
    return max(MIN_PART_SIZE, -(-needed // MIB) * MIB)


class ResumableUpload:
    """One multipart upload to the key of `target`."""

    def __init__(self, *, target: UPath, part_size: int) -> None:
        if not isinstance(target.fs, S3FileSystem):
            raise TypeError(f"{target} is not on S3")
        self._fs = target.fs
        bucket, key, _ = self._fs.split_path(target.path)
        self._names = {"Bucket": bucket, "Key": key}
        self.part_size = part_size
        self._upload_id: str | None = None
        self._etags: list[str] = []

    @property
    def offset(self) -> int:
        """The byte that the next part starts at."""
        return len(self._etags) * self.part_size

    def resume_or_start(self) -> int:
        """Continue the upload of a run before, or start one. Return the offset."""
        uploads = sorted(
            (
                upload
                for upload in self._call(
                    "list_multipart_uploads", Prefix=self._key
                ).get("Uploads", [])
                if upload["Key"] == self._key
            ),
            key=lambda upload: upload["Initiated"],
        )
        for stale in uploads[:-1]:
            self._abort(stale["UploadId"])
        if uploads:
            upload_id = uploads[-1]["UploadId"]
            parts = self._list_parts(upload_id)
            if all(
                part["PartNumber"] == number and part["Size"] == self.part_size
                for number, part in enumerate(parts, start=1)
            ):
                self._upload_id = upload_id
                self._etags = [part["ETag"] for part in parts]
                return self.offset
            self._abort(upload_id)
        return self.restart()

    def restart(self) -> int:
        """Drop every part, and start again at the first byte."""
        if self._upload_id is not None:
            self._abort(self._upload_id)
        self._upload_id = self._call("create_multipart_upload")["UploadId"]
        self._etags = []
        return 0

    def upload_part(self, data: bytes) -> None:
        response = self._call(
            "upload_part",
            UploadId=self._require_upload_id(),
            PartNumber=len(self._etags) + 1,
            Body=data,
        )
        self._etags.append(response["ETag"])

    def complete(self) -> None:
        """Make the object appear, whole, at its key."""
        self._call(
            "complete_multipart_upload",
            UploadId=self._require_upload_id(),
            MultipartUpload={
                "Parts": [
                    {"PartNumber": number, "ETag": etag}
                    for number, etag in enumerate(self._etags, start=1)
                ]
            },
        )
        self._upload_id = None
        self._fs.invalidate_cache(f"{self._names['Bucket']}/{self._key}")

    def abort(self) -> None:
        if self._upload_id is not None:
            self._abort(self._upload_id)
            self._upload_id = None
            self._etags = []

    @property
    def _key(self) -> str:
        return self._names["Key"]

    def _list_parts(self, upload_id: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        marker = 0
        while True:
            response = self._call(
                "list_parts", UploadId=upload_id, PartNumberMarker=marker
            )
            parts.extend(response.get("Parts", []))
            if not response.get("IsTruncated"):
                return parts
            marker = response["NextPartNumberMarker"]

    def _abort(self, upload_id: str) -> None:
        self._call("abort_multipart_upload", UploadId=upload_id)

    def _require_upload_id(self) -> str:
        if self._upload_id is None:
            raise RuntimeError("No upload is open. Call resume_or_start first.")
        return self._upload_id

    def _call(self, method: str, **parameters: Any) -> dict[str, Any]:
        names = dict(self._names)
        if method == "list_multipart_uploads":
            names.pop("Key")
        return cast("dict[str, Any]", self._fs.call_s3(method, **names, **parameters))
