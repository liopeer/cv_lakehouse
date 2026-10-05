#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Tell what an IMAGE_PATH names, and fetch the images that URLs name.

An IMAGE_PATH is a local path, read where the server runs, or an http or https URL.
The server holds no credentials, so a URL into private storage is presigned by the
caller. A presigned URL carries its signature in the query, so an error names a URL
without it.
"""

import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

URL_SCHEMES = ("http", "https")


class ImageFetchError(RuntimeError):
    pass


def is_url(image_path: str) -> bool:
    """Tell a URL from a local path. Refuse any scheme but http and https."""
    scheme = urlsplit(image_path).scheme
    if not scheme:
        return False
    if scheme not in URL_SCHEMES:
        raise ValueError(
            f"{redact_url(image_path)} has the scheme {scheme}. IMAGE_PATH takes a "
            "local path, or an http or https URL. Presign a URL into private storage."
        )
    return True


def fetch_urls(
    urls: Iterable[str], *, timeout_seconds: float, max_workers: int
) -> dict[str, bytes]:
    """Fetch every distinct URL once, several at a time. Map each URL to its bytes."""
    distinct = list(dict.fromkeys(urls))
    if not distinct:
        return {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(distinct))) as pool:
        bodies = pool.map(
            lambda url: fetch_url(url=url, timeout_seconds=timeout_seconds), distinct
        )
        return dict(zip(distinct, bodies, strict=True))


def redact_url(url: str) -> str:
    """Drop the query and the fragment, which hold the signature of a presigned URL."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch_url(*, url: str, timeout_seconds: float) -> bytes:
    """Fetch the body of one URL."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ImageFetchError(
            f"{redact_url(url)} answered {error.code} {error.reason}"
        ) from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise ImageFetchError(f"{redact_url(url)} could not be fetched: {error}") from None
