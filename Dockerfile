##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# The Dagster code location. Build and runtime are split, so the final image carries
# the virtualenv and the source, but not uv or its cache.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# The virtualenv lives outside the project, so the runtime stage copies it as a layer of its
# own. A source change then leaves that layer, and its digest, as it was.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /cv_lakehouse

# The dependencies from the lock file alone. This layer rebuilds only when
# pyproject.toml or uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev


FROM python:3.12-slim-bookworm AS runtime

# The project is on PYTHONPATH rather than installed. An install writes metadata that
# hashes the source, and that changes the digest of the virtualenv layer on every commit.
#
# The lake is mounted at /lake. The Triton server resolves the image paths that silver
# sends, so it mounts the lake at the same path.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/cv_lakehouse/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CV_LAKEHOUSE_ROOT=/lake

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin lakehouse

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

# `load_from_defs_folder` finds the project root by the pyproject.toml above it, so the
# file comes along with the source.
COPY --chown=10001:10001 pyproject.toml /cv_lakehouse/pyproject.toml
COPY --chown=10001:10001 src /cv_lakehouse/src

WORKDIR /cv_lakehouse

# The runs write bronze and silver, so uid 10001 needs write access to the lake mount.
USER 10001

EXPOSE 4000

CMD ["dagster", "code-server", "start", "-h", "0.0.0.0", "-p", "4000", "-m", "cv_lakehouse.definitions", "--location-name", "cv_lakehouse"]
