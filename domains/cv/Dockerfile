##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# The Dagster code location of the CV domain. Build and runtime are split, so the final
# image carries the virtualenv and the source, but not uv or its cache. The build
# context is the root of the repository, because the lock is the lock of the workspace.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# The virtualenv lives outside the project, so the runtime stage copies it as a layer of its
# own. A source change then leaves that layer, and its digest, as it was.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /cv_lakehouse

# The dependencies of lakehouse-cv and of core, from the lock file alone. This layer
# rebuilds only when a pyproject.toml or uv.lock changes. uv reads every member to
# resolve the workspace, studio included, and installs none of them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=packages/lakehouse_core/pyproject.toml,target=packages/lakehouse_core/pyproject.toml \
    --mount=type=bind,source=domains/cv/pyproject.toml,target=domains/cv/pyproject.toml \
    --mount=type=bind,source=studio/pyproject.toml,target=studio/pyproject.toml \
    uv sync --frozen --package lakehouse-cv --no-install-workspace --no-dev


FROM python:3.12-slim-bookworm AS runtime

# The project is on PYTHONPATH rather than installed. An install writes metadata that
# hashes the source, and that changes the digest of the virtualenv layer on every commit.
#
# The lake is mounted at /lake. The Triton server resolves the image paths that silver
# sends, so it mounts the lake at the same path.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/cv_lakehouse/packages/lakehouse_core/src:/cv_lakehouse/domains/cv/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CV_LAKEHOUSE_ROOT=/lake

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin lakehouse

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

# `load_from_defs_folder` finds the project root by the pyproject.toml above it, so the
# file comes along with the source.
COPY --chown=10001:10001 packages/lakehouse_core/src /cv_lakehouse/packages/lakehouse_core/src
COPY --chown=10001:10001 domains/cv/pyproject.toml /cv_lakehouse/domains/cv/pyproject.toml
COPY --chown=10001:10001 domains/cv/src /cv_lakehouse/domains/cv/src

WORKDIR /cv_lakehouse

# The runs write bronze and silver, so uid 10001 needs write access to the lake mount.
USER 10001

# 4000 is the Dagster code server. 8000 is the gold API, which the same image serves
# with `uvicorn --factory lakehouse_cv.gold_api.app:create_app_from_env --host 0.0.0.0`.
# 3000 is the webserver. The webserver and the daemon of a deployment run this image too,
# with `dagster-webserver` and `dagster-daemon run`.
EXPOSE 3000 4000 8000

CMD ["dagster", "code-server", "start", "-h", "0.0.0.0", "-p", "4000", "-m", "lakehouse_cv.definitions", "--location-name", "cv_lakehouse"]
