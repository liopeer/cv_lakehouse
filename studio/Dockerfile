##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# LightlyStudio (github.com/lightly-ai/lightly-studio, Apache-2.0) at a pinned tag, with
# the patches in patches/ on top, served against Postgres. The build context is this
# directory.
#
# Two images come out of this file, on one Python environment:
#   --target gui    the LightlyStudio server and its GUI. The default.
#   --target sync   the service that keeps LightlyStudio in step with gold.
#
# Keep the tag equal to the `lightly-studio` pin in pyproject.toml. A test compares them.
ARG LIGHTLY_STUDIO_REF=v1.1.2
# Where the GUI goes: beside the package in the venv, as in the wheel of upstream.
ARG UI_DIR=/opt/venv/lib/python3.12/site-packages/lightly_studio/dist_lightly_studio_view_app

# The source of upstream at the tag, with the patches applied. The .git directory stays,
# because the version export below reads the commit from it.
FROM python:3.12-slim AS source
ARG LIGHTLY_STUDIO_REF
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${LIGHTLY_STUDIO_REF}" https://github.com/lightly-ai/lightly-studio.git .
COPY patches/ /patches/
RUN git apply --verbose /patches/*.patch

# The Python environment. The versions come from the uv.lock of upstream, as
# constraints. torch comes from the backend in TORCH_BACKEND, which is the CPU build, so
# the image carries no CUDA: the server only embeds search queries.
#
# uv is the version that the workspace of upstream requires.
#
# --no-sources: upstream declares lightly-studio-serve as a workspace source, which uv
# installs editable, into /src. /src does not reach the final images.
FROM source AS app
ARG TORCH_BACKEND=cpu
ARG UI_DIR
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    LIGHTLY_STUDIO_ANALYTICS_ENABLED=false
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-hashes --no-emit-workspace \
      --package lightly-studio --extra cloud-storage -o /tmp/constraints.txt \
 && uv venv --python /usr/local/bin/python3.12 /opt/venv \
 && uv pip install --no-sources --torch-backend "${TORCH_BACKEND}" -c /tmp/constraints.txt \
      ./lightly_studio_serve "./lightly_studio[cloud-storage]"
# The GUI build needs the API schema and the version. An import of lightly_studio needs
# the GUI directory, so a placeholder stands in, as in `make export-schema` of upstream.
RUN mkdir -p "$UI_DIR" && touch "$UI_DIR/index.html" \
 && python -m lightly_studio.export_schema --output /tmp/openapi.json \
 && python -m lightly_studio.export_version --output /tmp/version.json \
 && rm -rf "$UI_DIR"
# This package, on the environment above. LightlyStudio is already there, at the version
# that pyproject.toml pins.
COPY pyproject.toml /studio/pyproject.toml
COPY src /studio/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-sources -c /tmp/constraints.txt /studio

# The GUI, a SvelteKit static build, generated against the exported schema.
FROM node:24.13.1-bookworm-slim AS view
# The default heap is too small for the Vite build. The CI of upstream sets the same.
ENV NODE_OPTIONS=--max-old-space-size=4096
COPY --from=source /src/lightly_studio_view /src/lightly_studio_view
COPY --from=app /tmp/openapi.json /src/lightly_studio/openapi.json
COPY --from=app /tmp/version.json /src/lightly_studio_view/src/lib/version.json
WORKDIR /src/lightly_studio_view
RUN npm ci && npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    LIGHTLY_STUDIO_ANALYTICS_ENABLED=false
COPY --from=app /opt/venv /opt/venv
# An import of lightly_studio needs the GUI directory. The gui target fills it.
ARG UI_DIR
RUN mkdir -p "$UI_DIR" && touch "$UI_DIR/index.html"
# Not root, and a uid that the deployment can name for the lake mount.
RUN groupadd --gid 10001 studio \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin studio
WORKDIR /home/studio

# The sync loads gold into Postgres, and serves the corrections of the curators.
FROM runtime AS sync
USER studio
EXPOSE 8002
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/healthz', timeout=2)"]
CMD ["python", "-m", "cv_lakehouse_studio.sync_service"]

FROM runtime AS gui
ARG UI_DIR
ENV LIGHTLY_STUDIO_MODEL_CACHE_DIR=/opt/lightly-studio/models \
    LIGHTLY_STUDIO_HOST=0.0.0.0 \
    LIGHTLY_STUDIO_PORT=8001
# The layout that `make build` of upstream gives the wheel: the GUI beside the package.
COPY --from=view /src/lightly_studio_view/build/ ${UI_DIR}/
COPY --from=app /tmp/version.json ${UI_DIR}/version-info.json
# The checkpoint of the default embedder, which embeds the text of a search. A network
# with no route to docs-assets.developer.apple.com then still starts the server. The
# download is owner only, and the service runs as another user.
RUN python -c "from lightly_studio.embed.mobileclip_embedder import _get_cached_mobileclip_checkpoint as c; print(c())" \
 && chmod -R a+rX "$LIGHTLY_STUDIO_MODEL_CACHE_DIR"
USER studio
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=2)"]
CMD ["python", "-m", "cv_lakehouse_studio.gui_entrypoint"]
