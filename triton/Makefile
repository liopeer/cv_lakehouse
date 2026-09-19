##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

.PHONY: image up down logs test lock

# The lock files that the images install. A new run keeps every pin that still
# resolves. `make lock UPGRADE=--upgrade` moves them to the newest versions.
#
# The PyTorch index mirrors old copies of PyPI packages, such as certifi. So every
# package takes its best version from either index, and the `.in` files pin torch by
# its local version, which only the PyTorch index has.
UV_COMPILE := uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
	--index-strategy unsafe-best-match --generate-hashes --emit-index-url \
	--custom-compile-command "make -C triton lock" $(UPGRADE)

# An AMD GPU shows up as /dev/kfd. `make up PLATFORM=cuda` overrides the choice.
PLATFORM ?= $(if $(wildcard /dev/kfd),rocm,cuda)
COMPOSE := RENDER_GID=$(shell getent group render | cut -d: -f3) \
	docker compose --profile $(PLATFORM)

image:
	docker build -f Dockerfile.$(PLATFORM) -t ghcr.io/liopeer/cv_lakehouse-triton:dev-$(PLATFORM) .

# Builds this checkout. The first start then builds the models into the cache volume.
up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f triton-$(PLATFORM)

test:
	uv run --frozen --directory shared_deps_server --extra export pytest

lock:
	$(UV_COMPILE) requirements-cuda.in -o requirements-cuda.txt \
		--extra-index-url https://download.pytorch.org/whl/cu126
	$(UV_COMPILE) requirements-rocm.in -o requirements-rocm.txt \
		--extra-index-url https://download.pytorch.org/whl/rocm7.2
