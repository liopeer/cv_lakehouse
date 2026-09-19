##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

.PHONY: image up down logs test lock

# The lock files that the images install. A new run keeps every pin that still
# resolves. `make lock UPGRADE=--upgrade` moves them to the newest versions.
#
# The lock of the base image constrains the resolve, and the packages that it pins stay
# out. The PyTorch and AMD indexes hold the torch of the base image. They also mirror old
# copies of PyPI packages, so every package takes its best version from any index.
UV_COMPILE := uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
	--index-strategy unsafe-best-match --generate-hashes --emit-index-url \
	--custom-compile-command "make -C triton lock" $(UPGRADE)
lock_against_base = --constraint base/requirements-$(1).txt \
	$(foreach package,$(shell sed -n 's/^\([a-z0-9._-]*\)==.*/\1/p' base/requirements-$(1).txt),--no-emit-package $(package))
ROCM_VERSION := $(shell sed -n 's/^ROCM_VERSION=//p' base/versions.env)

# An AMD GPU shows up as /dev/kfd. `make up PLATFORM=cuda` overrides the choice.
PLATFORM ?= $(if $(wildcard /dev/kfd),rocm,cuda)
COMPOSE := RENDER_GID=$(shell getent group render | cut -d: -f3) \
	docker compose --profile $(PLATFORM)

# Compose reads the tag suffix from TRITON_VERSION, so every recipe passes it on.
export TRITON_VERSION

image:
	docker build -f Dockerfile.$(PLATFORM) -t ghcr.io/liopeer/cv_lakehouse-triton:dev-$(PLATFORM) .

# Builds this checkout. The first start then builds the models into the cache volume.
# `make up TRITON_VERSION=X.Y.Z` pulls that release, and builds nothing.
up:
	$(COMPOSE) up -d $(if $(TRITON_VERSION),--pull always --no-build,--build)

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f triton-$(PLATFORM)

test:
	uv run --frozen --directory shared_deps_server --extra export pytest

lock:
	$(UV_COMPILE) requirements-cuda.in -o requirements-cuda.txt $(call lock_against_base,cuda) \
		--extra-index-url https://download.pytorch.org/whl/cpu
	$(UV_COMPILE) requirements-rocm.in -o requirements-rocm.txt $(call lock_against_base,rocm) \
		--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-$(ROCM_VERSION)/ \
		--no-emit-package triton
