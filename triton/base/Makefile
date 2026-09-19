##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Build and push the Triton base images. They build on a local machine, not in CI.
# README.md has the details.
#
#   make build PLATFORM=rocm    build ghcr.io/liopeer/cv_lakehouse-triton-base:rocm-dev
#   make push PLATFORM=rocm     tag it rocm-r25.12-<today>, push it, print the digest
#   make lock                   compile the requirements-*.txt lock files

.DEFAULT_GOAL := help

include versions.env
VERSION_KEYS := $(shell sed -n 's/^\([A-Z][A-Z0-9_]*\)=.*/\1/p' versions.env)

PLATFORM ?= $(if $(wildcard /dev/kfd),rocm,cuda)
DATE ?= $(shell date +%Y%m%d)
IMAGE := ghcr.io/liopeer/cv_lakehouse-triton-base
LOCAL_TAG := $(IMAGE):$(PLATFORM)-dev
RELEASE_TAG := $(IMAGE):$(PLATFORM)-r$(TRITON_CONTAINER_VERSION)-$(DATE)

.PHONY: help build push lock clean

help:  ## List targets.
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-8s %s\n", $$1, $$2}'

build:  ## Build the base image for PLATFORM, and print its size.
	docker buildx build --load \
		--file Dockerfile.$(PLATFORM) \
		$(foreach key,$(VERSION_KEYS),--build-arg $(key)='$($(key))') \
		--tag $(LOCAL_TAG) .
	docker image inspect --format '$(LOCAL_TAG): {{.Size}} bytes' $(LOCAL_TAG)

# Needs `docker login ghcr.io` with a token that has the write:packages scope.
push:  ## Tag the built image for PLATFORM with today's date, push it, and print the digest.
	docker tag $(LOCAL_TAG) $(RELEASE_TAG)
	docker push $(RELEASE_TAG)
	@echo "Pin this in triton/Dockerfile.$(PLATFORM):"
	@docker image inspect --format '{{index .RepoDigests 0}}' $(RELEASE_TAG)

# A new run keeps every pin that still resolves. `make lock UPGRADE=--upgrade` moves them.
# The PyTorch and AMD indexes mirror old copies of PyPI packages, so every package takes
# its best version from any index, and the `.in` files pin torch by its local version.
UV_COMPILE := uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
	--index-strategy unsafe-best-match --generate-hashes --emit-index-url --emit-find-links \
	--custom-compile-command "make -C triton/base lock" $(UPGRADE)

lock:  ## Compile the requirements-*.txt lock files.
	$(UV_COMPILE) requirements-build.in -o requirements-build.txt
	$(UV_COMPILE) requirements-cuda.in -o requirements-cuda.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu \
		--extra-index-url https://pypi.nvidia.com
	$(UV_COMPILE) requirements-rocm.in -o requirements-rocm.txt \
		--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-$(ROCM_VERSION)/ \
		--no-emit-package triton

clean:  ## Remove the local build cache and the dev tags.
	-docker image rm $(IMAGE):cuda-dev $(IMAGE):rocm-dev
	docker buildx prune --force
