##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

.PHONY: image test

IMAGE := ghcr.io/liopeer/cv_lakehouse-studio

# Two images from one Dockerfile: the LightlyStudio server, and the sync.
image:
	docker build --target gui -t $(IMAGE):dev .
	docker build --target sync -t $(IMAGE)-sync:dev .

# The tests need a Postgres with pgvector. They start one with Docker, or take the server
# in CV_LAKEHOUSE_STUDIO_TEST_POSTGRES_URL.
# studio is a member of the uv workspace at the root, and shares its lock.
test:
	uv run --frozen --package cv-lakehouse-studio pytest
