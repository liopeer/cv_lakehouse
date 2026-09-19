##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

.PHONY: image up down logs

image:
	docker build -t ghcr.io/liopeer/cv_lakehouse-triton:dev .

# Builds this checkout. The first start then builds the engines into the cache volume.
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f triton-server
