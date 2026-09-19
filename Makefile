#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
# Use `make help` to list targets.

.DEFAULT_GOAL := help
UV_RUN := uv run --frozen

.PHONY: help sync lock license-headers format format-check lint lint-fix typecheck test \
	defs dev image studio triton-up triton-down triton-logs check clean

help:  ## List targets.
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync:  ## Install dependencies from the lock file.
	uv sync --frozen

lock:  ## Update the lock file after a dependency change.
	uv lock

# The file list comes from git, so nothing that git ignores gets a header. An
# `__init__.py` stays empty, and Apple's vendored mobileclip keeps its own licence.
# A pattern is an fnmatch on the path, where `*` also matches a `/`.
license-headers: sync  ## Add the licence header to every source file.
	git ls-files --cached --others --exclude-standard -z \
		| xargs -0 $(UV_RUN) licenseheaders -t dev/licenseheader_lionelpeer.tmpl \
			-x '*__init__.py' 'triton/shared_deps_server/src/shared_deps/mobileclip/*' -f

format: license-headers  ## Format the code, and add the licence headers.
	$(UV_RUN) ruff format .

# A missing header shows up as a diff. uv.lock is left out because `sync` can touch it.
format-check: license-headers  ## Report format and licence header problems.
	git diff --exit-code -- ':!uv.lock'
	$(UV_RUN) ruff format --check .

lint: sync  ## Report lint problems.
	$(UV_RUN) ruff check .

lint-fix: sync  ## Fix lint problems.
	$(UV_RUN) ruff check --fix .

typecheck: sync  ## Check the types.
	$(UV_RUN) pyrefly check

test: sync  ## Run the unit tests.
	$(UV_RUN) pytest

defs: sync  ## Validate the Dagster definitions.
	$(UV_RUN) dg check defs

dev: sync  ## Start the Dagster UI on port 3000.
	$(UV_RUN) dg dev

image:  ## Build the Dagster code location image, as the release does.
	docker build -t cv_lakehouse:dev .

# Silver is Parquet. This builds a LightlyStudio database from it on demand, under
# $$CV_LAKEHOUSE_ROOT/.studio, which is a cache and safe to delete. A standalone script
# with its own dependencies, so the package needs no GUI: uv reads them from its header.
studio:  ## Browse silver in LightlyStudio. make studio DATASETS="wider_face"
	uv run tools/studio.py $(DATASETS)

# The MobileCLIP server that silver embeds with. `triton/README.md` has the details.
# It needs a Linux host with docker and the NVIDIA runtime.
triton-up:  ## Build and start the Triton server. gRPC on 8011, HTTP on 8010.
	$(MAKE) -C triton up

triton-down:  ## Stop the Triton server.
	$(MAKE) -C triton down

triton-logs:  ## Follow the Triton server log.
	$(MAKE) -C triton logs

check: format-check lint typecheck test defs  ## Run every check.

clean:  ## Remove the caches.
	rm -rf .ruff_cache .pytest_cache
	find src tests tools -name __pycache__ -type d -exec rm -rf {} +
