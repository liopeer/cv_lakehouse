#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
# Use `make help` to list targets.

.DEFAULT_GOAL := help
UV_RUN := uv run --frozen

.PHONY: help sync lock license-headers format format-check lint lint-fix typecheck test \
	defs dev api image check clean

help:  ## List targets.
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync:  ## Install dependencies from the lock file.
	uv sync --frozen --all-packages

lock:  ## Update the lock file after a dependency change.
	uv lock

# The file list comes from git, so nothing that git ignores gets a header. An
# `__init__.py` stays empty, and Apple's vendored mobileclip keeps its own licence.
# release-please writes the changelogs without a header, and the release checks run
# on its commit. A pattern is an fnmatch on the path, where `*` also matches a `/`.
license-headers: sync  ## Add the licence header to every source file.
	git ls-files --cached --others --exclude-standard -z \
		| xargs -0 $(UV_RUN) licenseheaders -t dev/licenseheader_lionelpeer.tmpl \
			-x '*__init__.py' 'triton/shared_deps_server/src/shared_deps/mobileclip/*' \
			'*CHANGELOG.md' -f

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

# One run per test directory. `domains/cv/tests` and `studio/tests` are both the package
# `tests`, so one run cannot import them both. studio has its own target, as it needs
# Docker.
test: sync  ## Run the unit tests.
	$(UV_RUN) pytest tests
	$(UV_RUN) pytest packages/lakehouse_core/tests
	$(UV_RUN) pytest domains/cv/tests

# The workspace has one virtualenv at its root, so dg uses it and not one per project.
defs: sync  ## Validate the Dagster definitions.
	$(UV_RUN) dg check defs --use-active-venv

dev: sync  ## Start the Dagster UI on port 3000.
	$(UV_RUN) dg dev --use-active-venv

api: sync  ## Serve gold over HTTP on port 8000.
	$(UV_RUN) uvicorn --factory lakehouse_cv.gold_api.app:create_app_from_env --port 8000

image:  ## Build the Dagster code location image, as the release does.
	docker build -f domains/cv/Dockerfile -t cv_lakehouse:dev .

check: format-check lint typecheck test defs  ## Run every check.

clean:  ## Remove the caches.
	rm -rf .ruff_cache .pytest_cache
	find packages domains studio/src studio/tests tests -name __pycache__ -type d -exec rm -rf {} +
