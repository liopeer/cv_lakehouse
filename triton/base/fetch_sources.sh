#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Fetch each repository at its pinned commit into a local repository, under the given
# branch names. Then send every git fetch from the given URL prefixes to those local
# repositories.
#
# build.py and CMake's FetchContent clone by branch or tag, not by commit. With this, a
# clone of a listed branch gets the pinned commit, and a clone of any other ref fails.
#
# stdin: one repository per line, as `<name> <url> <commit> <branch>...`. `<name>` is the
# last part of the URL that the build asks for, such as `core` for `.../core.git`. The
# branches are the names that the build asks for, such as `pinned`.
# Arguments: the URL prefixes to redirect, such as https://github.com/triton-inference-server/

set -euo pipefail

REPOS=/src/repos

mkdir -p "${REPOS}"
while read -r name url commit branches; do
    [[ -z "${name}" || "${name}" == \#* ]] && continue
    repo="${REPOS}/${name}.git"
    git init --quiet --bare "${repo}"
    git -C "${repo}" fetch --quiet --depth 1 "${url}" "${commit}"
    for branch in ${branches}; do
        git -C "${repo}" update-ref "refs/heads/${branch}" "${commit}"
    done
    git -C "${repo}" symbolic-ref HEAD "refs/heads/${branches%% *}"
    echo "${name} ${commit} ${branches}"
done

# A backend clones with --recursive, and git refuses a file:// submodule by default.
git config --global protocol.file.allow always
for prefix in "$@"; do
    git config --global --add "url.file://${REPOS}/.insteadOf" "${prefix}"
done
