#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Fail when a shared library or an executable under the given directories needs a
# library that the image does not have. The runtime stage drops the build tools, so a
# missing library would otherwise show up only when Triton loads a model.
#
# Usage: check_libraries.sh <dir> ...

set -euo pipefail

status=0
while read -r file; do
    # The runtime images have no `file`, so read the ELF magic bytes.
    if [[ $(head -c 4 "${file}" | od -An -c | tr -d ' ') != '177ELF' ]]; then
        continue
    fi
    missing=$(ldd "${file}" 2>/dev/null | grep 'not found' || true)
    if [[ -n "${missing}" ]]; then
        echo "${file}:" >&2
        echo "${missing}" >&2
        status=1
    fi
done < <(find "$@" -type f \( -name '*.so' -o -name '*.so.*' -o -perm -u+x \))

exit "${status}"
