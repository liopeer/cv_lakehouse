#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Copy the licence and notice files of each component into <dest>/<component>/. Fail
# when a component has none, so that nothing ships without its licence.
#
# Usage: collect_licences.sh <dest> <component>=<source> ...
# A source is one of:
#   <dir>          the licence files at the top of a source tree
#   deb:<package>  the licence files that an installed apt package lists
#   git:<repo>     the licence files at the top of the `pinned` branch of a bare repository
#   tree:<dir>     the licence files anywhere in a tree, with their relative paths, as for
#                  third_party's copy of the sources that Triton links statically

set -euo pipefail

dest=$1
shift

is_licence() {
    local base
    base=$(basename "$1")
    [[ "${base,,}" =~ ^(licen[cs]e|copying|copyright|notice|thirdpartynotices|third_party_notices)([._-].*)?$ ]]
}

missing=()
for spec in "$@"; do
    component=${spec%%=*}
    source=${spec#*=}
    out="${dest}/${component}"
    mkdir -p "${out}"
    found=0
    case "${source}" in
        deb:*)
            while read -r path; do
                if [[ -f "${path}" ]] && is_licence "${path}"; then
                    cp "${path}" "${out}/$(basename "$(dirname "${path}")")-$(basename "${path}")"
                    found=1
                fi
            done < <(dpkg -L "${source#deb:}")
            ;;
        git:*)
            repo=${source#git:}
            while read -r path; do
                if is_licence "${path}"; then
                    git -C "${repo}" show "pinned:${path}" > "${out}/${path}"
                    found=1
                fi
            done < <(git -C "${repo}" ls-tree --name-only pinned)
            ;;
        tree:*)
            root=${source#tree:}
            while read -r path; do
                if is_licence "${path}"; then
                    mkdir -p "${out}/$(dirname "${path}")"
                    cp "${root}/${path}" "${out}/${path}"
                    found=1
                fi
            done < <(cd "${root}" && find . -maxdepth 5 -type f -printf '%P\n')
            ;;
        *)
            for path in "${source}"/*; do
                if [[ -f "${path}" ]] && is_licence "${path}"; then
                    cp "${path}" "${out}/"
                    found=1
                fi
            done
            ;;
    esac
    if [[ "${found}" == 0 ]]; then
        missing+=("${component} (${source})")
    fi
done

if (( ${#missing[@]} )); then
    printf 'No licence file for: %s\n' "${missing[@]}" >&2
    exit 1
fi
