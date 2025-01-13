#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function verify_md5() {
    # Arg 1: Expected checksum for file
    # Arg 2: Path to file

    [[ $# -ne 2 ]]  \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 2, but got $#"; exit 1; }
    local ref_checksum="${1}"
    local file="${2}"

    if [[ "${OS}" == "Darwin" ]]; then
        local file_checksum="$(md5 -q $file)"
    else
        local file_checksum="$(md5sum $file | awk '{print $1}')"
    fi
    if [[ ${ref_checksum} != ${file_checksum} ]]; then
        echo "Mismatched MD5 checksum for file: ${file}. Expecting ${ref_checksum} but got ${file_checksum}. Exiting."
    fi
}

function patch_repo() {
    # Arg 1: Directory of repo to patch
    # Arg 2: Rev to start patching at
    # Arg 3: Directory 'setup-dir' containing patches in 'setup-dir/$name'

    [[ $# -ne 3 ]]  \
        && { echo "[${FUNCNAME[0]}] Invalid number of args, expecting 3, but got $#"; exit 1; }

    local repo_dir="${1}"
    local base_rev="${2}"
    local name="$(basename $repo_dir)"
    local patch_dir="${3}/$name"

    echo -e "[${FUNCNAME[0]}] Patching ${name}..."
    cd $repo_dir
    git fetch
    git reset --hard ${base_rev}

    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch

    echo -e "[${FUNCNAME[0]}] Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${repo_dir} dir.\n"
}
