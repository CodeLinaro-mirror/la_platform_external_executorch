#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# This script is run before building ExecuTorch binaries

if [[ "$(uname -m)" == "aarch64" ]]; then
  # On some Linux aarch64 systems, the "atomic" library is not found during linking.
  # To work around this, replace "atomic" with the literal ${ATOMIC_LIB} so the
  # build system uses the full path to the atomic library.
  file="extension/llm/tokenizers/third-party/sentencepiece/src/CMakeLists.txt"
  sed 's/list(APPEND SPM_LIBS "atomic")/list(APPEND SPM_LIBS ${ATOMIC_LIB})/' \
    "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"

  grep -n 'list(APPEND SPM_LIBS ${ATOMIC_LIB})' "$file" && \
    echo "the file $file has been modified for atomic to use full path"
fi

# Manually install build requirements because `python setup.py bdist_wheel` does
# not install them. TODO(dbort): Switch to using `python -m build --wheel`,
# which does install them. Though we'd need to disable build isolation to be
# able to see the installed torch package.

"${GITHUB_WORKSPACE}/${REPOSITORY}/install_requirements.sh"  --example
