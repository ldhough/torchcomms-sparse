#!/bin/bash
# Incremental NCCLX build for Perlmutter.
# Assumes deps were already built via a full build_ncclx_pm.sh run.
module load python

INSTALL_PREFIX=/pscratch/sd/l/ldhough/ncclx-deps \
NCCL_BUILD_SKIP_DEPS=1 \
bash "$(dirname "$0")/build_ncclx_pm.sh" "$@"
