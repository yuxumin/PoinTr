#!/usr/bin/env bash

set -x
NGPUS=$1
PORT=$2
PY_ARGS=${@:3}

python -m torch.distributed.launch --master_port=${PORT} --nproc_per_node=${NGPUS} main.py --launcher pytorch --sync_bn ${PY_ARGS}
