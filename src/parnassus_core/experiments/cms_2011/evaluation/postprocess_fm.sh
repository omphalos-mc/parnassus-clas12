#!/bin/bash
set -x

python preprocess_eval_fast.py --ip -m \
    -e $1 \
    -d $2 \
    -o "evals/parnassus_core_"$3 -n -1 --eta 2.5

set +x
