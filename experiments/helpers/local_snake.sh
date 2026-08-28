#!/bin/bash

source .pass

echo "args: $@"

snakemake -j 100 -k \
    --rerun-triggers mtime input \
    --printshellcmds \
    --latency-wait 60 \
    --resources db_connections=1 \
    --config use_venv=True venv_path=$venv_path $@
