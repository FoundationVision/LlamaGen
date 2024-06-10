# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12337 \
language/extract_t5_feature.py \
--data-path /path/to/high_aesthetic_10M \
--t5-path /path/to/high_aesthetic_10M_trunc_flan_t5_xl \
--trunc-caption \
"$@"
