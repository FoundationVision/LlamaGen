# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12344 \
tokenizer/consistencydecoder/reconstruction_cd_ddp.py \
"$@"