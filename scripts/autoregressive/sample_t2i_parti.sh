# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12347 \
autoregressive/sample/sample_t2i_ddp.py \
--prompt-csv evaluations/t2i/PartiPrompts.tsv \
--sample-dir samples_parti \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
"$@"
