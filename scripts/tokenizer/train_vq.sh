# !/bin/bash
set -x

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"