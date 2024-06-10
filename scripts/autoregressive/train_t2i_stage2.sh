# !/bin/bash
set -x

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
--data-path /path/to/high_aesthetic_10M \
--t5-feat-path /path/to/high_aesthetic_10M_flan_t5_xl \
--short-t5-feat-path /path/to/high_aesthetic_10M_trunc_flan_t5_xl \
--dataset t2i \
--image-size 512 \
"$@"
