# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12346 \
autoregressive/sample/sample_t2i_ddp.py \
--prompt-csv evaluations/t2i/coco_captions.csv \
--sample-dir samples_coco \
--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
"$@"
