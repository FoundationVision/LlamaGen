# !/bin/bash
set -x

=======
# Set default values for single-node training
nnodes=1                  # number of nodes
nproc_per_node=1          # number of processes per node
node_rank=0               # node rank
master_addr="localhost"   # address of the master node
master_port=29500         # port of the master node


>>>>>>> 515239903504a5248d5915e9c4399344dc2ebf60


torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"


#bash scripts/tokenizer/train_vq.sh --cloud-save-path . --data-path ~/dataset/imagenet.py --image-size 256 --vq-model VQ-16   