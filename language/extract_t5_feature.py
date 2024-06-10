import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import os
import json

from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder

CAPTION_KEY = {
    'blip': 0,
    'llava': 1,
    'llava_first': 2,
}
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, lst_dir, start, end, caption_key, trunc_caption=False):
        img_path_list = []
        for lst_name in sorted(os.listdir(lst_dir))[start: end+1]:
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    # caption = data[caption_key]
                    caption = data['text'][CAPTION_KEY[caption_key]]
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    if trunc_caption:
                        caption = caption.split('.')[0]
                    img_path_list.append((caption, code_dir, line_idx))
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        caption, code_dir, code_name = self.img_path_list[index]
        return caption, code_dir, code_name


        
#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.t5_path, exist_ok=True)

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.data_path, args.data_start, args.data_end, args.caption_key, args.trunc_caption)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    for caption, code_dir, code_name in loader:
        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
        os.makedirs(os.path.join(args.t5_path, code_dir[0]), exist_ok=True)
        np.save(os.path.join(args.t5_path, code_dir[0], '{}.npy'.format(code_name.item())), x)
        print(code_name.item())

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--data-end", type=int, required=True)
    parser.add_argument("--caption-key", type=str, default='blip', choices=list(CAPTION_KEY.keys()))
    parser.add_argument("--trunc-caption", action='store_true', default=False)
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)
