import argparse
import torch
import numpy as np

from tokenizer.tokenizer_image.vq_model import VQ_models
from torchvision.utils import save_image


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # load image code
    latent_dim = args.codebook_embed_dim
    latent_size = args.image_size // args.downsample_size
    codes = torch.from_numpy(np.load(args.code_path)).to(device)
    if codes.ndim == 3: # flip augmentation
        qzshape = (codes.shape[1], latent_dim, latent_size, latent_size)
    else:
        qzshape = (1, latent_dim, latent_size, latent_size)
    index_sample = codes.reshape(-1)
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]

    # save
    out_path = "sample_image_code.png"
    nrow = max(4, int(codes.shape[1]//2))
    save_image(samples, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print("Reconstructed image is saved to {}".format(out_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)