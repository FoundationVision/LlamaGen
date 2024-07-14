import torch
import torch.nn.functional as F

import os
import argparse
import numpy as np
from PIL import Image

import sys
sys.path.append("/Users/ryankwon03/Documents/GitHub/LlamaGen")

from tokenizer.tokenizer_image.vq_model import VQ_models
from dataset.augmentation import center_crop_arr


def main(args):
    
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu") #error
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    model.load_state_dict(model_weight)
    del checkpoint

    # output dir
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = args.image_path.replace('.jpg', '_{}.jpg'.format(args.suffix))
    out_path = out_path.replace('.jpeg', '_{}.jpeg'.format(args.suffix))
    out_path = out_path.replace('.png', '_{}.png'.format(args.suffix))
    out_filename = out_path.split('/')[-1]
    out_path = os.path.join(args.output_dir, out_filename)
    
    # load image
    pil_image = Image.open(args.image_path).convert("RGB")
    img = center_crop_arr(pil_image, args.image_size)
    # # preprocess
    # size_org = img.size
    # img = img.resize((input_size, input_size))
    img = np.array(img) / 255.
    x = 2.0 * img - 1.0 # x value is between [-1, 1]
    x = torch.tensor(x)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    x_input = x.float().to("cuda")

    # inference
    with torch.no_grad():
        latent, _, [_, _, indices] = model.encode(x_input)
        output = model.decode_code(indices, latent.shape) # output value is between [-1, 1]

    # postprocess
    output = F.interpolate(output, size=[args.image_size, args.image_size], mode='bicubic').permute(0, 2, 3, 1)[0]
    sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    # save        
    Image.fromarray(sample).save(out_path)
    print("Reconstructed image is saved to {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="assets/example.jpg")
    parser.add_argument("--output-dir", type=str, default="output_vq_demo")
    parser.add_argument("--suffix", type=str, default="tokenizer_image")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512, 1024], default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)