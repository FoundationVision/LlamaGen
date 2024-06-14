import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tokenizer.vqgan.model import VQModel
from tokenizer.vqgan.model import VQGAN_FROM_TAMING

# before running demo, make sure to:
# (1) download all needed models from https://github.com/CompVis/taming-transformers and put in pretrained_models/
# (2) pip install pytorch_lightning
# (3) python3 tools/convert_pytorch_lightning_to_torch.py
# (4) pip uninstall pytorch_lightning


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    cfg, ckpt = VQGAN_FROM_TAMING[args.vqgan]
    config = OmegaConf.load(cfg)
    model = VQModel(**config.model.get("params", dict()))
    model.init_from_ckpt(ckpt)
    model.to(device)
    model.eval()

    # load image
    img_path = args.image_path
    out_path = args.image_path.replace('.jpg', '_vqgan.jpg').replace('.jpeg', '_vqgan.jpeg').replace('.png', '_vqgan.png')
    input_size = args.image_size
    img = Image.open(img_path).convert("RGB")

    # preprocess
    size_org = img.size
    img = img.resize((input_size, input_size))
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
    output = F.interpolate(output, size=[size_org[1], size_org[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    # save        
    Image.fromarray(sample).save(out_path)
    print("Reconstructed image is saved to {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="assets/example.jpg")
    parser.add_argument("--vqgan", type=str, choices=list(VQGAN_FROM_TAMING.keys()), default="vqgan_openimage_f8_16384")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
