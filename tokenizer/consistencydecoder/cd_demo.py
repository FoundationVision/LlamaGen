import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import ConsistencyDecoderVAE


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16).to(device)

    # load image
    img_path = args.image_path
    out_path = args.image_path.replace('.jpg', '_cd.jpg').replace('.jpeg', '_cd.jpeg').replace('.png', '_cd.png')
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
    x_input = x.half().to(device)

    # inference
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        latent = vae.encode(x_input).latent_dist.sample().mul_(0.18215)
        # reconstruct:
        output = vae.decode(latent / 0.18215).sample # output value is between [-1, 1]

    # postprocess
    output = F.interpolate(output, size=[size_org[1], size_org[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    # save        
    Image.fromarray(sample).save(out_path)
    print("Reconstructed image is saved to {}".format(out_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="assets/example.jpg")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
