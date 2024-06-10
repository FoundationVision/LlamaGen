import time
import argparse
import torch
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.serve.gpt_model import GPT_models
from autoregressive.serve.llm import LLM 
from vllm import SamplingParams


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    print(f"image tokenizer is loaded")

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    latent_size = args.image_size // args.downsample_size
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
    prompt_token_ids = [[cind] for cind in class_labels]
    if args.cfg_scale > 1.0:
        prompt_token_ids.extend([[args.num_classes] for _ in range(len(prompt_token_ids))])
    # Create an LLM.
    llm = LLM(
        args=args, 
        model='autoregressive/serve/fake_json/{}.json'.format(args.gpt_model), 
        gpu_memory_utilization=0.9, 
        skip_tokenizer_init=True)
    print(f"gpt model is loaded")

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, 
        max_tokens=latent_size ** 2)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    t1 = time.time()
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False)
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.") 

    # decode to image
    index_sample = torch.tensor([output.outputs[0].token_ids for output in outputs], device=device)
    if args.cfg_scale > 1.0:
        index_sample = index_sample[:len(class_labels)]
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample_{}_vllm.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}_vllm.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="ckpt path for gpt model")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
