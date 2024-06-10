from PIL import Image
import gradio as gr
from tools.imagenet_en_cn import IMAGENET_1K_CLASSES
from huggingface_hub import hf_hub_download
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from vllm import SamplingParams
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.serve.llm import LLM
from autoregressive.serve.sampler import Sampler

device = "cuda"

model2ckpt = {
    "GPT-XL": ("vq_ds16_c2i.pt", "c2i_XL_384.pt", 384),
    "GPT-B": ("vq_ds16_c2i.pt", "c2i_B_256.pt", 256),
}

def load_model(args):
    ckpt_folder = "./"
    vq_ckpt, gpt_ckpt, image_size = model2ckpt[args.gpt_model]
    hf_hub_download(repo_id="FoundationVision/LlamaGen", filename=vq_ckpt, local_dir=ckpt_folder)
    hf_hub_download(repo_id="FoundationVision/LlamaGen", filename=gpt_ckpt, local_dir=ckpt_folder)
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(f"{ckpt_folder}{vq_ckpt}", map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # Create an LLM.
    args.image_size = image_size
    args.gpt_ckpt = f"{ckpt_folder}{gpt_ckpt}"
    llm = LLM(
        args=args, 
        model='serve/fake_json/{}.json'.format(args.gpt_model), 
        gpu_memory_utilization=0.6, 
        skip_tokenizer_init=True)
    print(f"gpt model is loaded")
    return vq_model, llm, image_size


def infer(cfg_scale, top_k, top_p, temperature, class_label, seed):
    llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler = Sampler(cfg_scale)
    args.cfg_scale = cfg_scale
    n = 4
    latent_size = image_size // args.downsample_size
    # Labels to condition the model with (feel free to change):
    class_labels = [class_label for _ in range(n)]
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    prompt_token_ids = [[cind] for cind in class_labels]
    if cfg_scale > 1.0:
        prompt_token_ids.extend([[args.num_classes] for _ in range(len(prompt_token_ids))])

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k, 
        max_tokens=latent_size ** 2)
    
    t1 = time.time()
    torch.manual_seed(seed)
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False)
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    

    index_sample = torch.tensor([output.outputs[0].token_ids for output in outputs], device=device)
    if cfg_scale > 1.0:
        index_sample = index_sample[:len(class_labels)]
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")
    # Convert to PIL.Image format:
    samples = samples.mul(127.5).add_(128.0).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    samples = [Image.fromarray(sample) for sample in samples]
    return samples


parser = argparse.ArgumentParser()
parser.add_argument("--gpt-model", type=str, default="GPT-XL")
parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
parser.add_argument("--from-fsdp", action='store_true')
parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
parser.add_argument("--compile", action='store_true', default=False)
parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--cfg-scale", type=float, default=4.0)
parser.add_argument("--cfg-interval", type=float, default=-1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
args = parser.parse_args()

vq_model, llm, image_size = load_model(args)

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center'>Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</h1>")

    with gr.Tabs():
        with gr.TabItem('Generate'):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        i1k_class = gr.Dropdown(
                            list(IMAGENET_1K_CLASSES.values()),
                            value='llama [羊驼]',
                            type="index", label='ImageNet-1K Class'
                        )
                    cfg_scale = gr.Slider(minimum=1, maximum=25, step=0.1, value=4.0, label='Classifier-free Guidance Scale')
                    top_k = gr.Slider(minimum=1, maximum=16384, step=1, value=4000, label='Top-K')
                    top_p = gr.Slider(minimum=0., maximum=1.0, step=0.1, value=1.0, label="Top-P")
                    temperature = gr.Slider(minimum=0., maximum=1.0, step=0.1, value=1.0, label='Temperature')
                    seed = gr.Slider(minimum=0, maximum=1000, step=1, value=42, label='Seed')
                    # seed = gr.Number(value=0, label='Seed')
                    button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    output = gr.Gallery(label='Generated Images', height=700)
                    button.click(infer, inputs=[cfg_scale, top_k, top_p, temperature, i1k_class, seed], outputs=[output])
    demo.queue()
    demo.launch(debug=True)
