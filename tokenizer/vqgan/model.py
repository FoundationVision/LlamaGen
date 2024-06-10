import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.vqgan.layer import Encoder, Decoder
from tokenizer.vqgan.quantize import VectorQuantizer2 as VectorQuantizer


VQGAN_FROM_TAMING = {
    'vqgan_imagenet_f16_1024': (
        'tokenizer/vqgan/configs/vqgan_imagenet_f16_1024.yaml',
        'pretrained_models/vqgan_imagenet_f16_1024/ckpts/last.pth'),
    'vqgan_imagenet_f16_16384': (
        'tokenizer/vqgan/configs/vqgan_imagenet_f16_16384.yaml', 
        'pretrained_models/vqgan_imagenet_f16_16384/ckpts/last.pth'),
    'vqgan_openimage_f8_256': (
        'tokenizer/vqgan/configs/vqgan_openimage_f8_256.yaml', 
        'pretrained_models/vq-f8-n256/model.pth'),
    'vqgan_openimage_f8_16384': (
        'tokenizer/vqgan/configs/vqgan_openimage_f8_16384.yaml',
        'pretrained_models/vq-f8/model.pth'),
}

class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 **kwargs,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list(), logging=True):
        model_weight = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(model_weight.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_weight[k]
        missing, unexpected = self.load_state_dict(model_weight, strict=False)
        if logging:
            print(f"Restored from {path}")
            print(f"Missing Keys in State Dict: {missing}")
            print(f"Unexpected Keys in State Dict: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
