# Modified from:
#   stylegan2-pytorch: https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
#   stylegan2-pytorch: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
#   maskgit: https://github.com/google-research/maskgit/blob/main/maskgit/nets/discriminator.py
import math
import torch
import torch.nn as nn
try:
    from kornia.filters import filter2d
except:
    pass

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, channel_multiplier=1, image_size=256):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        log_size = int(math.log(image_size, 2))
        in_channel = channels[image_size]

        blocks = [nn.Conv2d(input_nc, in_channel, 3, padding=1), leaky_relu()]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(DiscriminatorBlock(in_channel, out_channel))
            in_channel = out_channel
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channel, channels[4], 3, padding=1),
            leaky_relu(),
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            leaky_relu(),
            nn.Linear(channels[4], 1)
        )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x



class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def exists(val):
    return val is not None
