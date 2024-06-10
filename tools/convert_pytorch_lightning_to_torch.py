import os
import torch

MODEL_PATH = 'pretrained_models'
pt_lightnings = [
    'vqgan_imagenet_f16_1024/ckpts/last.ckpt',
    'vqgan_imagenet_f16_16384/ckpts/last.ckpt',
    'vq-f8-n256/model.ckpt',
    'vq-f8/model.ckpt',
]
pts = [
    'vqgan_imagenet_f16_1024/ckpts/last.pth',
    'vqgan_imagenet_f16_16384/ckpts/last.pth',
    'vq-f8-n256/model.pth',
    'vq-f8/model.pth',
]

for pt_l, pt in zip(pt_lightnings, pts):
    pt_l_weight = torch.load(os.path.join(MODEL_PATH, pt_l), map_location='cpu')
    pt_weight = {
        'state_dict': pt_l_weight['state_dict']
    }
    pt_path = os.path.join(MODEL_PATH, pt)
    torch.save(pt_weight, pt_path)
    print(f'saving to {pt_path}')
