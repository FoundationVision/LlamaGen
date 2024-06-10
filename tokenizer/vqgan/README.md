## Pretrained VQVAE Models

### install
```
pip install omegaconf
pip install einops
```
* download all needed models from https://github.com/CompVis/taming-transformers and put in pretrained_models/
* pip install pytorch_lightning
* python3 tools/convert_pytorch_lightning_to_torch.py
* pip uninstall pytorch_lightning

### demo
```
cd ${THIS_REPO_ROOT}
python3 tokenizer/vqgan/taming_vqgan_demo.py
```

### acknowledge
Codes in this folder are modified from from https://github.com/CompVis/taming-transformers

