## serving by vLLM

### Install
```
pip install vllm==0.4.1
```

### Comparison (A100)

Method | params | baseline(s) | vllm(s) | speed-up ratio 
--- |:---:|:---:|:---:|:---:
[GPT-B](./fake_json/GPT-B.json)    | 111M | 7.80    | 2.39      |  326 %
[GPT-L](./fake_json/GPT-L.json)    | 343M | 13.72   | 3.48      |  380 %
[GPT-XL](./fake_json/GPT-XL.json)  | 775M | 19.76   | 4.84      |  408 %
[GPT-XXL](./fake_json/GPT-XXL.json)| 1.4B | 26.38   | 6.36      |  414 %
[GPT-3B](./fake_json/GPT-3B.json)  | 3.1B | 14.73   | 6.26      |  235 %

```
### GPT-B
# 7.80 seconds
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_B_384.pt --image-size 384

# 2.39 seconds
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_B_384.pt --image-size 384


### GPT-L
# 13.72 seconds
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_L_384.pt --gpt-model GPT-L --image-size 384

# 3.48 seconds
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_L_384.pt --gpt-model GPT-L --image-size 384


### GPT-XL
# 19.76 seconds
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XL_384.pt --gpt-model GPT-XL --image-size 384

# 4.84 seconds
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XL_384.pt --gpt-model GPT-XL --image-size 384


### GPT-XXL
# 26.38 seconds
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL_384.pt --from-fsdp --gpt-model GPT-XXL --image-size 384

# 6.36 seconds
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL_384.pt --from-fsdp --gpt-model GPT-XXL --image-size 384


### GPT-3B
# 14.73 seconds
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_3B_384.pt --from-fsdp --gpt-model GPT-3B --image-size 384

# 6.26 seconds
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_3B_384.pt --from-fsdp --gpt-model GPT-3B --image-size 384

```
