# Evaluations from [GigaGAN](https://github.com/mingukkang/GigaGAN/tree/main/evaluation)

```
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch
pip install clean_fid
```

```
python3 evaluations/t2i/evaluation.py \
--eval_res 256 \ 
--batch_size 256 \
--how_many 30000 \
--ref_data "coco2014" \
--ref_type "val2014" \
--eval_res 256 \
--batch_size 256 \
--ref_dir "/path/to/coco" \
--fake_dir "/path/to/generation" \
$@
```
