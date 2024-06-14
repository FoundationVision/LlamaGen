from autoregressive.models.gpt import ModelArgs, Transformer
from huggingface_hub import PyTorchModelHubMixin


class TransformerHF(Transformer, PyTorchModelHubMixin, repo_url="https://github.com/FoundationVision/LlamaGen", license="mit", tags=["llamagen", "text-to-image"]):
    pass


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return TransformerHF(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return TransformerHF(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return TransformerHF(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return TransformerHF(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return TransformerHF(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return TransformerHF(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return TransformerHF(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return TransformerHF(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models_HF = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}
