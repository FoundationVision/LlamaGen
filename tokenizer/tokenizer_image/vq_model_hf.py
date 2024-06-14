from huggingface_hub import PyTorchModelHubMixin

from tokenizer.tokenizer_image.vq_model import ModelArgs, VQModel

class VQModelHF(VQModel, PyTorchModelHubMixin, repo_url="https://github.com/FoundationVision/LlamaGen", license="mit", tags=["llamagen", "text-to-image"]):
    pass

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModelHF(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModelHF(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models_HF = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
