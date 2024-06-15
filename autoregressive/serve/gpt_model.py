from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

from vllm.attention import AttentionMetadata
from vllm.attention import Attention as pagedAttention

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from autoregressive.serve.sampler import Sampler

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'
    cfg_scale: float = 4.0

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    # def token_drop(self, labels, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     labels = torch.where(drop_ids, self.num_classes, labels)
    #     return labels

    # def forward(self, labels, train, force_drop_ids=None):
    def forward(self, labels):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                  GPT Model                                    #
#################################################################################
# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        # self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        # self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w_merged = nn.Linear(config.dim, hidden_dim * 2, bias=False)
        self.act_fn = SiluAndMul()

        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        # self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    # def forward(self, x):
    #     return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    def forward(self, x):
        x = self.w_merged(x)
        x = self.act_fn(x)
        x = self.w2(x)
        # return self.ffn_dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        # pagedAttention
        if config.dim // config.n_head == 100:
            self.attn = None  # for this case, we need to overwrite the attn in AttentionMonkeyPatch
        else:
            self.attn = pagedAttention(self.n_head, self.head_dim, self.head_dim**-0.5, num_kv_heads=self.n_kv_head)

        # 2d rotary pos embedding
        grid_size = int(config.block_size ** 0.5)
        assert grid_size * grid_size == config.block_size
        freqs_cis = precompute_freqs_cis_2d(grid_size, config.dim // config.n_head, config.rope_base, config.cls_token_num)
        self.register_buffer('freqs_cis', freqs_cis)


    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor, 
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):  
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(*xq.shape[:-1], 1, self.n_head, self.head_dim)
        xk = xk.view(*xk.shape[:-1], 1, self.n_kv_head, self.head_dim)
        freqs_cis = self.freqs_cis[positions].unsqueeze(1)        
        xq = apply_rotary_emb_bs(xq, freqs_cis)
        xk = apply_rotary_emb_bs(xk, freqs_cis)
        xq = xq.flatten(1)
        xk = xk.flatten(1)

        output = self.attn(xq, xk, xv, kv_cache, attn_metadata)
        output = self.wo(output)
        
        return output


class AttentionMonkeyPatch(Attention):
    """
    Note:
    In vllm, PagedAttention supports head sizes [64, 80, 96, 112, 128, 256].
    However, LlamaGen-3B model has head size 100 (for some historical reasons).
    Here we hack Attnetion to enable vllm support head size 100.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # overwrite PagedAttention
        # hard-coded 112 for LlamaGen-3B model
        self.attn = pagedAttention(self.n_head, 112, 100**-0.5, num_kv_heads=self.n_kv_head)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(*xq.shape[:-1], 1, self.n_head, self.head_dim)
        xk = xk.view(*xk.shape[:-1], 1, self.n_kv_head, self.head_dim)
        freqs_cis = self.freqs_cis[positions].unsqueeze(1)
        xq = apply_rotary_emb_bs(xq, freqs_cis)
        xk = apply_rotary_emb_bs(xk, freqs_cis)
        xq = xq.flatten(1)
        xk = xk.flatten(1)
        ############ padding to 112 to make vllm happy ############
        zero_pad = torch.zeros(xq.shape[0], self.n_head, 112 - 100, device=xq.device, dtype=xq.dtype)
        xq = xq.reshape(xq.shape[0], self.n_head, self.head_dim)
        xk = xk.reshape(xk.shape[0], self.n_kv_head, self.head_dim)
        xv = xv.reshape(xv.shape[0], self.n_kv_head, self.head_dim)
        xq = torch.concat([xq, zero_pad], dim=-1).flatten(1)
        xk = torch.concat([xk, zero_pad], dim=-1).flatten(1)
        xv = torch.concat([xv, zero_pad], dim=-1).flatten(1)

        output = self.attn(xq, xk, xv, kv_cache, attn_metadata)
        ############ de-padding to 100 ############
        output = output.reshape(output.shape[0], self.n_head, 112)
        output = output[..., :100].flatten(1)

        output = self.wo(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        if config.dim // config.n_head == 100:
            self.attention = AttentionMonkeyPatch(config)
        else:
            self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: AttentionMetadata):
        h = x + self.attention(self.attention_norm(x), positions, kv_cache, attn_metadata)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
        

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.cfg_scale = config.cfg_scale
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        else:
            raise Exception("vllm only supports c2i now, please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.sampler = Sampler(config.cfg_scale)

    def forward(
        self, 
        input_ids: torch.Tensor=None,
        positions: torch.Tensor=None,
        kv_caches: List[torch.Tensor]=None,
        attn_metadata: AttentionMetadata=None,
    ):
        # if positions.max() == 0: # prefill in inference
        #     token_embeddings = self.cls_embedding(input_ids)
        # else: # decode_n_tokens(kv cache) in inference
        #     token_embeddings = self.tok_embeddings(input_ids)
        cond_ids = torch.clamp(input_ids, max=self.num_classes)
        token_embeddings = self.cls_embedding(cond_ids) * (positions.max() == 0) + \
            self.tok_embeddings(input_ids) * (positions.max() != 0)

        hh = token_embeddings
        # transformer blocks
        for layer_id, layer in enumerate(self.layers):
            hh = layer(hh, positions, kv_caches[layer_id], attn_metadata)
        
        # output layers
        hh = self.norm(hh)
        return hh

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.output.weight, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
        

    def custom_load_state_dict(self, model_weights):
        model_weights = model_weights.copy()
        for layer_id in range(len(self.layers)):
            branch1 = f'layers.{layer_id}.feed_forward.w1.weight'
            branch3 = f'layers.{layer_id}.feed_forward.w3.weight'
            branch_merged = f'layers.{layer_id}.feed_forward.w_merged.weight'
            model_weights[branch_merged] = torch.cat(
                [model_weights[branch1], model_weights[branch3]], dim=0
            )
            model_weights.pop(branch1)
            model_weights.pop(branch3)

        if 'freqs_cis' in model_weights:
            model_weights.pop('freqs_cis')
        
        self.load_state_dict(model_weights, strict=False)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def apply_rotary_emb_bs(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(xshaped.size(0), xshaped.size(1), 1, xshaped.size(3), 2) # (bs, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}