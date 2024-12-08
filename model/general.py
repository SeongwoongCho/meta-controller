import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import repeat, rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import init_weights_vit_timm
from model.custom_layers import Identity, Linear, LayerNorm, MLP, LayerScale
from typing import Union


class AttentionBlock(nn.Module):
    """
    Task-specific Attention Block
    """

    def __init__(self, dim_q, dim_k, dim_v, dim, num_heads, attn_drop=0., proj_drop=0., n_tsparam=0, lora_rank=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        proj_fn = partial(Linear, n_tsparam=n_tsparam, lora_rank=lora_rank, out_features=dim, bias=True)
        
        self.q = proj_fn(in_features=dim_q)
        self.k = proj_fn(in_features=dim_k)
        self.v = proj_fn(in_features=dim_v)
        self.proj = proj_fn(in_features=dim)

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def forward(self, q, k, v, t_idx=None, attn_mask=None, pos_bias=None, is_causal=False):
        """
        Params: 
            q:         torch.Tensor     (B, L, dim_q)
            k:         torch.Tensor     (B, S, dim_k) 
            v:         torch.Tensor     (B, S, dim_v)
            t_idx:     torch.LongTensor (B,)
            attn_mask: torch.BoolTensor (B, L, S) or None 
                        >> A boolean mask where a value of True indicates that the element should take part in attention. 
            pos_bias:  torch.FloatTensor(B, nH, L, S) or None
                        >> A position bias between query and key
        
        Outputs:
            o:         torch.Tensor     (B, L, dim) 
        """

        assert attn_mask is None or attn_mask.dtype == torch.bool 
        
        q = self.q(q, t_idx=t_idx)
        k = self.k(k, t_idx=t_idx)
        v = self.v(v, t_idx=t_idx)
    
        # Split dimension by num_heads chunks (B, *, (nH d)) ->  ((nH B), *, d)
        q = torch.cat(q.split(self.dim // self.num_heads, dim=-1), dim=0) 
        k = torch.cat(k.split(self.dim // self.num_heads, dim=-1), dim=0)
        v = torch.cat(v.split(self.dim // self.num_heads, dim=-1), dim=0)
            
        if attn_mask is not None:
            attn_mask = repeat(attn_mask, 'B ... -> (nH B) ...', nH=self.num_heads)

        if is_causal:
            causal_mask = torch.ones(q.size(1), q.size(1), dtype=torch.bool, device=q.device).tril(diagonal=0)
            causal_mask = F.pad(causal_mask, (k.size(1) - q.size(1), 0), value=True)
            if attn_mask is None:
                attn_mask = causal_mask
                attn_mask = repeat(attn_mask, '... -> B ...', B=q.size(0))
            else:
                attn_mask = attn_mask * causal_mask[None]

        # handling all infinities in attn_mask
        if attn_mask is not None:
            attn_mask = torch.where(attn_mask.float().sum(dim=2, keepdim=True).repeat(1, 1, attn_mask.shape[2]).bool(),
                                    attn_mask, torch.ones_like(attn_mask))

        # augment attn_mask
        if pos_bias is not None:
            pos_bias = rearrange(pos_bias, 'B nH ... -> (nH B) ...')
            if attn_mask is None:
                attn_mask = pos_bias
            else:
                attn_mask = torch.where(attn_mask, pos_bias, torch.ones_like(pos_bias) * float('-inf'))

        # scaled dot product attention
        dropout_p = self.attn_drop if self.training else 0
        o = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=dropout_p)
        
        # merge heads & proj.
        o = rearrange(o, '(nH B) L d -> B L (nH d)', nH=self.num_heads)  
        o = self.proj(o, t_idx=t_idx)
        dropout_p = self.proj_drop if self.training else 0
        o = F.dropout(o, p=dropout_p) 
        return o


class SelfAttentionBlock(nn.Module):
    """
    Task-specific Self-Attention Block, 
    which is simliar to Blocks in timm Vision Transformer (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., attn_drop=0., drop=0., drop_path=0., init_values=None, act_layer=nn.GELU, n_tsparam=0, lora_rank=0):
        super().__init__()
        self.norm1 = LayerNorm(n_tsparam=n_tsparam, normalized_shape=dim)        
        self.attn = AttentionBlock(dim_q=dim, dim_k=dim, dim_v=dim, dim=dim, num_heads=num_heads, attn_drop=attn_drop,
                                     proj_drop=drop, n_tsparam=n_tsparam, lora_rank=lora_rank)
        self.ls1 = LayerScale(n_tsparam, dim, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else Identity()
        
        self.norm2 = LayerNorm(n_tsparam=n_tsparam, normalized_shape=dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                       n_tsparam=n_tsparam, lora_rank=lora_rank)
        self.ls2 = LayerScale(n_tsparam, dim, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x, t_idx=None, attn_mask=None, pos_bias=None, is_causal=False):
        res = x
        x = self.norm1(x, t_idx=t_idx)
        x = self.attn(q=x, k=x, v=x, t_idx=t_idx, attn_mask=attn_mask, pos_bias=pos_bias, is_causal=is_causal)
        x = self.ls1(x, t_idx=t_idx)
        x = self.drop_path1(x)
        x = x + res

        res = x
        x = self.norm2(x, t_idx=t_idx)
        x = self.mlp(x, t_idx=t_idx)
        x = self.ls2(x, t_idx=t_idx)
        x = self.drop_path2(x)
        x = x + res
        return x


class Transformer(nn.Module):
    """
    Spatio-Temporal Transformer with spatial and temporal position biases
    """
    def __init__(self, n_blocks, num_heads, hidden_dim, attn_drop=0.1, drop=0.1, drop_path=0.1, n_tsparam=0, lora_rank=0, layerscale=None):
        super().__init__()

        self.norm_pre = nn.LayerNorm(hidden_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=dpr[blk_idx],
                n_tsparam=n_tsparam,
                lora_rank=lora_rank,
                init_values=layerscale,
            )
            for blk_idx in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.apply(init_weights_vit_timm)

    def lora_parameters(self):
        for block in self.blocks:
            for name, param in block.named_parameters():
                if 'lora' in name:
                    yield param

    def bias_parameters(self):
        for block in self.blocks:
            for name, param in block.named_parameters():
                if name.endswith('bias'):
                    yield param

    def layerscale_parameters(self):
        for block in self.blocks:
            for name, param in block.named_parameters():
                if 'ls1' in name or 'ls2' in name:
                    yield param

    def task_specific_parameters(self):
        for p in self.lora_parameters():
            yield p
        
        for p in self.bias_parameters():
            if p.ndim == 2:
                yield p
        
        for p in self.layerscale_parameters():
            if p.ndim == 2:
                yield p

    def forward(self, x, t_idx=None, attn_mask=None, is_causal=False):
        """
        Params:
            x:          torch.Tensor     (B, N, d)
            t_idx:      torch.LongTensor (B,)

        Outputs:
            x:          torch.Tensor     (B, N, d)
        """
        x = self.norm_pre(x)
        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t_idx=t_idx, attn_mask=attn_mask, is_causal=is_causal)
        x = self.norm_out(x)

        return x
