import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum    
from einops import repeat, rearrange
from timm.models.layers.helpers import to_2tuple


class Linear(nn.Linear):
    """
    Weight-Switching (LoRa) & Bias-Switching (BitFit) Linear layer
    """
    def __init__(self, n_tsparam=0, lora_rank=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tsparam = n_tsparam
        self.lora_rank = self.low_rank = lora_rank
        self.alpha = 1

        if n_tsparam > 0:
            # bias tuning
            if self.bias is not None:
                self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tsparam).contiguous())

        # lora tuning
        if lora_rank > 0:
            out_features, in_features = self.weight.data.shape
            lora_A = torch.empty((lora_rank, in_features), device=self.weight.data.device, dtype=self.weight.data.dtype)
            lora_B = torch.empty((out_features, lora_rank), device=self.weight.data.device, dtype=self.weight.data.dtype)
            nn.init.zeros_(lora_A)
            nn.init.normal_(lora_B)

            if n_tsparam > 0:
                self.lora_A = nn.Parameter(repeat(lora_A, '... -> T ...', T=n_tsparam).contiguous())
                self.lora_B = nn.Parameter(repeat(lora_B, '... -> T ...', T=n_tsparam).contiguous())
            else:
                self.lora_A = nn.Parameter(lora_A)
                self.lora_B = nn.Parameter(lora_B)

    def forward(self, input, t_idx=None):
        assert input.ndim == 3
        
        if self.n_tsparam > 0:
            if t_idx is None:
                t_idx = torch.zeros(input.shape[0], device=input.device, dtype=torch.long)

            result = F.linear(input, self.weight)
            if self.lora_rank > 0:
                lora_A = self.lora_A[t_idx]
                lora_B = self.lora_B[t_idx]

                res = einsum('brc,bnc->brn', lora_A, input)
                res = einsum('brn,bdr->bnd', res, lora_B)
                result += self.alpha / self.low_rank * res

            if self.bias is not None:
                result += self.bias[t_idx][:, None]

        else:
            result = F.linear(input, self.weight, self.bias)
            
        return result


class LayerNorm(nn.LayerNorm):
    """
    Bias-Switching LayerNorm
    """
    def __init__(self, n_tsparam=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_tsparam = 0
        
        self.n_tsparam = n_tsparam
        if self.n_tsparam > 0:
            assert self.elementwise_affine
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tsparam).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tsparam > 0:
            if t_idx is None:
                t_idx = torch.zeros(input.shape[0], device=input.device, dtype=torch.long)
            output = F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)
                
            assert t_idx.ndim == 1
            bias = self.bias[t_idx]
            for _ in range(output.ndim - 2):
                bias = bias[:, None]
            return output + bias
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    """
    Bias-Switching MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True,
                 drop=0., n_tsparam=0, lora_rank=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = Linear(n_tsparam, lora_rank, in_features=in_features, out_features=hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = Linear(n_tsparam, lora_rank, in_features=hidden_features, out_features=out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, t_idx=None):
        x = self.fc1(x, t_idx=t_idx)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, t_idx=t_idx)
        x = self.drop2(x)
        return x


class Conv2d(nn.Conv2d):
    """
    Bias-Switching Conv2d layer
    """
    def __init__(self, n_tsparam=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is None:
            n_tsparam = 0

        self.n_tsparam = n_tsparam
        if self.n_tsparam > 0:
            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tsparam).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tsparam > 0:
            assert t_idx is not None
            output = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            if t_idx.ndim == 1:
                bias = self.bias[t_idx][:, :, None, None]
            else:
                raise NotImplementedError

            return output + bias
        else:
            return F.conv2d(input, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)


class Identity(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x


class LayerScale(nn.Module):
    def __init__(self, n_tsparams, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.n_tsparams = n_tsparams
        self.inplace = inplace
        if self.n_tsparams > 0:
            self.gamma = nn.Parameter(init_values * torch.ones(n_tsparams, dim))
        else:
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, t_idx=None):
        if self.n_tsparams > 0:
            assert x.ndim == 3
            if t_idx is None:
                t_idx = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            return x * self.gamma[t_idx][:, None]
        else:
            return x.mul_(self.gamma) if self.inplace else x * self.gamma
        

