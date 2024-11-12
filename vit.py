"""Vision Transformer implementation."""
from typing import Optional

import torch
from torch import nn
from math import pi
import einops


class MHA(nn.Module):
    """Multi-head attention in transformer using flash attention https://arxiv.org/abs/2205.14135."""
    def __init__(
        self,
        dim_query_in: int,
        dim_kv_in: int,
        dim_out: int,
        n_heads: int,
        dropout_p: float = 0.0,
        bias=False,
    ):
        """Initialize.

        In this case we do not need the sequence length since the pytorch flash attention
        implementation is dynamic w.r.t. the sequence length.
        """
        assert dim_query_out % n_heads == 0
        assert dim_value_out % n_heads == 0
        assert 1 > dropout_p >= 0, "dropout_p must be a probability smaller than 1."
        super().__init__()

        self._n_heads = n_heads

        # Transformation to be applied to the input used to make queries.
        self._query_transf = nn.Linear(dim_query_in, dim_out, bias=bias)

        # Dimensionality of each separate query in the multiple heads.
        self._dim_out = dim_out
        self._head_dim = dim_out//n_heads

        # The attention product requires the keys and queries to have the same dimensionality.
        self._key_transf = nn.Linear(dim_kv_in, dim_out, bias=bias)
        self._value_transf = nn.Linear(dim_kv_in, dim_out, bias=bias)

        self._out_transf = nn.Linear(dim_value_out, dim_out, bias=True)
        self._dropout_p = dropout_p

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass."""
        # We need to transform the keys queries and values to be able to apply multiple attention
        # heads.
        B, NQ, _ = q_in.shape

        queries = self._query_transf(q_in)
        queries = queries.view(B, NQ, self._n_heads, self._head_dim).transpose(1, 2)

        _, NKV, _ = kv_in.shape
        keys = self._key_transf(kv_in)
    
        values = self._value_transf(kv_in)

        keys = keys.view(B, NKV, self._n_heads, self._head_dim).transpose(1, 2)
        values = values.view(B, NKV, self._n_heads, self._head_dim).transpose(1, 2)

        # Torch implementation of flash attention.
        out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=attention_mask, dropout_p=self._dropout_p if self.training else 0, is_causal=False)

        # Concatenate the channels from the multiple heads before applying the final linear
        # projection.
        out = out.transpose(1, 2).contiguous().view(B, NQ, self._dim_out)
        return self._out_transf(out)




class DropPath(nn.Module):
    """Randomly drop compute path."""
    def __init__(self, drop_p: float):
        super().__init__()

        def _drop_path(x, drop_p: float, training: bool = False):
            if not training:
                return x
            keep_p = 1 - drop_p
            shape = (x.shape[0],) + (1,)*(x.ndim - 1)
        
            # Add random numbers in [0, 1[ and add keep_p to it and then convert it to 0 or 1. If keep_p
            # is 1 it will always be 1.
            threshold = keep_p + torch.rand(shape, dtype=x.dtype, device=x.device)
            threshold.floor_()
        
            # Keep total tensor norm the the same by dividing by the keep probability.
            output = torch.div(x, keep_p) * threshold
            return output

        self._drop = lambda x, t: _drop_path(x, drop_p=drop_p, training=t) if drop_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self._drop(x)


class MLP(nn.Module):
    """Feedforward network."""
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout_p: float = 0.):
        super().__init__()
        self._lin_1 = nn.Linear(dim_in, dim_hidden)
        self._lin_2 = nn.Linear(dim_hidden, dim_out)
        def _do(x: torch.Tensor):
            """Dropout."""
            return torch.nn.functional.dropout(x, p=dropout_p, training=self.training)
        self._do = _do

    def forward(self, x):
        # Only apply activation in expanded space.
        x = x + self._do(torch.nn.functional.gelu(self._lin_1(x)))
        x = x + self._do(self._lin_2(x))


class SelfAttentionBlock(nn.Module):
    """Implementation of a transformer block, which contains a self-attention layer."""
    def __init__(self, dim: int, ffn_expansion_factor: int = 4, mlp_dropout_p: float = 0, drop_path_p: float = 0, **attn_kwargs):
        """Initialize the transformer block.

        Postnormalized transformer block.

        Args:
            ...mostly obvious...
        """
        assert 1 > drop_path_p >= 0, "drop_path_p must be a probability smaller than 1."
        assert 1 > mlp_dropout_p >= 0, "mlp_dropout_p must be a probability smaller than 1."
        super().__init__()
        self._attn = MHA(
            dim_query_in=dim,
            dim_kv_in=dim,
            dim_out=dim,
            **attn_kwargs
        )

        self._ffn = MLP(dim_in=dim, dim_hidden=ffn_expansion_factor*dim, dim_out=dim, dropout_p=mlp_dropout_p)
        self._ln_attn = nn.LayerNorm([dim])
        self._ln_ffn = nn.LayerNorm([dim])
        self._drop_path = DropPath(drop_p=drop_path_p)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        normed = self._ln_attn(x)
        x = x + self._drop_path(self._attn(normed, normed))
        x = x + self._drop_path(self._ffn(x))
        return x


# TODO: For very large image sizes we might need to run a somewhat more substantial CNN before
# going into the transformer to not get hit too much by the O(N^2) scaling in the number of pixels.
class Patchify(nn.Module):
    """Patchify an image and encode it to a sequence."""
    def __init__(self, im_h: int, im_w: int, patch_h: int, patch_w: int, dim: int):
        assert im_h % patch_h == 0, "Image and patch heights must be divisible."
        assert im_w % patch_w == 0, "Image and patch widths must be divisible."
        super().__init__()
        self._proj = nn.Conv2D(3, dim, kernel_size=(path_h, patch_w), stride=(patch_h, patch_w))

    def forward(self, x: torch.Tensor):
        return einops.rearrange(self._proj(x), "b c h w -> b (h w) c")



#class ViT(nn.Module):
#    """Vision Transformer."""
#    def __init__(self, dim: 
