"""Positional encodings.

Positional encodings are the Fourier encodings from the original transformer paper
https://arxiv.org/abs/1706.03762.

We assume we are working with 2D image tensors of fixed size so that everything can be completely
precomputed.
"""
import torch
import math
from torch import nn

class PE2D(nn.Module):
    """2D positional encoding for images.

    If the dimensionality and image resolution does not change we can precompute almost everything.

    The original formulation of PEs in "Attention is all you need" include a term 1e4**(2i/d) which
    can become numerically unstable for very large d values since we raise 1e4 to a very small
    power. We reformulate it for numerical safety as follows:

    max_fq**(2i/d) = exp(log(max_fq**(2i/d))) = exp((2i/d)*log(max_fq)
    """
    def __init__(self, height:int, width: int, dim: int, max_fq=1e4):
        assert dim%4 == 0
        super().__init__()
        log = math.log(max_fq) # Scalar
        fqs = torch.arange(dim//4, dtype=torch.float) # (dim//4,)

        # Note that dim is replaced by dim//2 because we have two dimensions and will concatenate
        # their positional encodings in the end.
        fqs = torch.exp(2*(fqs/(dim//2))*log).unsqueeze(0) # (1, dim//4)

        h = torch.arange(height, dtype=torch.float).unsqueeze(1) # (h, 1)
        pe_h = (h/fqs) #(h, dim//4)
        pe_h = torch.cat([torch.sin(pe_h), torch.cos(pe_h)], dim=-1) #(h, dim//2)

        w = torch.arange(width, dtype=torch.float).unsqueeze(1) #(w, 1)
        pe_w = (w/fqs) #(w, dim//4)
        pe_w = torch.cat([torch.sin(pe_w), torch.cos(pe_w)], dim=-1) #(w, dim//2)

        pe_h = pe_h.unsqueeze(1).repeat(1, width, 1) # (h, w, dim//2)
        pe_w = pe_w.unsqueeze(0).repeat(height, 1, 1) # (h, w, dim//2)

        # Concat and also already add a batch axis which can then be broadcasted automatically
        # in the forward function.
        pe_2d = torch.cat([pe_h, pe_w], dim=-1).unsqueeze(0) #(1, h, w, dim)
        self.register_buffer("_pe_2d", pe_2d)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Positionally encode image batch.

        Args:
            x: Image batch of shape (B, H, W, C)

        Returns:
            Image batch with PEs summed at every location. Shape (B, H, W, C).
        """
        return x + self._pe_2d

if __name__ == "__main__":
    h, w = 100, 100
    d = 32
    pe = PE2D(h, w, d, max_fq=1e4)

    # Test that it works.
    pe(torch.randn(10, 100, 100, 32))
    
    # Plot some PEs
    pe_h = pe._pe_2d[0, :, 0, :d//4]
    import matplotlib.pyplot as plt
    for i in range(d//4):
        plt.plot(torch.arange(h), pe_h[..., i])
    # Verify correctness, 7/8 comes from 32//4 and then the fact that arange stops at 1 off the end.
    #plt.plot(torch.arange(h), pe_h[..., d//4 - 1])
    #plt.plot(torch.arange(h), torch.sin(torch.arange(h)/(10000**(7/8))))
       
    plt.xlabel("Pixel height position")
    plt.ylabel("Sine height positional encoding")
    plt.show()
