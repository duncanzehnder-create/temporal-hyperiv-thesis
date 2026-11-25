"""Baseline HyperIV: the released networks, read through one interface."""
import torch
from torch import nn

from hyperiv.hyperiv_util import HyperNetwork, SetEmbeddingNetwork


class LayerNorm(nn.LayerNorm):
    """Written out, because the fused kernel gave wrong gradients on our GPU."""

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight + self.bias


def use_plain_attention():
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False


class HyperIV(nn.Module):
    """forward(row, refs, x): today's nine quotes [B, 9, 3] and points x = (k, tau) [B, n, 2] to volatilities."""

    def __init__(self):
        super().__init__()
        use_plain_attention()
        surface = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1), nn.Softplus())
        self.model = HyperNetwork(SetEmbeddingNetwork(3, sum(p.numel() for p in surface.parameters())), surface)
        for layer in self.model._hyper.attention_layers:
            layer.norm1, layer.norm2 = LayerNorm(128), LayerNorm(128)

    def encode(self, refs):
        hyper = self.model._hyper
        e = hyper.fc1(refs)
        for layer in hyper.attention_layers:
            e = layer(e)
        return e.mean(dim=1)

    def weights(self, row, refs):
        return self.model._hyper(refs)

    def decode(self, theta, x):
        params = {name: theta[:, a:b].reshape(-1, *shape) for name, shape, a, b in self.model._params_lookup}
        return torch.vmap(self.model._call, in_dims=(0, 0))(params, x).squeeze(-1)

    def forward(self, row, refs, x):
        return self.decode(self.weights(row, refs), x)
