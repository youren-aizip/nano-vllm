import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-Max trick
        # if E_i ~ Exp(1), then the distribution of argmax(p_i / E_i) is the same as the sampling from [p_1, p_2, ...]
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
