import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    weight.shape = (vocab_size / tp_size, embedding_dim)
    VocabParallelEmbedding (parallel along the vocab dimension):

                   embed_dim (full)
               ←───────────────────────→
             ┌───────────────────────────┐ ↑
             │          GPU 0            │ │ vocab/tp
             ├───────────────────────────┤
             │          GPU 1            │ │ vocab/tp
      vocab  ├───────────────────────────┤
             │          GPU 2            │ │ vocab/tp
             ├───────────────────────────┤
             │          GPU 3            │ │ vocab/tp
             └───────────────────────────┘ ↓

    Operation: Embedding lookup

    Example (vocab_size=400000, tp=4):
        Input:  token_ids = [3, 100005, 50, 200000]
        GPU 0 (vocab 0-99999):       [emb_3,     0,       emb_50,    0       ]
        GPU 1 (vocab 100000-199999): [  0,    emb_100005,    0,      0       ]
        GPU 2 (vocab 200000-299999): [  0,       0,          0,   emb_200000 ]
        GPU 3 (vocab 300000-399999): [  0,       0,          0,      0       ]
        Result: all_reduce(sum) → [emb_3, emb_100005, emb_50, emb_200000]

    Communication: all-reduce (sum)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # mask out the input_ids that are not in the current rank
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    weight.shape = (vocab_size / tp_size, embedding_dim)
    ParallelLMHead (parallel along the vocab/column dimension):

    Same weight layout as VocabParallelEmbedding, but uses linear instead of lookup.
    When used as linear (Y = X @ W.T), the weight is transposed, so we split by column:

                      vocab_size (split by column)
               ←────────────────────────────────────────→
                  vocab/tp  vocab/tp  vocab/tp  vocab/tp
               ┌─────────┬─────────┬─────────┬─────────┐ ↑
               │         │         │         │         │ │
      embed    │  GPU0   │  GPU1   │  GPU2   │  GPU3   │ │ embed_dim (full)
               │         │         │         │         │ │
               └─────────┴─────────┴─────────┴─────────┘ ↓

    Math:
        Input:      X ∈ R^{batch × embed_dim}
        Original:   W ∈ R^{vocab × embed_dim}, W.T ∈ R^{embed_dim × vocab}
        Split:      W.T = [W_0.T | W_1.T | ... | W_{tp-1}.T]  (split by column)
        Compute:    logits_i = X @ W_i.T, logits_i ∈ R^{batch × (vocab/tp)}
        Result:     logits = concat([logits_0, logits_1, ..., logits_{tp-1}])

    Communication: gather + concat (only on rank 0) - similar to ColumnParallel
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # Prefill: x.shape = (total_tokens, hidden_dim)
        # Decode: x.shape = (batch_size, hidden_dim)
        context = get_context()
        if context.is_prefill:
            # the head only needs the last token of each sequence
            # cu_seqlens_q is the cumulative sum of the sequence lengths
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)  # gather the logits from all ranks to rank 0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
