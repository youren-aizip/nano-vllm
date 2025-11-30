import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    weight.shape = (output_size / tp_size, input_size)
    ColumnParallelLinear (parallel along the output/column dimension):

    W.T (used in Y = X @ W.T):
                      output_size (split by column)
              ←───────────────────────────────────────→
                 out/tp   out/tp   out/tp   out/tp
              ┌─────────┬─────────┬─────────┬─────────┐ ↑
              │         │         │         │         │ │
       in     │  GPU0   │  GPU1   │  GPU2   │  GPU3   │ │ in (full)
              │         │         │         │         │ │
              └─────────┴─────────┴─────────┴─────────┘ ↓

    Math:
        Input:      X ∈ R^{batch × in}
        Original:   W ∈ R^{out × in}, W.T ∈ R^{in × out}
        Split:      W.T = [W_0.T | W_1.T | ... | W_{tp-1}.T]  (split by column)
        Compute:    Y_i = X @ W_i.T, Y_i ∈ R^{batch × (out/tp)}
        Result:     Y = concat([Y_0, Y_1, ..., Y_{tp-1}])

    Communication: none (outputs can be directly concatenated)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)  # tp_dim=0

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # load splited weight to the current rank
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        HuggingFace checkpoint stores gate_proj and up_proj separately,
        but nanovllm merges them into a single gate_up_proj for efficiency.

        Weight loading (per GPU):
            HuggingFace Checkpoint:              gate_up_proj on this GPU:
            ┌─────────────────────┐
            │     gate_proj       │              ┌─────────────────────────┐
            │  (inter × hidden)   │  ──────────→ │ gate     │     up       │
            └─────────────────────┘              │(inter/tp)│  (inter/tp)  │
            ┌─────────────────────┐              └─────────────────────────┘
            │      up_proj        │  ──────────→       ↑           ↑
            │  (inter × hidden)   │              shard_id=0   shard_id=1
            └─────────────────────┘

            Step 1: narrow(param_data) to locate gate or up region
            Step 2: chunk(loaded_weight) to get this GPU's slice
            Step 3: copy to param_data
        """
        param_data = param.data
        # the start index of the shard gate or up
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    weight.shape = (output_size, input_size / tp_size)
    RowParallelLinear (parallel along the input/row dimension):

    W.T (used in Y = X @ W.T):
                          output_size (full)
              ←──────────────────────────────────────────→
              ┌──────────────────────────────────────────┐ ↑
              │                  GPU0                    │ │ in/tp
              ├──────────────────────────────────────────┤
              │                  GPU1                    │ │ in/tp
       in     ├──────────────────────────────────────────┤
              │                  GPU2                    │ │ in/tp
              ├──────────────────────────────────────────┤
              │                  GPU3                    │ │ in/tp
              └──────────────────────────────────────────┘ ↓

    Math:
        Input:      X ∈ R^{batch × in}
        Original:   W ∈ R^{out × in}, W.T ∈ R^{in × out}
        Split:      W.T = [W_0.T; W_1.T; ... ; W_{tp-1}.T]  (split by row)
                    X = [X_0 | X_1 | ... | X_{tp-1}], X_i ∈ R^{batch × (in/tp)}
        Compute:    Y_i = X_i @ W_i.T, Y_i ∈ R^{batch × out}
        Result:     Y = Σ Y_i = Y_0 + Y_1 + ... + Y_{tp-1}

    Communication: all-reduce (sum) to get the full output
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)  # tp_dim=1

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
