from pytorch_block_sparse import BlockSparseLinear, BlockSparseMatrix
from pytorch_block_sparse.block_sparse_linear import BlockSparseLinearFunction
from torch import nn
from sparsify import get_blocks
import torch
import math

class BlockSparseFromLinear(BlockSparseLinear):
    OPTIMIZED_BLOCK_SIZE = 32

    def __init__(
        self,
        torch_nn_linear,        # Only use values from the linear layer (no in_size, out_size, bias, etc.)
        verbose = False,
        **kwargs                # Pass any additional arguments to the sparsify function
    ):
        super(BlockSparseLinear, self).__init__()
        self.fn = BlockSparseLinearFunction.apply
        self.verbose = verbose
        self.block_shape = (self.OPTIMIZED_BLOCK_SIZE, self.OPTIMIZED_BLOCK_SIZE)   # Modify to always set block size to optimal (32x32)
        self._optimized = (
            self.block_shape[0] == self.OPTIMIZED_BLOCK_SIZE and self.block_shape[1] == self.OPTIMIZED_BLOCK_SIZE
        )

        # Modify to use values from the linear layer
        in_features = torch_nn_linear.in_features
        out_features = torch_nn_linear.out_features
        bias = torch_nn_linear.bias is not None
        weight = torch_nn_linear.weight.data

        if in_features % self.block_shape[1] != 0:
            raise Exception(
                f"BlockSparseLinear invalid block_shape={self.block_shape[1]}, should be divisor of {in_features}"
            )
        if out_features % self.block_shape[0] != 0:
            raise Exception(
                f"BlockSparseLinear invalid block_shape={self.block_shape[0]}, should be divisor of {out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features

        # Get the block indices and number of blocks
        blocks = get_blocks(weight, self.block_shape)
        self.block_count = len(blocks)
        
        with torch.no_grad():
            weight = BlockSparseMatrix.from_dense(weight, self.block_shape, self.block_count, blocks=blocks)

        # Why do this?
        # density = torch_nn_linear.weight.data.numel() / (self.block_shape[0] * self.block_shape[1] * self.block_count)
        # weight.multiply_(1.0 / math.sqrt(density))

        self.weight = weight

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))
            with torch.no_grad():
                self.bias.copy_(torch_nn_linear.bias)
        else:
            self.register_parameter("bias", None)


def sparse_convert(sparse_model, **kwargs):
    """Convert the psuedo-sparse attention weights in a model to block sparse ones."""
    base_sparse_model = sparse_model.base_model
    for layer in base_sparse_model.encoder.layer:
        attn = layer.attention.self
        attn.query = BlockSparseFromLinear(attn.query, **kwargs)
        attn.key = BlockSparseFromLinear(attn.key, **kwargs)
        attn.value = BlockSparseFromLinear(attn.value, **kwargs)
    return sparse_model