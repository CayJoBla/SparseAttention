import torch
from pytorch_block_sparse import BlockSparseMatrix


def get_sliding_window(weights, window_size=4):
    """
    Keep only the weights within a specified window along the diagonals of the attention weight tensor.

    Parameters:
        weight (torch.Tensor): Attention weight tensor
        window_size (int): Size of the window along the diagonals

    Returns:
        (torch.Tensor): Sparse attention weight tensor with values outside the window set to 0
    """
    assert weights.dim() == 2, "Input tensor must be 2D"
    assert weights.size(0) == weights.size(1), "Input tensor must be square"
    assert window_size % 2 == 0, "Window size must be an even number"

    # Create mask
    idx = torch.arange(weights.size(0))
    diag_idx = idx.unsqueeze(0) - idx.unsqueeze(1)
    mask = diag_idx.abs() < window_size//2 + 1

    return weights * mask.float()


def dense_sparse_convert(model, sparsify_fn, **kwargs):
    """Replace the dense attention weights in a model with sparse ones."""
    base_model = model.base_model  # May not need this
    for bert_layer in base_model.encoder.layer:
        attn = bert_layer.attention.self
        attn.query.weight.data = sparsify_fn(attn.query.weight.data, **kwargs)
        attn.key.weight.data = sparsify_fn(attn.key.weight.data, **kwargs)
        attn.value.weight.data = sparsify_fn(attn.value.weight.data, **kwargs)
    return model


def get_blocks(weights, block_shape):
    """Returns the block indices for a sliding window pseudo-sparse weight matrix."""
    window_size = compute_window_size(weights)
    half_window = window_size // 2

    assert half_window % block_shape[0] == 0, "Half window must be divisible by block shape"
    blocks_per_window = half_window // block_shape[0]

    # Create block mask
    X, Y = BlockSparseMatrix.blocks_count_(weights.size(), block_shape)
    diag_idx = torch.arange(X).unsqueeze(0) - torch.arange(Y).unsqueeze(1)
    return (diag_idx.abs() <= blocks_per_window).nonzero()


def compute_window_size(weights):
    """Compute the window size for a psuedo-sparse weight matrix."""
    assert weights.dim() == 2, "Input tensor must be 2D"
    assert weights.size(0) > 0 and weights.size(1) > 0, "Input tensor must be non-empty"
    
    return (weights[0].nonzero().max().item()) * 2


def compute_density(sparse_weights):
    """Compute the density of a psuedo-sparse weight matrix."""
    total_size = sparse_weights.shape[0] * sparse_weights.shape[1]
    sparse_size = sparse_weights.data.shape[0]  * sparse_weights.data.shape[1] 
    return sparse_size / total_size
