"""
Rotary Position Embedding (RoPE)
Used by LLaMA, Mistral, GPT-NeoX
Fixes positional encoding collapse issues
"""
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute rotation frequencies for RoPE
    
    Args:
        dim: embedding dimension (must be even)
        seq_len: max sequence length
        theta: base for frequency computation
    
    Returns:
        Complex tensor of shape [seq_len, dim/2]
    """
    # Compute frequencies for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices
    t = torch.arange(seq_len, dtype=torch.float32)
    
    # Outer product: positions × frequencies
    freqs = torch.outer(t, freqs)
    
    # Convert to complex numbers (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to queries and keys
    
    Args:
        xq: queries [batch, seq_len, n_heads, head_dim]
        xk: keys [batch, seq_len, n_heads, head_dim]
        freqs_cis: precomputed frequencies [seq_len, head_dim/2]
    
    Returns:
        Rotated queries and keys
    """
    # Reshape to complex numbers (pairs of dimensions)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Get frequencies for current sequence length
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    
    # Apply rotation (complex multiplication)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)