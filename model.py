"""
Decoder-only transformer language model
Pure PyTorch implementation from scratch
No pretrained weights, no external models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import precompute_freqs_cis, apply_rotary_emb

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings
    Gives the model a sense of token position/order
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal pattern
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
        Returns:
            x with positional information added
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Allows model to focus on different parts of the sequence
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.freqs_cis = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
    
        # Precompute RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.shape[0] < seq_len:
            self.freqs_cis = precompute_freqs_cis(
                dim=self.d_k,
                seq_len=max(seq_len, 512),
                theta=10000.0
            ).to(x.device)
    
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
    
        # Reshape for multi-head (but don't transpose yet)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
    
        # ✨ APPLY ROPE (replaces positional encoding)
        Q, K = apply_rotary_emb(Q, K, self.freqs_cis)
    
        # NOW transpose Q and K for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    
        # Apply mask if provided (for padding or causal masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
    
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
    
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
    
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
     
        # Final linear projection
        output = self.W_o(attn_output)
    
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    Two linear layers with GELU activation
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear + GELU
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Second linear
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block
    Multi-head attention + Feed-forward + Layer norms + Residuals
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-LN: Multi-head attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
    
        # Pre-LN: Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
    
        return x


class SolTransformer(nn.Module):
    """
    Decoder-only transformer language model
    Next-token prediction, pre-LayerNorm blocks, rotary position embeddings
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        final_norm=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm. Pre-LN blocks leave the last residual stream
        # un-normalised, so it is normalised once before the output projection.
        # final_norm=False rebuilds the older layout, for checkpoints saved
        # before this layer was added.
        self.final_norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Token IDs [batch_size, seq_len]
            mask: Optional attention mask
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Token embeddings
        x = self.token_embedding(x)
        
        # Scale embeddings (standard transformer scaling)
        x = x * math.sqrt(self.d_model)
        
        # RoPE handles positional encoding in attention layers
        x = self.dropout(x)
        
        # CREATE CAUSAL MASK IF NOT PROVIDED
        if mask is None:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq, seq]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final norm, then project to vocabulary
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate EOS token
                # (You'll need to define this based on your tokenizer)
        
        return generated


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*70)
    print("SOL TRANSFORMER ARCHITECTURE")
    print("="*70)
    print()
    
    # Create model
    vocab_size = 945  # From tokenizer
    model = SolTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model created:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dim: 256")
    print(f"  Attention heads: 4")
    print(f"  Transformer layers: 4")
    print(f"  Feed-forward dim: 1024")
    print(f"  Total parameters: {count_parameters(model):,}")
    print()
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Testing forward pass...")
    print(f"  Input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")
    
    print()
    print("="*70)
    print("✅ TRANSFORMER ARCHITECTURE READY")
    print("="*70)