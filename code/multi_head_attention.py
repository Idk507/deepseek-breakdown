"""
Multi-Head Attention Implementation
====================================

This file contains:
1. From-scratch NumPy implementation for educational purposes
2. Production-ready PyTorch implementation
3. Example usage and testing code

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from typing import Optional, Tuple

# ============================================================================
# PART 1: FROM-SCRATCH NUMPY IMPLEMENTATION
# ============================================================================

class MultiHeadAttentionNumPy:
    """
    Multi-Head Attention implemented from scratch using NumPy.
    
    This is for educational purposes to understand the mechanics.
    For production, use the PyTorch version below.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize weight matrices (Xavier/Glorot initialization)
        scale = 1.0 / math.sqrt(d_model)
        
        # Query, Key, Value projection matrices for all heads
        # Shape: (d_model, d_model) - we'll split by heads during forward pass
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        
        # Output projection matrix
        self.W_o = np.random.randn(d_model, d_model) * scale
        
    def scaled_dot_product_attention(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len, d_k)
            K: Key matrix (batch_size, num_heads, seq_len, d_k)
            V: Value matrix (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: Attention output (batch_size, num_heads, seq_len, d_k)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        # Step 1: Compute Q·K^T
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Step 2: Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.d_k)
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Set masked positions to large negative value (will become ~0 after softmax)
            scores = np.where(mask == 0, -1e9, scores)
        
        # Step 4: Apply softmax
        # Subtract max for numerical stability
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Step 5: Multiply by V
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Reshaped tensor (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back into single dimension.
        
        Args:
            x: Input tensor (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            Combined tensor (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Reshape to (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            Q: Query (batch_size, seq_len_q, d_model)
            K: Key (batch_size, seq_len_k, d_model)
            V: Value (batch_size, seq_len_v, d_model)
            mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q.shape[0]
        
        # Step 1: Linear projections
        # Q, K, V shape: (batch_size, seq_len, d_model)
        Q_proj = np.matmul(Q, self.W_q)
        K_proj = np.matmul(K, self.W_k)
        V_proj = np.matmul(V, self.W_v)
        
        # Step 2: Split into multiple heads
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # Step 3: Apply scaled dot-product attention for each head
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask
        )
        
        # Step 4: Concatenate heads
        # Shape: (batch_size, seq_len_q, d_model)
        concat_output = self.combine_heads(attention_output)
        
        # Step 5: Final linear projection
        output = np.matmul(concat_output, self.W_o)
        
        return output, attention_weights


# ============================================================================
# PART 2: PYTORCH IMPLEMENTATION (PRODUCTION-READY)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Attention module (PyTorch implementation).
        
        This is a production-ready implementation with proper gradient support,
        GPU acceleration, and optimizations.
        """
        
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            """
            Initialize Multi-Head Attention.
            
            Args:
                d_model: Dimension of the model
                num_heads: Number of attention heads
                dropout: Dropout rate (default: 0.1)
            """
            super(MultiHeadAttention, self).__init__()
            
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Linear layers for Q, K, V projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            
            # Output projection
            self.W_o = nn.Linear(d_model, d_model)
            
            # Dropout
            self.dropout = nn.Dropout(dropout)
            
            # Scaling factor
            self.scale = math.sqrt(self.d_k)
            
        def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                query: Query tensor (batch_size, seq_len_q, d_model)
                key: Key tensor (batch_size, seq_len_k, d_model)
                value: Value tensor (batch_size, seq_len_v, d_model)
                mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
                
            Returns:
                output: (batch_size, seq_len_q, d_model)
                attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
            """
            batch_size = query.size(0)
            
            # Linear projections
            Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
            K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
            V = self.W_v(value)  # (batch_size, seq_len_v, d_model)
            
            # Split into multiple heads
            # Reshape: (batch_size, seq_len, num_heads, d_k)
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
            K = K.view(batch_size, -1, self.num_heads, self.d_k)
            V = V.view(batch_size, -1, self.num_heads, self.d_k)
            
            # Transpose: (batch_size, num_heads, seq_len, d_k)
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            # Scaled dot-product attention
            # Scores: (batch_size, num_heads, seq_len_q, seq_len_k)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            # (batch_size, num_heads, seq_len_q, d_k)
            attention_output = torch.matmul(attention_weights, V)
            
            # Concatenate heads
            # Transpose: (batch_size, seq_len_q, num_heads, d_k)
            attention_output = attention_output.transpose(1, 2).contiguous()
            
            # Reshape: (batch_size, seq_len_q, d_model)
            attention_output = attention_output.view(batch_size, -1, self.d_model)
            
            # Final linear projection
            output = self.W_o(attention_output)
            
            return output, attention_weights
    
    PYTORCH_AVAILABLE = True
    
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Only NumPy implementation will work.")


# ============================================================================
# PART 3: EXAMPLE USAGE AND TESTING
# ============================================================================

def test_numpy_implementation():
    """Test the NumPy implementation with a small example."""
    print("=" * 70)
    print("Testing NumPy Implementation")
    print("=" * 70)
    
    # Parameters
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    
    # Create random input
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, d_model)
    
    # Initialize attention
    mha = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads)
    
    # Forward pass (self-attention)
    output, attention_weights = mha.forward(X, X, X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights for first sample, first head:")
    print(attention_weights[0, 0])
    print(f"\nSum of attention weights (should be ~1.0 for each row):")
    print(np.sum(attention_weights[0, 0], axis=-1))
    print()


def test_pytorch_implementation():
    """Test the PyTorch implementation with a small example."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping PyTorch test.")
        return
    
    print("=" * 70)
    print("Testing PyTorch Implementation")
    print("=" * 70)
    
    # Parameters
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    
    # Create random input
    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()  # Set to evaluation mode
    
    # Forward pass (self-attention)
    with torch.no_grad():
        output, attention_weights = mha(X, X, X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights for first sample, first head:")
    print(attention_weights[0, 0])
    print(f"\nSum of attention weights (should be ~1.0 for each row):")
    print(torch.sum(attention_weights[0, 0], dim=-1))
    print()


def demo_with_mask():
    """Demonstrate attention with masking (causal attention for autoregressive models)."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping mask demo.")
        return
    
    print("=" * 70)
    print("Demonstrating Causal Masking (for autoregressive models)")
    print("=" * 70)
    
    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 4
    
    # Create input
    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Create causal mask (lower triangular)
    # Position i can only attend to positions <= i
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    print(f"Causal mask (1 = attend, 0 = mask):")
    print(mask[0, 0].int())
    print()
    
    # Initialize attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()
    
    # Forward pass with mask
    with torch.no_grad():
        output, attention_weights = mha(X, X, X, mask=mask)
    
    print(f"Attention weights for first head (with causal masking):")
    print(attention_weights[0, 0])
    print("\nNotice: Upper triangle is ~0 (masked positions)")
    print()


def visualize_attention_patterns():
    """Create a simple visualization of attention patterns."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("=" * 70)
    print("Attention Pattern Example")
    print("=" * 70)
    
    # Simple example with interpretable input
    seq_len = 6
    d_model = 8
    num_heads = 2
    
    # Create a simple pattern: alternate between two types
    X = torch.zeros(1, seq_len, d_model)
    X[0, ::2, :4] = 1.0   # Even positions: first half is 1
    X[0, 1::2, 4:] = 1.0  # Odd positions: second half is 1
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.eval()
    
    with torch.no_grad():
        output, attention_weights = mha(X, X, X)
    
    print("Input pattern (1s in different positions):")
    print("Positions 0,2,4 have 1s in first half")
    print("Positions 1,3,5 have 1s in second half")
    print()
    
    for head in range(num_heads):
        print(f"\nHead {head + 1} attention weights:")
        print(attention_weights[0, head].numpy())
        print("(Each row shows where that position attends)")


def compare_numpy_vs_pytorch():
    """Compare outputs between NumPy and PyTorch implementations."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Cannot compare implementations.")
        return
    
    print("=" * 70)
    print("Comparing NumPy vs PyTorch Implementations")
    print("=" * 70)
    
    # Small example for comparison
    batch_size = 1
    seq_len = 3
    d_model = 16
    num_heads = 4
    
    # Create same input for both
    np.random.seed(42)
    X_np = np.random.randn(batch_size, seq_len, d_model)
    X_torch = torch.from_numpy(X_np).float()
    
    # NumPy version
    mha_np = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads)
    output_np, attn_np = mha_np.forward(X_np, X_np, X_np)
    
    # PyTorch version (copy weights from NumPy for fair comparison)
    mha_torch = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    with torch.no_grad():
        mha_torch.W_q.weight.copy_(torch.from_numpy(mha_np.W_q.T).float())
        mha_torch.W_k.weight.copy_(torch.from_numpy(mha_np.W_k.T).float())
        mha_torch.W_v.weight.copy_(torch.from_numpy(mha_np.W_v.T).float())
        mha_torch.W_o.weight.copy_(torch.from_numpy(mha_np.W_o.T).float())
        mha_torch.W_q.bias.zero_()
        mha_torch.W_k.bias.zero_()
        mha_torch.W_v.bias.zero_()
        mha_torch.W_o.bias.zero_()
    
    mha_torch.eval()
    with torch.no_grad():
        output_torch, attn_torch = mha_torch(X_torch, X_torch, X_torch)
    
    # Compare
    output_diff = np.abs(output_np - output_torch.numpy()).max()
    attn_diff = np.abs(attn_np - attn_torch.numpy()).max()
    
    print(f"Maximum difference in outputs: {output_diff:.2e}")
    print(f"Maximum difference in attention weights: {attn_diff:.2e}")
    print("\n(Small differences expected due to floating point precision)")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTI-HEAD ATTENTION IMPLEMENTATION AND TESTING")
    print("=" * 70 + "\n")
    
    # Run all tests
    test_numpy_implementation()
    test_pytorch_implementation()
    demo_with_mask()
    visualize_attention_patterns()
    compare_numpy_vs_pytorch()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)