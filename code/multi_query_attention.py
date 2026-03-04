"""
Multi-Query Attention Implementation
=====================================

This file contains:
1. From-scratch NumPy implementation for educational purposes
2. Production-ready PyTorch implementation
3. Comparison with Multi-Head Attention
4. Performance benchmarking and analysis
5. Example usage with KV caching demonstration

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from typing import Optional, Tuple
import time

# ============================================================================
# PART 1: FROM-SCRATCH NUMPY IMPLEMENTATION
# ============================================================================

class MultiQueryAttentionNumPy:
    """
    Multi-Query Attention implemented from scratch using NumPy.
    
    Key difference from Multi-Head Attention:
    - Multiple query heads (Q_1, Q_2, ..., Q_h)
    - Single shared key projection (K)
    - Single shared value projection (V)
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multi-Query Attention.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of query heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize weight matrices
        scale = 1.0 / math.sqrt(d_model)
        
        # Query projections: one per head (h separate projections)
        # Each W_q[i] projects d_model -> d_k
        self.W_q = [np.random.randn(d_model, self.d_k) * scale 
                    for _ in range(num_heads)]
        
        # SHARED Key projection: single projection for all heads
        # Projects d_model -> d_k
        self.W_k = np.random.randn(d_model, self.d_k) * scale
        
        # SHARED Value projection: single projection for all heads
        # Projects d_model -> d_k
        self.W_v = np.random.randn(d_model, self.d_k) * scale
        
        # Output projection: concatenated heads -> d_model
        self.W_o = np.random.randn(num_heads * self.d_k, d_model) * scale
        
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
            Q: Query matrix (..., seq_len_q, d_k)
            K: Key matrix (..., seq_len_k, d_k)
            V: Value matrix (..., seq_len_v, d_k)
            mask: Optional mask (..., seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output (..., seq_len_q, d_k)
            attention_weights: Attention weights (..., seq_len_q, seq_len_k)
        """
        # Compute Q·K^T and scale
        scores = np.matmul(Q, K.T) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Multiply by values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        Q_input: np.ndarray,
        K_input: np.ndarray,
        V_input: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, list]:
        """
        Forward pass of Multi-Query Attention.
        
        Args:
            Q_input: Query input (batch_size, seq_len_q, d_model)
            K_input: Key input (batch_size, seq_len_k, d_model)
            V_input: Value input (batch_size, seq_len_v, d_model)
            mask: Optional mask (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: List of attention weights for each head
        """
        batch_size, seq_len_q, _ = Q_input.shape
        seq_len_k = K_input.shape[1]
        
        # CRITICAL: Compute shared K and V once for all heads
        # This is the key difference from Multi-Head Attention
        K_shared = np.matmul(K_input, self.W_k)  # (batch, seq_len_k, d_k)
        V_shared = np.matmul(V_input, self.W_v)  # (batch, seq_len_k, d_k)
        
        # Compute attention for each query head
        head_outputs = []
        attention_weights_list = []
        
        for i in range(self.num_heads):
            # Project queries with head-specific projection
            Q_i = np.matmul(Q_input, self.W_q[i])  # (batch, seq_len_q, d_k)
            
            # Compute attention using shared K and V
            head_output_batch = []
            attn_weights_batch = []
            
            for b in range(batch_size):
                output, attn_weights = self.scaled_dot_product_attention(
                    Q_i[b], K_shared[b], V_shared[b],
                    mask[b] if mask is not None else None
                )
                head_output_batch.append(output)
                attn_weights_batch.append(attn_weights)
            
            head_outputs.append(np.stack(head_output_batch))
            attention_weights_list.append(np.stack(attn_weights_batch))
        
        # Concatenate all heads
        # Each head: (batch, seq_len_q, d_k)
        # Concatenate along last dimension: (batch, seq_len_q, h * d_k)
        concat_output = np.concatenate(head_outputs, axis=-1)
        
        # Final linear projection
        output = np.matmul(concat_output, self.W_o)
        
        return output, attention_weights_list


class KVCacheNumPy:
    """
    KV Cache for efficient autoregressive generation.
    Demonstrates the memory savings of MQA vs MHA.
    """
    
    def __init__(self, num_heads: int, d_k: int, max_seq_len: int = 2048):
        """
        Initialize KV cache.
        
        For MQA: num_heads = 1 (shared K, V)
        For MHA: num_heads = h (separate K, V per head)
        """
        self.num_heads = num_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Cache storage: (num_heads, max_seq_len, d_k)
        self.k_cache = np.zeros((num_heads, max_seq_len, d_k))
        self.v_cache = np.zeros((num_heads, max_seq_len, d_k))
        self.current_len = 0
    
    def update(self, k_new: np.ndarray, v_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add new key-value pairs to cache.
        
        Args:
            k_new: New keys (num_heads, new_tokens, d_k)
            v_new: New values (num_heads, new_tokens, d_k)
            
        Returns:
            Full cached keys and values up to current position
        """
        new_tokens = k_new.shape[1]
        
        # Store new K, V in cache
        self.k_cache[:, self.current_len:self.current_len + new_tokens, :] = k_new
        self.v_cache[:, self.current_len:self.current_len + new_tokens, :] = v_new
        
        self.current_len += new_tokens
        
        # Return full cache up to current length
        return (self.k_cache[:, :self.current_len, :],
                self.v_cache[:, :self.current_len, :])
    
    def get_memory_size(self) -> int:
        """Return memory size in bytes (assuming float32)."""
        return (self.k_cache.nbytes + self.v_cache.nbytes)


# ============================================================================
# PART 2: PYTORCH IMPLEMENTATION (PRODUCTION-READY)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MultiQueryAttention(nn.Module):
        """
        Multi-Query Attention module (PyTorch implementation).
        
        Production-ready implementation with:
        - Proper gradient support
        - GPU acceleration
        - Efficient KV caching
        - Optimized memory usage
        """
        
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            """
            Initialize Multi-Query Attention.
            
            Args:
                d_model: Dimension of the model
                num_heads: Number of query heads
                dropout: Dropout rate
            """
            super(MultiQueryAttention, self).__init__()
            
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Query projections: separate for each head
            self.W_q = nn.Linear(d_model, d_model)  # Will split into heads
            
            # SHARED Key projection: single projection
            self.W_k = nn.Linear(d_model, self.d_k)
            
            # SHARED Value projection: single projection
            self.W_v = nn.Linear(d_model, self.d_k)
            
            # Output projection
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)
            
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            """
            Forward pass.
            
            Args:
                query: (batch_size, seq_len_q, d_model)
                key: (batch_size, seq_len_k, d_model)
                value: (batch_size, seq_len_v, d_model)
                mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
                use_cache: Whether to use/return KV cache
                past_kv: Past cached (K, V) if available
                
            Returns:
                output: (batch_size, seq_len_q, d_model)
                attention_weights: (batch_size, num_heads, seq_len_q, total_seq_len_k)
                present_kv: Current (K, V) cache if use_cache=True
            """
            batch_size, seq_len_q, _ = query.shape
            
            # Project queries (will be split into heads)
            Q = self.W_q(query)  # (batch, seq_len_q, d_model)
            
            # Reshape Q into heads: (batch, num_heads, seq_len_q, d_k)
            Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k)
            Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len_q, d_k)
            
            # CRITICAL: Shared K and V projections
            K = self.W_k(key)    # (batch, seq_len_k, d_k)
            V = self.W_v(value)  # (batch, seq_len_v, d_k)
            
            # Handle KV cache for autoregressive generation
            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    # Concatenate past and current K, V
                    K = torch.cat([past_k, K], dim=1)
                    V = torch.cat([past_v, V], dim=1)
                present_kv = (K, V)
            else:
                present_kv = None
            
            # Expand K and V to match query heads
            # From: (batch, seq_len_k, d_k)
            # To: (batch, num_heads, seq_len_k, d_k)
            K = K.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            V = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_weights, V)
            
            # Concatenate heads
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len_q, self.d_model)
            
            # Output projection
            output = self.W_o(attention_output)
            
            return output, attention_weights, present_kv
    
    
    class MultiHeadAttention(nn.Module):
        """
        Standard Multi-Head Attention for comparison.
        """
        
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            super(MultiHeadAttention, self).__init__()
            
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Separate projections for each head
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)
            
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            batch_size = query.size(0)
            
            # Project Q, K, V
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)
            
            # Split into heads
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # Handle KV cache
            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2)
                    V = torch.cat([past_v, V], dim=2)
                present_kv = (K, V)
            else:
                present_kv = None
            
            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attention_output = torch.matmul(attention_weights, V)
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, -1, self.d_model)
            
            output = self.W_o(attention_output)
            
            return output, attention_weights, present_kv
    
    PYTORCH_AVAILABLE = True
    
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Only NumPy implementation will work.")


# ============================================================================
# PART 3: COMPARISON AND BENCHMARKING
# ============================================================================

def compare_kv_cache_memory():
    """Compare KV cache memory usage between MHA and MQA."""
    print("=" * 70)
    print("KV Cache Memory Comparison: MHA vs MQA")
    print("=" * 70)
    
    # Model configuration
    num_heads = 32
    d_k = 128
    max_seq_len = 2048
    num_layers = 40
    
    # MHA: separate K, V for each head
    mha_cache = KVCacheNumPy(num_heads=num_heads, d_k=d_k, max_seq_len=max_seq_len)
    mha_memory = mha_cache.get_memory_size()
    mha_total = mha_memory * num_layers
    
    # MQA: shared K, V (single head)
    mqa_cache = KVCacheNumPy(num_heads=1, d_k=d_k, max_seq_len=max_seq_len)
    mqa_memory = mqa_cache.get_memory_size()
    mqa_total = mqa_memory * num_layers
    
    print(f"\nConfiguration:")
    print(f"  Heads: {num_heads}")
    print(f"  Head dimension: {d_k}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Number of layers: {num_layers}")
    
    print(f"\nPer Layer:")
    print(f"  MHA cache: {mha_memory / 1024 / 1024:.2f} MB")
    print(f"  MQA cache: {mqa_memory / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {mha_memory / mqa_memory:.1f}x")
    
    print(f"\nTotal Model:")
    print(f"  MHA cache: {mha_total / 1024 / 1024 / 1024:.2f} GB")
    print(f"  MQA cache: {mqa_total / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Reduction: {mha_total / mqa_total:.1f}x")
    print(f"  Memory saved: {(mha_total - mqa_total) / 1024 / 1024 / 1024:.2f} GB")
    print()


def compare_parameters():
    """Compare parameter counts between MHA and MQA."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping parameter comparison.")
        return
    
    print("=" * 70)
    print("Parameter Count Comparison: MHA vs MQA")
    print("=" * 70)
    
    d_model = 512
    num_heads = 8
    
    # Create models
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    
    # Count parameters
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k per head: {d_model // num_heads}")
    
    print(f"\nParameter Breakdown:")
    print(f"  MHA W_q: {d_model * d_model:,}")
    print(f"  MHA W_k: {d_model * d_model:,}")
    print(f"  MHA W_v: {d_model * d_model:,}")
    print(f"  MHA W_o: {d_model * d_model:,}")
    print(f"  MHA Total: {mha_params:,}")
    
    print(f"\n  MQA W_q: {d_model * d_model:,}")
    print(f"  MQA W_k: {d_model * (d_model // num_heads):,}")
    print(f"  MQA W_v: {d_model * (d_model // num_heads):,}")
    print(f"  MQA W_o: {d_model * d_model:,}")
    print(f"  MQA Total: {mqa_params:,}")
    
    print(f"\nReduction:")
    print(f"  Parameters saved: {mha_params - mqa_params:,}")
    print(f"  Reduction ratio: {mha_params / mqa_params:.2f}x")
    print(f"  Percentage reduction: {(1 - mqa_params / mha_params) * 100:.1f}%")
    print()


def benchmark_inference_speed():
    """Benchmark inference speed of MHA vs MQA."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping speed benchmark.")
        return
    
    print("=" * 70)
    print("Inference Speed Benchmark: MHA vs MQA")
    print("=" * 70)
    
    batch_size = 8
    seq_len = 512
    d_model = 512
    num_heads = 8
    num_iterations = 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Heads: {num_heads}")
    print(f"Iterations: {num_iterations}")
    
    # Create models
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0).to(device)
    mqa = MultiQueryAttention(d_model, num_heads, dropout=0.0).to(device)
    mha.eval()
    mqa.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = mha(x, x, x)
            _ = mqa(x, x, x)
    
    # Benchmark MHA
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mha(x, x, x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mha_time = time.time() - start
    
    # Benchmark MQA
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = mqa(x, x, x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mqa_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  MHA time: {mha_time:.3f}s ({mha_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"  MQA time: {mqa_time:.3f}s ({mqa_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"  Speedup: {mha_time/mqa_time:.2f}x")
    print()


def demonstrate_autoregressive_generation():
    """Demonstrate autoregressive generation with KV caching."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping generation demo.")
        return
    
    print("=" * 70)
    print("Autoregressive Generation with KV Caching")
    print("=" * 70)
    
    d_model = 256
    num_heads = 8
    max_gen_len = 10
    
    mqa = MultiQueryAttention(d_model, num_heads, dropout=0.0)
    mqa.eval()
    
    print(f"\nGenerating {max_gen_len} tokens autoregressively...")
    print("(Simulated - tracking cache growth)\n")
    
    # Start with initial prompt
    prompt = torch.randn(1, 5, d_model)  # batch=1, seq_len=5
    past_kv = None
    
    print("Step 0: Process initial prompt")
    print(f"  Input shape: {prompt.shape}")
    
    with torch.no_grad():
        output, _, past_kv = mqa(prompt, prompt, prompt, use_cache=True)
    
    k_cache, v_cache = past_kv
    print(f"  KV cache initialized: K{k_cache.shape}, V{v_cache.shape}")
    
    # Generate tokens one by one
    for step in range(1, max_gen_len + 1):
        # New token (in real scenario, this would be from previous output)
        new_token = torch.randn(1, 1, d_model)
        
        print(f"\nStep {step}: Generate token {step}")
        print(f"  New token shape: {new_token.shape}")
        
        # Forward pass with cache
        with torch.no_grad():
            output, _, past_kv = mqa(new_token, new_token, new_token,
                                     use_cache=True, past_kv=past_kv)
        
        k_cache, v_cache = past_kv
        print(f"  Updated KV cache: K{k_cache.shape}, V{v_cache.shape}")
        print(f"  Total cached tokens: {k_cache.shape[1]}")
    
    print("\n✓ Generation complete!")
    print(f"  Final cache size: {k_cache.shape[1]} tokens")
    print(f"  Memory per token (MQA): {k_cache[0].numel() * 4 / 1024:.2f} KB")
    print(f"  Memory per token (MHA): {k_cache[0].numel() * num_heads * 4 / 1024:.2f} KB")
    print(f"  Memory saved per token: {k_cache[0].numel() * (num_heads - 1) * 4 / 1024:.2f} KB")
    print()


def visualize_attention_difference():
    """Visualize how MHA and MQA produce different attention patterns."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("=" * 70)
    print("Attention Pattern Visualization: MHA vs MQA")
    print("=" * 70)
    
    seq_len = 5
    d_model = 16
    num_heads = 4
    
    # Create input with clear pattern
    x = torch.zeros(1, seq_len, d_model)
    for i in range(seq_len):
        x[0, i, i % d_model] = 1.0
    
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    mqa = MultiQueryAttention(d_model, num_heads, dropout=0.0)
    mha.eval()
    mqa.eval()
    
    with torch.no_grad():
        _, mha_attn, _ = mha(x, x, x)
        _, mqa_attn, _ = mqa(x, x, x)
    
    print("\nMHA Attention Weights (Head 0):")
    print(mha_attn[0, 0].numpy())
    print("\nMQA Attention Weights (Head 0):")
    print(mqa_attn[0, 0].numpy())
    
    print("\nKey Observation:")
    print("- MHA: Each head can attend based on different K/V representations")
    print("- MQA: All heads share the same K/V, but different Q produces different attention")
    print()


# ============================================================================
# PART 4: EXAMPLE USAGE AND TESTING
# ============================================================================

def test_numpy_implementation():
    """Test the NumPy implementation."""
    print("=" * 70)
    print("Testing NumPy Multi-Query Attention")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, d_model)
    
    mqa = MultiQueryAttentionNumPy(d_model=d_model, num_heads=num_heads)
    
    output, attention_weights = mqa.forward(X, X, X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    print(f"Each attention weight shape: {attention_weights[0].shape}")
    
    print(f"\nAttention weights for head 1, sample 1:")
    print(attention_weights[0][0])
    print(f"\nRow sums (should be ~1.0): {np.sum(attention_weights[0][0], axis=-1)}")
    print()


def test_pytorch_implementation():
    """Test the PyTorch implementation."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("=" * 70)
    print("Testing PyTorch Multi-Query Attention")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 4
    d_model = 512
    num_heads = 8
    
    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_len, d_model)
    
    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads)
    mqa.eval()
    
    with torch.no_grad():
        output, attention_weights, _ = mqa(X, X, X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights for head 1, sample 1:")
    print(attention_weights[0, 0])
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTI-QUERY ATTENTION: IMPLEMENTATION AND ANALYSIS")
    print("=" * 70 + "\n")
    
    # Run all tests and comparisons
    test_numpy_implementation()
    test_pytorch_implementation()
    
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70 + "\n")
    
    compare_parameters()
    compare_kv_cache_memory()
    benchmark_inference_speed()
    demonstrate_autoregressive_generation()
    visualize_attention_difference()
    
    print("=" * 70)
    print("All tests and benchmarks completed!")
    print("=" * 70)
