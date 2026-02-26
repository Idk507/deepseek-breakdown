"""
Key-Value Cache Implementation
Complete code implementation with detailed examples

This file contains:
1. Basic KV Cache implementation
2. Multi-head attention with cache
3. Complete transformer layer with cache
4. Numerical examples
5. Performance comparisons
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import time


# ============================================================================
# PART 1: BASIC KV CACHE CLASS
# ============================================================================

class KVCache:
    """
    Simple KV cache for storing and managing past key-value pairs.
    
    This is the fundamental data structure for caching K and V matrices
    during autoregressive generation.
    """
    
    def __init__(
        self, 
        batch_size: int,
        max_seq_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        """
        Initialize KV cache with pre-allocated memory.
        
        Args:
            batch_size: Number of sequences in batch
            max_seq_length: Maximum sequence length to cache
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            dtype: Data type for cache (fp32, fp16, etc.)
            device: Device to store cache on ('cpu' or 'cuda')
        """
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate cache tensors
        self.key_cache = torch.zeros(
            batch_size, num_heads, max_seq_length, head_dim,
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            batch_size, num_heads, max_seq_length, head_dim,
            dtype=dtype, device=device
        )
        
        # Track current sequence length
        self.current_length = 0
    
    def update(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key and value tensors.
        
        Args:
            key: New keys [batch_size, num_heads, new_seq_len, head_dim]
            value: New values [batch_size, num_heads, new_seq_len, head_dim]
        
        Returns:
            Complete key and value tensors including cached history
        """
        batch_size, num_heads, new_seq_len, head_dim = key.shape
        
        # Store new keys and values in cache
        end_pos = self.current_length + new_seq_len
        self.key_cache[:batch_size, :, self.current_length:end_pos] = key
        self.value_cache[:batch_size, :, self.current_length:end_pos] = value
        
        # Update current length
        self.current_length = end_pos
        
        # Return full cache up to current position
        full_keys = self.key_cache[:batch_size, :, :self.current_length]
        full_values = self.value_cache[:batch_size, :, :self.current_length]
        
        return full_keys, full_values
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cached K and V without updating."""
        return (
            self.key_cache[:, :, :self.current_length],
            self.value_cache[:, :, :self.current_length]
        )
    
    def reset(self):
        """Reset cache to empty state."""
        self.current_length = 0
        self.key_cache.zero_()
        self.value_cache.zero_()
    
    def get_seq_length(self) -> int:
        """Get current sequence length."""
        return self.current_length
    
    def get_memory_usage(self) -> int:
        """Get cache size in bytes."""
        return (self.key_cache.element_size() * self.key_cache.nelement() +
                self.value_cache.element_size() * self.value_cache.nelement())


# ============================================================================
# PART 2: MULTI-HEAD ATTENTION WITH KV CACHE
# ============================================================================

class MultiHeadAttentionWithCache(nn.Module):
    """
    Multi-head attention with KV cache support.
    
    This demonstrates how to integrate KV cache into the attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV cache.
        
        Args:
            hidden_states: Input [batch_size, seq_len, d_model]
            attention_mask: Optional mask [batch_size, 1, seq_len, cached_seq_len]
            kv_cache: Optional cache from previous steps
            use_cache: Whether to use and return cache
        
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            kv_cache: Updated cache if use_cache=True, else None
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Step 1: Project to Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Step 2: Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)
        
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.transpose(1, 2)
        
        # Step 3: Update cache if provided
        if kv_cache is not None:
            keys, values = kv_cache.update(keys, values)
        
        # Step 4: Compute attention scores
        # Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Step 5: Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Step 6: Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Step 7: Apply attention to values
        attn_output = torch.matmul(attn_weights, values)
        
        # Step 8: Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Step 9: Final output projection
        output = self.out_proj(attn_output)
        
        return output, kv_cache if use_cache else None


# ============================================================================
# PART 3: TRANSFORMER LAYER WITH KV CACHE
# ============================================================================

class TransformerLayerWithCache(nn.Module):
    """
    Complete transformer layer with KV cache support.
    Includes attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttentionWithCache(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass through transformer layer.
        
        Args:
            hidden_states: Input [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            kv_cache: Optional cache from previous step
            use_cache: Whether to use and return cache
        
        Returns:
            output: Layer output [batch_size, seq_len, d_model]
            kv_cache: Updated cache if use_cache=True
        """
        # Self-attention with residual connection
        normed = self.norm1(hidden_states)
        attn_output, kv_cache = self.attention(
            normed,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attn_output
        
        # Feed-forward with residual connection
        normed = self.norm2(hidden_states)
        ff_output = self.feed_forward(normed)
        hidden_states = hidden_states + ff_output
        
        return hidden_states, kv_cache


# ============================================================================
# PART 4: UTILITY FUNCTIONS
# ============================================================================

def create_causal_mask(
    seq_len: int,
    cached_len: int = 0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.
    
    Args:
        seq_len: Current sequence length (new tokens)
        cached_len: Length of cached sequence
        device: Device to create mask on
    
    Returns:
        Causal mask [1, 1, seq_len, total_len]
        
    Example:
        seq_len=3, cached_len=2
        Mask allows attending to:
        Token 0 (new): positions 0,1 (cached)
        Token 1 (new): positions 0,1,2 (cached + first new)
        Token 2 (new): positions 0,1,2,3 (cached + first two new)
    """
    total_len = seq_len + cached_len
    
    # Create upper triangular matrix
    # 1s indicate positions that should be masked (can't attend to)
    mask = torch.triu(
        torch.ones(seq_len, total_len, device=device),
        diagonal=cached_len + 1
    )
    
    # Convert 1s to -inf for masked positions
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    # Add batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)


# ============================================================================
# PART 5: NUMERICAL EXAMPLE
# ============================================================================

def numerical_example():
    """
    Work through a concrete numerical example step by step.
    """
    print("=" * 80)
    print("NUMERICAL EXAMPLE: KV CACHE IN ACTION")
    print("=" * 80)
    
    # Simple configuration
    batch_size = 1
    d_model = 4
    num_heads = 2
    head_dim = 2
    max_seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # Create layer
    layer = TransformerLayerWithCache(d_model, num_heads, d_ff=16)
    layer.eval()
    
    # Create cache
    cache = KVCache(batch_size, max_seq_len, num_heads, head_dim)
    
    # Initial prompt: 3 tokens
    prompt_len = 3
    prompt = torch.randn(batch_size, prompt_len, d_model)
    
    print(f"\n" + "-" * 80)
    print("STEP 1: Process initial prompt (3 tokens)")
    print("-" * 80)
    print(f"Input shape: {prompt.shape}")
    
    with torch.no_grad():
        # Create causal mask for prompt
        mask = create_causal_mask(prompt_len, 0)
        
        # Process prompt and build cache
        output, cache = layer(prompt, attention_mask=mask, kv_cache=cache, use_cache=True)
        
        print(f"Output shape: {output.shape}")
        print(f"Cache length: {cache.get_seq_length()}")
        print(f"Cache memory: {cache.get_memory_usage()} bytes")
    
    # Generate 5 new tokens
    print(f"\n" + "-" * 80)
    print("STEP 2: Generate new tokens (one at a time)")
    print("-" * 80)
    
    with torch.no_grad():
        for i in range(5):
            # New token (in practice, this would be the previous output)
            new_token = torch.randn(batch_size, 1, d_model)
            
            # Create causal mask
            # New token can attend to all cached tokens plus itself
            cached_len = cache.get_seq_length()
            mask = create_causal_mask(1, cached_len)
            
            # Process new token with cache
            output, cache = layer(
                new_token,
                attention_mask=mask,
                kv_cache=cache,
                use_cache=True
            )
            
            print(f"  Token {i+1}: Processed 1 token, cache now has {cache.get_seq_length()} tokens")
    
    print(f"\n✓ Generated 5 tokens using cache!")
    print(f"✓ Final cache length: {cache.get_seq_length()}")
    print(f"✓ Only processed 1 token per step instead of reprocessing all!")


# ============================================================================
# PART 6: PERFORMANCE COMPARISON
# ============================================================================

def compare_with_without_cache():
    """
    Compare generation speed with and without KV cache.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: WITH vs WITHOUT CACHE")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    d_model = 256
    num_heads = 8
    head_dim = d_model // num_heads
    prompt_len = 20
    num_new_tokens = 30
    max_seq_len = 100
    
    print(f"\nConfiguration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  New tokens to generate: {num_new_tokens}")
    
    # Create layer
    layer = TransformerLayerWithCache(d_model, num_heads, d_ff=1024)
    layer.eval()
    
    # Initial prompt
    prompt = torch.randn(batch_size, prompt_len, d_model)
    
    # ========================================================================
    # METHOD 1: WITHOUT CACHE (INEFFICIENT)
    # ========================================================================
    print(f"\n" + "-" * 80)
    print("METHOD 1: WITHOUT CACHE")
    print("-" * 80)
    
    with torch.no_grad():
        start_time = time.time()
        
        current_seq = prompt.clone()
        
        for i in range(num_new_tokens):
            # Process entire sequence every time!
            seq_len = current_seq.shape[1]
            mask = create_causal_mask(seq_len, 0)
            
            output, _ = layer(current_seq, attention_mask=mask, use_cache=False)
            
            # Get last token output and append (simulating generation)
            next_token = torch.randn(batch_size, 1, d_model)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}: Processed {current_seq.shape[1]} tokens")
        
        time_without_cache = time.time() - start_time
    
    print(f"Time: {time_without_cache:.3f}s")
    
    # ========================================================================
    # METHOD 2: WITH CACHE (EFFICIENT)
    # ========================================================================
    print(f"\n" + "-" * 80)
    print("METHOD 2: WITH CACHE")
    print("-" * 80)
    
    with torch.no_grad():
        start_time = time.time()
        
        # Create cache
        cache = KVCache(batch_size, max_seq_len, num_heads, head_dim)
        
        # Process initial prompt
        mask = create_causal_mask(prompt_len, 0)
        output, cache = layer(prompt, attention_mask=mask, kv_cache=cache, use_cache=True)
        print(f"  Processed prompt: {prompt_len} tokens")
        
        # Generate new tokens
        for i in range(num_new_tokens):
            # Only process 1 token at a time!
            new_token = torch.randn(batch_size, 1, d_model)
            
            cached_len = cache.get_seq_length()
            mask = create_causal_mask(1, cached_len)
            
            output, cache = layer(
                new_token,
                attention_mask=mask,
                kv_cache=cache,
                use_cache=True
            )
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}: Processed 1 token, cache has {cache.get_seq_length()} tokens")
        
        time_with_cache = time.time() - start_time
    
    print(f"Time: {time_with_cache:.3f}s")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Without cache: {time_without_cache:.3f}s")
    print(f"With cache:    {time_with_cache:.3f}s")
    print(f"Speedup:       {time_without_cache / time_with_cache:.2f}x")
    print(f"\nCache memory:  {cache.get_memory_usage() / 1024:.2f} KB")
    print("=" * 80)


# ============================================================================
# PART 7: VISUALIZATION OF CACHE GROWTH
# ============================================================================

def visualize_cache_growth():
    """
    Show how cache grows during generation.
    """
    print("\n" + "=" * 80)
    print("VISUALIZATION: CACHE GROWTH DURING GENERATION")
    print("=" * 80)
    
    batch_size = 1
    d_model = 8
    num_heads = 2
    head_dim = 4
    max_seq_len = 20
    
    cache = KVCache(batch_size, max_seq_len, num_heads, head_dim)
    
    print("\nSequence: 'The cat sat on the mat'")
    print("-" * 80)
    
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    
    for i, token in enumerate(tokens):
        # Simulate processing token
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)
        
        cache.update(k, v)
        
        # Visual representation
        cache_len = cache.get_seq_length()
        bar = '█' * cache_len + '░' * (len(tokens) - cache_len)
        
        print(f"Token {i+1} '{token:5s}': [{bar}] Cache length: {cache_len}")
    
    print("-" * 80)
    print(f"Final cache stores K and V for all {cache.get_seq_length()} tokens")
    print(f"Total memory: {cache.get_memory_usage()} bytes")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("KEY-VALUE CACHE IMPLEMENTATION EXAMPLES")
    print("🚀" * 40)
    
    # Run all examples
    numerical_example()
    compare_with_without_cache()
    visualize_cache_growth()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print("""
KEY COMPONENTS:

1. KVCache class:
   - Stores K and V matrices from previous steps
   - Provides update() to add new K and V
   - Tracks current sequence length
   
2. MultiHeadAttentionWithCache:
   - Standard attention with cache parameter
   - Updates cache when use_cache=True
   - Returns both output and updated cache
   
3. Usage pattern:
   a) First step: Process prompt, build cache
   b) Subsequent steps: Process 1 token, update cache
   c) Each step uses cached K and V instead of recomputing

PERFORMANCE GAINS:
   - Computation: O(n²) → O(n) per token
   - Speedup: 2-20x depending on sequence length
   - Trade-off: Uses O(n×d) memory for cache

PRACTICAL TIPS:
   - Pre-allocate cache for max sequence length
   - Use fp16 to reduce memory by 50%
   - Consider Multi-Query Attention for 95% memory reduction
   - Monitor cache size in production systems
    """)
    print("=" * 80)
