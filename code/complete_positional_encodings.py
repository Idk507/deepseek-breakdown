"""
Complete Positional Encoding Implementation
============================================

This file contains implementations of all major positional encoding techniques:
1. Sinusoidal Positional Encoding
2. Learned Positional Encoding  
3. Rotary Positional Embedding (RoPE)
4. ALiBi (Attention with Linear Biases)
5. Integer Positional Encoding
6. Binary Positional Encoding
7. Relative Positional Encoding
8. Advanced variants and comparisons

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from typing import Tuple, Optional, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 1. SINUSOIDAL POSITIONAL ENCODING
# ============================================================================

class SinusoidalPositionalEncoding:
    """
    Original positional encoding from 'Attention Is All You Need'.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency calculation (default 10000)
        """
        assert d_model % 2 == 0, "d_model must be even"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute positional encodings
        self.pe = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding matrix."""
        pe = np.zeros((self.max_len, self.d_model))
        
        # Position indices
        position = np.arange(0, self.max_len)[:, np.newaxis]
        
        # Dimension indices for frequencies
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(self.base) / self.d_model))
        
        # Apply sine to even dimensions
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len: int) -> np.ndarray:
        """Get positional encodings for sequence."""
        return self.pe[:seq_len, :]
    
    def analyze_frequencies(self):
        """Analyze and print frequency information."""
        print("Sinusoidal Encoding - Frequency Analysis")
        print("=" * 80)
        
        for i in range(0, min(self.d_model, 16), 2):
            freq = 1.0 / (self.base ** (i / self.d_model))
            wavelength = 2 * np.pi / freq
            
            print(f"Dimension {i:2d}/{i+1:2d}: frequency = {freq:.6f}, "
                  f"wavelength = {wavelength:.2f}")
        
        print()


# ============================================================================
# 2. LEARNED POSITIONAL ENCODING
# ============================================================================

class LearnedPositionalEncoding:
    """
    Learned positional embeddings (like BERT, GPT-2).
    
    Embeddings are learned parameters, updated during training.
    """
    
    def __init__(self, max_len: int, d_model: int, seed: int = 42):
        """
        Initialize learned positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
            seed: Random seed for initialization
        """
        np.random.seed(seed)
        
        self.max_len = max_len
        self.d_model = d_model
        
        # Initialize position embeddings with small random values
        scale = 1.0 / np.sqrt(d_model)
        self.embeddings = np.random.randn(max_len, d_model) * scale
        
        # Store gradients for demonstration
        self.gradients = np.zeros_like(self.embeddings)
    
    def forward(self, seq_len: int) -> np.ndarray:
        """Get position embeddings for sequence."""
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max {self.max_len}"
        return self.embeddings[:seq_len, :]
    
    def update(self, learning_rate: float = 0.01):
        """Update embeddings using stored gradients."""
        self.embeddings -= learning_rate * self.gradients
        self.gradients.fill(0)
    
    def accumulate_gradient(self, pos: int, grad: np.ndarray):
        """Accumulate gradient for a position."""
        self.gradients[pos] += grad


# ============================================================================
# 3. ROTARY POSITIONAL EMBEDDING (RoPE)
# ============================================================================

class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes position by rotating query and key vectors.
    Used in LLaMA, PaLM, GPT-NeoX.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency calculation
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute inverse frequencies
        self.inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        
        # Precompute sin and cos for all positions
        self.sin_cached, self.cos_cached = self._create_cache()
    
    def _create_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute sin and cos values for all positions."""
        positions = np.arange(self.max_len)
        
        # Compute m * theta for each position and frequency
        freqs = positions[:, np.newaxis] * self.inv_freq[np.newaxis, :]
        
        # Compute sin and cos
        sin_cached = np.sin(freqs)  # (max_len, d_model/2)
        cos_cached = np.cos(freqs)  # (max_len, d_model/2)
        
        return sin_cached, cos_cached
    
    def apply_rotary_embedding(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor (seq_len, d_model) or (batch, seq_len, d_model)
            start_pos: Starting position (for cached generation)
            
        Returns:
            Rotated tensor with same shape as input
        """
        if x.ndim == 2:
            seq_len, d_model = x.shape
            batch_size = 1
            x = x[np.newaxis, :, :]
        else:
            batch_size, seq_len, d_model = x.shape
        
        assert d_model == self.d_model, f"Input dim {d_model} != RoPE dim {self.d_model}"
        
        # Get sin and cos for this sequence
        sin = self.sin_cached[start_pos:start_pos + seq_len, :]  # (seq_len, d/2)
        cos = self.cos_cached[start_pos:start_pos + seq_len, :]  # (seq_len, d/2)
        
        # Reshape input into pairs
        x_pairs = x.reshape(batch_size, seq_len, -1, 2)  # (batch, seq_len, d/2, 2)
        
        # Extract x0 and x1 from each pair
        x0 = x_pairs[..., 0]  # (batch, seq_len, d/2)
        x1 = x_pairs[..., 1]  # (batch, seq_len, d/2)
        
        # Apply rotation
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        
        # Recombine pairs
        rotated = np.stack([x0_new, x1_new], axis=-1)
        rotated = rotated.reshape(batch_size, seq_len, d_model)
        
        if batch_size == 1:
            return rotated[0]
        return rotated
    
    def demonstrate_relative_property(self):
        """Demonstrate that RoPE encodes relative positions."""
        print("RoPE - Relative Position Property")
        print("=" * 80)
        
        # Create sample query and key
        q = np.random.randn(self.d_model)
        k = np.random.randn(self.d_model)
        
        # Test at different absolute positions
        for m in [0, 10, 100]:
            for n in [0, 10, 100]:
                # Rotate query at position m, key at position n
                q_m = self.apply_rotary_embedding(q[np.newaxis, :], start_pos=m)[0]
                k_n = self.apply_rotary_embedding(k[np.newaxis, :], start_pos=n)[0]
                
                # Compute attention score
                score = np.dot(q_m, k_n)
                
                print(f"Positions (m={m:3d}, n={n:3d}), "
                      f"relative={n-m:4d}, score={score:8.4f}")
        
        print("\nNote: Scores depend only on (n-m), not absolute positions!")
        print()


# ============================================================================
# 4. ALiBi (ATTENTION WITH LINEAR BIASES)
# ============================================================================

class ALiBiPositionalBias:
    """
    ALiBi (Attention with Linear Biases).
    
    Adds position-dependent bias to attention scores.
    Used in BLOOM, MPT, StarCoder.
    """
    
    def __init__(self, num_heads: int):
        """
        Initialize ALiBi.
        
        Args:
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads
        
        # Compute head-specific slopes
        self.slopes = self._get_slopes()
    
    def _get_slopes(self) -> np.ndarray:
        """
        Compute head-specific slopes.
        
        Returns:
            slopes: (num_heads,)
        """
        # m_h = 2^(-8h/H)
        slopes = []
        for h in range(1, self.num_heads + 1):
            slope = 2 ** (-8 * h / self.num_heads)
            slopes.append(slope)
        return np.array(slopes)
    
    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Get ALiBi bias matrix for sequence.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        # Create distance matrix
        positions = np.arange(seq_len)
        distance = positions[np.newaxis, :] - positions[:, np.newaxis]
        distance = np.abs(distance)  # (seq_len, seq_len)
        
        # Apply head-specific slopes
        # (num_heads, 1, 1) * (1, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
        bias = -self.slopes[:, np.newaxis, np.newaxis] * distance[np.newaxis, :, :]
        
        return bias
    
    def visualize_attention_spans(self, seq_len: int = 20):
        """Visualize how different heads have different attention spans."""
        print("ALiBi - Head-Specific Attention Spans")
        print("=" * 80)
        
        bias = self.get_bias(seq_len)
        
        # Focus on query at position seq_len//2
        query_pos = seq_len // 2
        
        print(f"Biases for query at position {query_pos}:\n")
        print("Key Pos |", end="")
        for h in range(min(4, self.num_heads)):
            print(f"  Head {h+1}  |", end="")
        print()
        print("-" * 80)
        
        for key_pos in range(seq_len):
            print(f"   {key_pos:2d}   |", end="")
            for h in range(min(4, self.num_heads)):
                print(f" {bias[h, query_pos, key_pos]:7.3f} |", end="")
            print()
        
        print("\nObservation: Head 1 has largest penalty (smallest attention span)")
        print("             Head 4 has smallest penalty (largest attention span)")
        print()


# ============================================================================
# 5. INTEGER POSITIONAL ENCODING
# ============================================================================

class IntegerPositionalEncoding:
    """Simple integer-based positional encoding."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def forward(self, seq_len: int, normalize: bool = True) -> np.ndarray:
        """Generate integer positional encodings."""
        positions = np.arange(seq_len)[:, np.newaxis]
        
        if normalize and seq_len > 1:
            positions = positions / (seq_len - 1)
        
        pe = np.repeat(positions, self.d_model, axis=1)
        return pe


# ============================================================================
# 6. BINARY POSITIONAL ENCODING
# ============================================================================

class BinaryPositionalEncoding:
    """Binary positional encoding."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def forward(self, seq_len: int) -> np.ndarray:
        """Generate binary positional encodings."""
        pe = np.zeros((seq_len, self.d_model))
        
        for pos in range(seq_len):
            for i in range(self.d_model):
                pe[pos, i] = (pos >> i) & 1
        
        return pe


class GrayCodeEncoding:
    """Gray code positional encoding."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def _binary_to_gray(self, n: int) -> int:
        """Convert binary to Gray code."""
        return n ^ (n >> 1)
    
    def forward(self, seq_len: int) -> np.ndarray:
        """Generate Gray code positional encodings."""
        pe = np.zeros((seq_len, self.d_model))
        
        for pos in range(seq_len):
            gray = self._binary_to_gray(pos)
            for i in range(self.d_model):
                pe[pos, i] = (gray >> i) & 1
        
        return pe


# ============================================================================
# 7. RELATIVE POSITIONAL ENCODING
# ============================================================================

class RelativePositionalEncoding:
    """
    Relative positional encoding (T5-style).
    
    Learns embeddings for relative distances.
    """
    
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, 
                 num_heads: int = 8, seed: int = 42):
        """
        Initialize relative positional encoding.
        
        Args:
            num_buckets: Number of relative position buckets
            max_distance: Maximum distance to represent
            num_heads: Number of attention heads
            seed: Random seed
        """
        np.random.seed(seed)
        
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        
        # Learned relative position biases
        self.relative_attention_bias = np.random.randn(num_heads, num_buckets) * 0.01
    
    def _relative_position_bucket(self, relative_position: int) -> int:
        """
        Map relative position to bucket index.
        
        Uses logarithmic bucketing for larger distances.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        ret = 0
        n = -relative_position
        
        # Half the buckets for exact positions (small distances)
        # Half for logarithmically bigger distances
        num_buckets_exact = num_buckets // 2
        max_exact = num_buckets_exact // 2
        
        is_small = n < max_exact
        
        if is_small:
            ret = n + num_buckets_exact
        else:
            # Logarithmic bucketing for larger distances
            val_if_large = max_exact + (
                np.log(n / max_exact) / np.log(max_distance / max_exact) * 
                (num_buckets_exact - 1)
            )
            val_if_large = min(val_if_large, num_buckets - 1)
            ret = val_if_large + num_buckets_exact
        
        return int(ret)
    
    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Get relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        # Compute relative positions
        positions = np.arange(seq_len)
        relative_positions = positions[np.newaxis, :] - positions[:, np.newaxis]
        
        # Map to buckets
        buckets = np.zeros_like(relative_positions)
        for i in range(seq_len):
            for j in range(seq_len):
                buckets[i, j] = self._relative_position_bucket(relative_positions[i, j])
        
        # Look up biases
        bias = np.zeros((self.num_heads, seq_len, seq_len))
        for h in range(self.num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    bias[h, i, j] = self.relative_attention_bias[h, buckets[i, j]]
        
        return bias


# ============================================================================
# 8. COMPARISON AND VISUALIZATION
# ============================================================================

def compare_all_encodings(seq_len: int = 32, d_model: int = 8):
    """Compare all positional encoding methods."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE POSITIONAL ENCODING COMPARISON")
    print("=" * 80)
    print(f"Sequence length: {seq_len}, Embedding dimension: {d_model}\n")
    
    # Create all encoders
    sinusoidal = SinusoidalPositionalEncoding(d_model)
    learned = LearnedPositionalEncoding(seq_len, d_model)
    rope = RotaryPositionalEmbedding(d_model)
    integer = IntegerPositionalEncoding(d_model)
    binary = BinaryPositionalEncoding(d_model)
    
    # Generate encodings
    encodings = {
        'Sinusoidal': sinusoidal.forward(seq_len),
        'Learned': learned.forward(seq_len),
        'Integer': integer.forward(seq_len, normalize=True),
        'Binary': binary.forward(seq_len)
    }
    
    # RoPE is applied to vectors, so create sample and rotate
    sample = np.random.randn(seq_len, d_model)
    encodings['RoPE (applied)'] = rope.apply_rotary_embedding(sample)
    
    # Show first few positions for each
    show_pos = min(8, seq_len)
    show_dim = min(8, d_model)
    
    for name, pe in encodings.items():
        print(f"\n{name}:")
        print("-" * 80)
        print("Pos |", end="")
        for d in range(show_dim):
            print(f"  D{d}   |", end="")
        print()
        print("-" * 80)
        
        for pos in range(show_pos):
            print(f"{pos:3d} |", end="")
            for d in range(show_dim):
                print(f" {pe[pos, d]:6.3f} |", end="")
            print()


def visualize_all_encodings(seq_len: int = 64, d_model: int = 16):
    """Create visualization comparing all encoding methods."""
    print("\n" + "=" * 80)
    print("GENERATING ENCODING VISUALIZATIONS")
    print("=" * 80)
    
    # Create encoders
    sinusoidal = SinusoidalPositionalEncoding(d_model)
    learned = LearnedPositionalEncoding(seq_len, d_model)
    integer = IntegerPositionalEncoding(d_model)
    binary = BinaryPositionalEncoding(d_model)
    
    # Generate encodings
    sin_pe = sinusoidal.forward(seq_len)
    learned_pe = learned.forward(seq_len)
    int_pe = integer.forward(seq_len, normalize=True)
    bin_pe = binary.forward(seq_len)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sinusoidal
    im1 = axes[0, 0].imshow(sin_pe.T, aspect='auto', cmap='RdBu_r', 
                            interpolation='nearest', vmin=-1, vmax=1)
    axes[0, 0].set_title('Sinusoidal Encoding', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Learned
    im2 = axes[0, 1].imshow(learned_pe.T, aspect='auto', cmap='viridis', 
                            interpolation='nearest')
    axes[0, 1].set_title('Learned Encoding', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Dimension')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Integer
    im3 = axes[1, 0].imshow(int_pe.T, aspect='auto', cmap='plasma', 
                            interpolation='nearest')
    axes[1, 0].set_title('Integer Encoding (Normalized)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Dimension')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Binary
    im4 = axes[1, 1].imshow(bin_pe.T, aspect='auto', cmap='binary', 
                            interpolation='nearest')
    axes[1, 1].set_title('Binary Encoding', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Dimension')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/home/claude/all_encodings.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: all_encodings.png")
    plt.close()


def benchmark_computational_cost():
    """Benchmark computational cost of different encodings."""
    import time
    
    print("\n" + "=" * 80)
    print("COMPUTATIONAL COST BENCHMARK")
    print("=" * 80)
    
    seq_len = 512
    d_model = 512
    iterations = 1000
    
    print(f"Configuration: seq_len={seq_len}, d_model={d_model}, iterations={iterations}\n")
    
    # Sinusoidal
    sin_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
    start = time.time()
    for _ in range(iterations):
        _ = sin_enc.forward(seq_len)
    sin_time = time.time() - start
    
    # Integer
    int_enc = IntegerPositionalEncoding(d_model)
    start = time.time()
    for _ in range(iterations):
        _ = int_enc.forward(seq_len)
    int_time = time.time() - start
    
    # Binary
    bin_enc = BinaryPositionalEncoding(d_model)
    start = time.time()
    for _ in range(iterations):
        _ = bin_enc.forward(seq_len)
    bin_time = time.time() - start
    
    # RoPE (including rotation)
    rope_enc = RotaryPositionalEmbedding(d_model, max_len=seq_len)
    x = np.random.randn(seq_len, d_model)
    start = time.time()
    for _ in range(iterations):
        _ = rope_enc.apply_rotary_embedding(x)
    rope_time = time.time() - start
    
    # Results
    print("Method       | Time (ms) | Relative Speed")
    print("-" * 50)
    print(f"Sinusoidal   | {sin_time*1000:8.2f}  | {sin_time/int_time:5.2f}x")
    print(f"Integer      | {int_time*1000:8.2f}  | 1.00x (baseline)")
    print(f"Binary       | {bin_time*1000:8.2f}  | {bin_time/int_time:5.2f}x")
    print(f"RoPE         | {rope_time*1000:8.2f}  | {rope_time/int_time:5.2f}x")
    print()


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

def demo_sinusoidal():
    """Demonstrate sinusoidal encoding."""
    print("\n" + "=" * 80)
    print("1. SINUSOIDAL POSITIONAL ENCODING")
    print("=" * 80)
    
    sin_enc = SinusoidalPositionalEncoding(d_model=8)
    sin_enc.analyze_frequencies()
    
    pe = sin_enc.forward(10)
    print("Encodings for first 10 positions (first 8 dims):")
    print("-" * 80)
    for pos in range(10):
        print(f"Pos {pos}: {pe[pos, :8]}")
    print()


def demo_rope():
    """Demonstrate RoPE."""
    print("\n" + "=" * 80)
    print("2. ROTARY POSITIONAL EMBEDDING (RoPE)")
    print("=" * 80)
    
    rope = RotaryPositionalEmbedding(d_model=8)
    rope.demonstrate_relative_property()


def demo_alibi():
    """Demonstrate ALiBi."""
    print("\n" + "=" * 80)
    print("3. ALiBi (ATTENTION WITH LINEAR BIASES)")
    print("=" * 80)
    
    alibi = ALiBiPositionalBias(num_heads=8)
    alibi.visualize_attention_spans(seq_len=15)


def demo_comparisons():
    """Run all comparison demos."""
    compare_all_encodings(seq_len=16, d_model=8)
    visualize_all_encodings(seq_len=64, d_model=16)
    benchmark_computational_cost()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPLETE POSITIONAL ENCODING IMPLEMENTATION")
    print("All Major Techniques with Detailed Analysis")
    print("=" * 80)
    
    # Run individual demonstrations
    demo_sinusoidal()
    demo_rope()
    demo_alibi()
    
    # Run comparisons
    demo_comparisons()
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
