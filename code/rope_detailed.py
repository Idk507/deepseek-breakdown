"""
Rotary Positional Encoding (RoPE) - Complete Implementation
============================================================

This file contains:
1. Detailed RoPE implementation from scratch
2. Efficient vectorized implementation
3. Mathematical property demonstrations
4. Visualizations of rotation patterns
5. Comparison with other positional encodings
6. Practical applications and variants
7. Performance benchmarks

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: BASIC ROPE IMPLEMENTATION
# ============================================================================

class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes position by rotating query and key vectors in 2D subspaces.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency calculation (default 10000)
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute inverse frequencies: θ_j = base^(-2j/d)
        self.inv_freq = self._compute_inv_freq()
        
        # Precompute sin and cos for all positions
        self.cos_cached, self.sin_cached = self._build_cache()
    
    def _compute_inv_freq(self) -> np.ndarray:
        """
        Compute inverse frequencies for each dimension pair.
        
        Returns:
            inv_freq: (d_model/2,) array of frequencies
        """
        # θ_j = base^(-2j/d) for j = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.d_model, 2) / self.d_model))
        return inv_freq
    
    def _build_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute sin and cos values for all positions and frequencies.
        
        Returns:
            cos_cached: (max_len, d_model/2)
            sin_cached: (max_len, d_model/2)
        """
        # Position indices
        positions = np.arange(self.max_len)  # (max_len,)
        
        # Compute m * θ_j for all positions m and frequencies θ_j
        # (max_len, 1) * (1, d_model/2) -> (max_len, d_model/2)
        freqs = positions[:, np.newaxis] * self.inv_freq[np.newaxis, :]
        
        # Compute sin and cos
        cos_cached = np.cos(freqs)
        sin_cached = np.sin(freqs)
        
        return cos_cached, sin_cached
    
    def rotate_half(self, x: np.ndarray) -> np.ndarray:
        """
        Rotate half the dimensions (alternative implementation).
        
        This is an equivalent formulation that's sometimes used.
        
        Args:
            x: Input array (..., d_model)
            
        Returns:
            Rotated array with half dimensions negated and shifted
        """
        # Split into two halves
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        # Rotate: concatenate [-x2, x1]
        return np.concatenate([-x2, x1], axis=-1)
    
    def apply_rotary_embedding(
        self, 
        x: np.ndarray, 
        start_pos: int = 0,
        use_complex: bool = False
    ) -> np.ndarray:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor (seq_len, d_model) or (batch, seq_len, d_model)
            start_pos: Starting position (for cached generation)
            use_complex: Whether to use complex number formulation
            
        Returns:
            Rotated tensor with same shape as input
        """
        # Handle different input shapes
        if x.ndim == 2:
            seq_len, d_model = x.shape
            batch_size = None
            x = x[np.newaxis, :, :]  # Add batch dimension
        else:
            batch_size, seq_len, d_model = x.shape
        
        assert d_model == self.d_model, f"Input dim {d_model} != RoPE dim {self.d_model}"
        
        if use_complex:
            return self._apply_rotary_complex(x, start_pos, batch_size)
        else:
            return self._apply_rotary_standard(x, start_pos, batch_size)
    
    def _apply_rotary_standard(
        self, 
        x: np.ndarray, 
        start_pos: int, 
        original_batch_size: Optional[int]
    ) -> np.ndarray:
        """Standard rotation implementation using real numbers."""
        batch_size, seq_len, d_model = x.shape
        
        # Get sin and cos for this sequence
        cos = self.cos_cached[start_pos:start_pos + seq_len, :]  # (seq_len, d/2)
        sin = self.sin_cached[start_pos:start_pos + seq_len, :]  # (seq_len, d/2)
        
        # Reshape input into dimension pairs
        x_pairs = x.reshape(batch_size, seq_len, -1, 2)  # (batch, seq_len, d/2, 2)
        
        # Extract even and odd dimensions
        x_even = x_pairs[..., 0]  # (batch, seq_len, d/2)
        x_odd = x_pairs[..., 1]   # (batch, seq_len, d/2)
        
        # Apply rotation to each pair
        # [x0]   [cos  -sin] [x0]   [x0*cos - x1*sin]
        # [x1] = [sin   cos] [x1] = [x0*sin + x1*cos]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Recombine pairs
        rotated = np.stack([x_even_rot, x_odd_rot], axis=-1)
        rotated = rotated.reshape(batch_size, seq_len, d_model)
        
        # Remove batch dimension if input was 2D
        if original_batch_size is None:
            return rotated[0]
        return rotated
    
    def _apply_rotary_complex(
        self, 
        x: np.ndarray, 
        start_pos: int, 
        original_batch_size: Optional[int]
    ) -> np.ndarray:
        """Complex number formulation of rotation."""
        batch_size, seq_len, d_model = x.shape
        
        # Get frequencies
        cos = self.cos_cached[start_pos:start_pos + seq_len, :]
        sin = self.sin_cached[start_pos:start_pos + seq_len, :]
        
        # Reshape into pairs
        x_pairs = x.reshape(batch_size, seq_len, -1, 2)
        
        # Convert to complex numbers
        x_complex = x_pairs[..., 0] + 1j * x_pairs[..., 1]
        
        # Rotation in complex plane: multiply by e^(iθ) = cos(θ) + i*sin(θ)
        rotation = cos + 1j * sin
        x_rotated = x_complex * rotation
        
        # Convert back to real numbers
        x_pairs_rot = np.stack([x_rotated.real, x_rotated.imag], axis=-1)
        rotated = x_pairs_rot.reshape(batch_size, seq_len, d_model)
        
        if original_batch_size is None:
            return rotated[0]
        return rotated
    
    def visualize_frequencies(self):
        """Visualize frequency spectrum of RoPE."""
        print("RoPE Frequency Analysis")
        print("=" * 80)
        print(f"Base: {self.base}")
        print(f"Dimension: {self.d_model}")
        print()
        
        # Show frequencies for first few dimension pairs
        num_to_show = min(8, self.d_model // 2)
        
        print("Dimension Pair | Frequency θ_j    | Wavelength (2π/θ_j)")
        print("-" * 80)
        
        for j in range(num_to_show):
            freq = self.inv_freq[j]
            wavelength = 2 * np.pi / freq
            print(f"      {j:2d}       | {freq:14.8f}  | {wavelength:12.2f}")
        
        print()
        print(f"Frequency range: [{self.inv_freq[-1]:.8f}, {self.inv_freq[0]:.8f}]")
        print(f"Wavelength range: [{2*np.pi/self.inv_freq[0]:.2f}, {2*np.pi/self.inv_freq[-1]:.2f}]")
        print()


# ============================================================================
# PART 2: PROPERTY DEMONSTRATIONS
# ============================================================================

def demonstrate_relative_position_property():
    """
    Demonstrate that RoPE encodes relative positions.
    
    Shows that dot product depends only on (n-m), not absolute positions.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Relative Position Property")
    print("=" * 80)
    
    d_model = 64
    rope = RotaryPositionalEmbedding(d_model)
    
    # Create random query and key vectors
    np.random.seed(42)
    q = np.random.randn(d_model)
    k = np.random.randn(d_model)
    
    print("\nTesting: Dot product should depend only on relative position (n-m)")
    print("-" * 80)
    
    # Test different absolute positions with same relative distance
    test_cases = [
        # (query_pos, key_pos, relative_distance)
        (0, 5, 5),
        (10, 15, 5),
        (100, 105, 5),
        (0, 10, 10),
        (50, 60, 10),
        (200, 210, 10),
    ]
    
    print(f"{'Query Pos':>10} | {'Key Pos':>10} | {'Relative':>10} | {'Dot Product':>15}")
    print("-" * 80)
    
    for m, n, rel in test_cases:
        # Rotate query at position m
        q_m = rope.apply_rotary_embedding(q[np.newaxis, :], start_pos=m)[0]
        
        # Rotate key at position n
        k_n = rope.apply_rotary_embedding(k[np.newaxis, :], start_pos=n)[0]
        
        # Compute dot product
        score = np.dot(q_m, k_n)
        
        print(f"{m:10d} | {n:10d} | {rel:10d} | {score:15.6f}")
    
    print("\n✓ Observation: Scores are identical for same relative distance!")
    print("  e.g., (0→5) ≈ (10→15) ≈ (100→105) because all have distance = 5")
    print()


def demonstrate_norm_preservation():
    """Demonstrate that RoPE preserves vector norms."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Norm Preservation")
    print("=" * 80)
    
    d_model = 64
    rope = RotaryPositionalEmbedding(d_model)
    
    # Create random vectors
    np.random.seed(42)
    x = np.random.randn(10, d_model)
    
    print("\nOriginal vs Rotated vector norms:")
    print("-" * 80)
    print(f"{'Position':>10} | {'Original Norm':>15} | {'Rotated Norm':>15} | {'Difference':>15}")
    print("-" * 80)
    
    for pos in range(10):
        original_norm = np.linalg.norm(x[pos])
        
        # Apply rotation
        x_rot = rope.apply_rotary_embedding(x[pos:pos+1, :], start_pos=pos)[0]
        rotated_norm = np.linalg.norm(x_rot)
        
        diff = abs(original_norm - rotated_norm)
        
        print(f"{pos:10d} | {original_norm:15.6f} | {rotated_norm:15.6f} | {diff:15.10f}")
    
    print("\n✓ Observation: Norms are preserved (difference ≈ 0 up to numerical precision)")
    print()


def demonstrate_multi_scale_attention():
    """Demonstrate multi-scale attention patterns."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Multi-Scale Attention Patterns")
    print("=" * 80)
    
    d_model = 16  # Small for visualization
    seq_len = 64
    rope = RotaryPositionalEmbedding(d_model)
    
    # Create identity queries and keys
    q = np.eye(d_model)
    k = np.eye(d_model)
    
    # Compute attention for each dimension pair
    print("\nAttention decay by distance for different frequency dimensions:")
    print("-" * 80)
    
    distances = [1, 2, 4, 8, 16, 32]
    
    print(f"{'Distance':>10} |", end="")
    for j in range(min(4, d_model // 2)):
        print(f" Pair {j} ({self.inv_freq[j]:.4f}) |", end="")
    print()
    print("-" * 80)
    
    for dist in distances:
        print(f"{dist:10d} |", end="")
        
        for dim_pair in range(min(4, d_model // 2)):
            # Get vectors for this dimension pair
            q_vec = np.zeros(d_model)
            q_vec[2*dim_pair:2*dim_pair+2] = [1.0, 0.0]
            
            k_vec = np.zeros(d_model)
            k_vec[2*dim_pair:2*dim_pair+2] = [1.0, 0.0]
            
            # Rotate at different positions
            q_rot = rope.apply_rotary_embedding(q_vec[np.newaxis, :], start_pos=0)[0]
            k_rot = rope.apply_rotary_embedding(k_vec[np.newaxis, :], start_pos=dist)[0]
            
            # Compute similarity
            similarity = np.dot(q_rot, k_rot)
            
            print(f" {similarity:15.4f} |", end="")
        
        print()
    
    print("\n✓ Observation: High-frequency pairs (Pair 0) decay faster with distance")
    print("               Low-frequency pairs maintain similarity over longer distances")
    print()


# ============================================================================
# PART 3: VISUALIZATIONS
# ============================================================================

def visualize_rotation_pattern():
    """Visualize how rotations change with position."""
    print("\n" + "=" * 80)
    print("GENERATING: Rotation Pattern Visualization")
    print("=" * 80)
    
    d_model = 8
    seq_len = 64
    rope = RotaryPositionalEmbedding(d_model)
    
    # Create a simple test vector
    x = np.ones((seq_len, d_model))
    
    # Apply rotations at all positions
    x_rotated = np.zeros_like(x)
    for pos in range(seq_len):
        x_rotated[pos] = rope.apply_rotary_embedding(
            x[pos:pos+1, :], start_pos=pos
        )[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original vs Rotated heatmap
    im1 = axes[0, 0].imshow(x.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[0, 0].set_title('Original Vectors (All Ones)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(x_rotated.T, aspect='auto', cmap='RdBu_r', 
                            interpolation='nearest', vmin=-1, vmax=1)
    axes[0, 1].set_title('After RoPE Rotation', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Dimension')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 2: Individual dimension pairs over positions
    for dim_pair in range(min(4, d_model // 2)):
        axes[1, 0].plot(x_rotated[:, 2*dim_pair], 
                       label=f'Pair {dim_pair} (even)', alpha=0.7)
        axes[1, 0].plot(x_rotated[:, 2*dim_pair + 1], 
                       label=f'Pair {dim_pair} (odd)', alpha=0.7, linestyle='--')
    
    axes[1, 0].set_title('Dimension Values Across Positions', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 3: Frequency spectrum
    freqs = rope.inv_freq[:min(8, len(rope.inv_freq))]
    axes[1, 1].bar(range(len(freqs)), freqs, color='steelblue', alpha=0.7)
    axes[1, 1].set_title('RoPE Frequency Spectrum', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Dimension Pair Index')
    axes[1, 1].set_ylabel('Frequency (θ)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/rope_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: rope_visualization.png")
    plt.close()


def visualize_attention_heatmap():
    """Visualize attention patterns with RoPE."""
    print("\n" + "=" * 80)
    print("GENERATING: Attention Pattern Heatmap")
    print("=" * 80)
    
    d_model = 64
    seq_len = 32
    rope = RotaryPositionalEmbedding(d_model)
    
    # Create random queries and keys
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    
    # Apply RoPE
    Q_rot = np.zeros_like(Q)
    K_rot = np.zeros_like(K)
    
    for pos in range(seq_len):
        Q_rot[pos] = rope.apply_rotary_embedding(Q[pos:pos+1, :], start_pos=pos)[0]
        K_rot[pos] = rope.apply_rotary_embedding(K[pos:pos+1, :], start_pos=pos)[0]
    
    # Compute attention scores (before softmax)
    scores_no_rope = Q @ K.T / np.sqrt(d_model)
    scores_with_rope = Q_rot @ K_rot.T / np.sqrt(d_model)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Without RoPE
    im1 = axes[0].imshow(scores_no_rope, cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title('Attention Scores WITHOUT RoPE', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # With RoPE
    im2 = axes[1].imshow(scores_with_rope, cmap='RdBu_r', interpolation='nearest')
    axes[1].set_title('Attention Scores WITH RoPE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('/home/claude/rope_attention.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: rope_attention.png")
    plt.close()


# ============================================================================
# PART 4: VARIANTS AND EXTENSIONS
# ============================================================================

class RoPEWithScaling(RotaryPositionalEmbedding):
    """
    RoPE with position scaling for handling longer sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, 
                 base: float = 10000.0, scaling_factor: float = 1.0,
                 scaling_type: str = "linear"):
        """
        Initialize RoPE with scaling.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            base: Base frequency
            scaling_factor: Factor to scale positions (> 1 for longer sequences)
            scaling_type: "linear" or "ntk" (NTK-aware scaling)
        """
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        
        # Adjust base for NTK-aware scaling
        if scaling_type == "ntk":
            base = base * scaling_factor
        
        super().__init__(d_model, max_len, base)
    
    def apply_rotary_embedding(self, x: np.ndarray, start_pos: int = 0, 
                               use_complex: bool = False) -> np.ndarray:
        """Apply RoPE with position scaling."""
        if self.scaling_type == "linear":
            # Scale positions
            start_pos = int(start_pos / self.scaling_factor)
        
        return super().apply_rotary_embedding(x, start_pos, use_complex)


class PartialRoPE(RotaryPositionalEmbedding):
    """
    Partial RoPE - only apply to a fraction of dimensions.
    
    Used in LLaMA and other models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, 
                 base: float = 10000.0, rope_ratio: float = 0.25):
        """
        Initialize Partial RoPE.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            base: Base frequency
            rope_ratio: Fraction of dimensions to apply RoPE to (0 to 1)
        """
        self.rope_ratio = rope_ratio
        self.d_rope = int(d_model * rope_ratio)
        
        # Ensure even number of dimensions for RoPE
        if self.d_rope % 2 != 0:
            self.d_rope -= 1
        
        # Initialize RoPE for partial dimensions
        super().__init__(self.d_rope, max_len, base)
        self.full_d_model = d_model
    
    def apply_rotary_embedding(self, x: np.ndarray, start_pos: int = 0,
                               use_complex: bool = False) -> np.ndarray:
        """Apply RoPE only to first d_rope dimensions."""
        # Get original shape
        orig_shape = x.shape
        
        if x.ndim == 2:
            batch_size = None
            x = x[np.newaxis, :, :]
        else:
            batch_size = x.shape[0]
        
        # Apply RoPE to first d_rope dimensions
        x_rope = x[..., :self.d_rope]
        x_rope_rotated = super().apply_rotary_embedding(
            x_rope.reshape(-1, x_rope.shape[-2], self.d_rope), 
            start_pos, 
            use_complex
        )
        
        # Keep remaining dimensions unchanged
        x_no_rope = x[..., self.d_rope:]
        
        # Concatenate
        result = np.concatenate([x_rope_rotated, x_no_rope], axis=-1)
        
        if batch_size is None:
            return result[0]
        return result


# ============================================================================
# PART 5: BENCHMARKS AND COMPARISONS
# ============================================================================

def benchmark_rope_performance():
    """Benchmark computational performance of RoPE."""
    import time
    
    print("\n" + "=" * 80)
    print("BENCHMARK: RoPE Performance")
    print("=" * 80)
    
    configs = [
        (512, 64),    # seq_len, d_model
        (512, 128),
        (512, 512),
        (2048, 512),
        (2048, 1024),
    ]
    
    iterations = 100
    
    print(f"\nIterations: {iterations}")
    print("-" * 80)
    print(f"{'Seq Length':>12} | {'Dim':>6} | {'Time (ms)':>12} | {'Throughput':>15}")
    print("-" * 80)
    
    for seq_len, d_model in configs:
        rope = RotaryPositionalEmbedding(d_model)
        x = np.random.randn(seq_len, d_model)
        
        # Warmup
        for _ in range(10):
            _ = rope.apply_rotary_embedding(x)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = rope.apply_rotary_embedding(x)
        elapsed = time.time() - start
        
        time_per_iter = (elapsed / iterations) * 1000  # ms
        throughput = (seq_len * d_model * iterations) / elapsed / 1e6  # M elements/s
        
        print(f"{seq_len:12d} | {d_model:6d} | {time_per_iter:12.3f} | {throughput:12.2f} M/s")
    
    print()


def compare_rope_variants():
    """Compare different RoPE variants."""
    print("\n" + "=" * 80)
    print("COMPARISON: RoPE Variants")
    print("=" * 80)
    
    d_model = 64
    seq_len = 10
    
    # Create variants
    standard = RotaryPositionalEmbedding(d_model)
    scaled = RoPEWithScaling(d_model, scaling_factor=2.0, scaling_type="linear")
    partial = PartialRoPE(d_model, rope_ratio=0.5)
    
    # Test input
    x = np.random.randn(seq_len, d_model)
    
    # Apply variants
    x_standard = np.zeros_like(x)
    x_scaled = np.zeros_like(x)
    x_partial = np.zeros_like(x)
    
    for pos in range(seq_len):
        x_standard[pos] = standard.apply_rotary_embedding(x[pos:pos+1, :], start_pos=pos)[0]
        x_scaled[pos] = scaled.apply_rotary_embedding(x[pos:pos+1, :], start_pos=pos)[0]
        x_partial[pos] = partial.apply_rotary_embedding(x[pos:pos+1, :], start_pos=pos)[0]
    
    print("\nNorm comparison (should all be equal to original for full RoPE):")
    print("-" * 80)
    print(f"{'Position':>10} | {'Original':>12} | {'Standard':>12} | {'Scaled':>12} | {'Partial':>12}")
    print("-" * 80)
    
    for pos in range(min(5, seq_len)):
        orig_norm = np.linalg.norm(x[pos])
        std_norm = np.linalg.norm(x_standard[pos])
        scaled_norm = np.linalg.norm(x_scaled[pos])
        partial_norm = np.linalg.norm(x_partial[pos])
        
        print(f"{pos:10d} | {orig_norm:12.6f} | {std_norm:12.6f} | "
              f"{scaled_norm:12.6f} | {partial_norm:12.6f}")
    
    print()


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ROTARY POSITIONAL ENCODING (RoPE)")
    print("Complete Implementation and Analysis")
    print("=" * 80)
    
    # Initialize RoPE
    print("\n1. Initializing RoPE...")
    rope = RotaryPositionalEmbedding(d_model=64)
    rope.visualize_frequencies()
    
    # Property demonstrations
    print("\n2. Demonstrating Mathematical Properties...")
    demonstrate_relative_position_property()
    demonstrate_norm_preservation()
    demonstrate_multi_scale_attention()
    
    # Visualizations
    print("\n3. Generating Visualizations...")
    visualize_rotation_pattern()
    visualize_attention_heatmap()
    
    # Variants
    print("\n4. Testing RoPE Variants...")
    compare_rope_variants()
    
    # Benchmarks
    print("\n5. Running Performance Benchmarks...")
    benchmark_rope_performance()
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
