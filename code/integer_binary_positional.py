"""
Integer and Binary Positional Encoding Implementation
======================================================

This file contains:
1. Integer positional encoding (basic, normalized, multi-scale, learnable)
2. Binary positional encoding (standard, normalized, Gray code, multi-level)
3. Hybrid encodings combining multiple approaches
4. Comprehensive visualization and analysis tools
5. Comparison with sinusoidal encoding

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: INTEGER POSITIONAL ENCODING
# ============================================================================

class IntegerPositionalEncoding:
    """
    Integer positional encoding.
    
    Directly uses position index, normalized to [0, 1] range.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize integer positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, seq_len: int, normalize: bool = True) -> np.ndarray:
        """
        Generate integer positional encodings.
        
        Args:
            seq_len: Sequence length
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        
        if normalize:
            # Normalize to [0, 1]
            if seq_len > 1:
                positions = positions / (seq_len - 1)
            else:
                positions = positions * 0  # Single position -> 0
        
        # Repeat across all dimensions
        pe = np.repeat(positions, self.d_model, axis=1)  # (seq_len, d_model)
        
        return pe
    
    def visualize(self, seq_len: int = 20):
        """Visualize integer positional encoding."""
        pe = self.forward(seq_len, normalize=True)
        
        print(f"Integer Positional Encoding (normalized)")
        print("=" * 80)
        print(f"Dimension: {self.d_model}, Sequence length: {seq_len}")
        print()
        
        # Show first 8 dimensions for first 10 positions
        show_pos = min(10, seq_len)
        show_dim = min(8, self.d_model)
        
        print("Position |", end="")
        for d in range(show_dim):
            print(f"  Dim{d}  |", end="")
        print()
        print("-" * 80)
        
        for pos in range(show_pos):
            print(f"   {pos:2d}    |", end="")
            for d in range(show_dim):
                print(f" {pe[pos, d]:6.3f} |", end="")
            print()


class MultiScaleIntegerEncoding:
    """
    Multi-scale integer encoding.
    
    Different dimensions use different scaling factors.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, alpha: float = 1.0):
        """
        Initialize multi-scale integer encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            alpha: Scaling exponent (controls progression)
        """
        self.d_model = d_model
        self.max_len = max_len
        self.alpha = alpha
        
        # Compute scale factors for each dimension
        self.scales = self._compute_scales()
    
    def _compute_scales(self) -> np.ndarray:
        """Compute dimension-specific scale factors."""
        dim_indices = np.arange(self.d_model)
        scales = (dim_indices / (self.d_model - 1)) ** self.alpha
        return scales
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate multi-scale integer encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        
        # Normalize positions
        if seq_len > 1:
            normalized_positions = positions / (seq_len - 1)
        else:
            normalized_positions = positions * 0
        
        # Apply dimension-specific scales
        pe = normalized_positions * self.scales[np.newaxis, :]  # (seq_len, d_model)
        
        return pe


class LearnableIntegerEncoding:
    """
    Learnable integer encoding.
    
    Starts with integer encoding, adds learnable weights and biases.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, seed: int = 42):
        """
        Initialize learnable integer encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            seed: Random seed
        """
        np.random.seed(seed)
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Initialize learnable parameters
        self.weights = np.ones(d_model)  # Start with identity
        self.biases = np.zeros(d_model)  # Start with no bias
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate learnable integer encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        
        if seq_len > 1:
            normalized_positions = positions / (seq_len - 1)
        else:
            normalized_positions = positions * 0
        
        # Apply learnable transformation: w * pos + b
        pe = normalized_positions * self.weights + self.biases
        
        return pe
    
    def update_parameters(self, weight_grad: np.ndarray, bias_grad: np.ndarray, 
                         learning_rate: float = 0.01):
        """
        Update learnable parameters.
        
        Args:
            weight_grad: Gradient for weights
            bias_grad: Gradient for biases
            learning_rate: Learning rate
        """
        self.weights -= learning_rate * weight_grad
        self.biases -= learning_rate * bias_grad


# ============================================================================
# PART 2: BINARY POSITIONAL ENCODING
# ============================================================================

class BinaryPositionalEncoding:
    """
    Binary positional encoding.
    
    Represents position as binary number, each dimension is one bit.
    """
    
    def __init__(self, d_model: int, max_len: int = None):
        """
        Initialize binary positional encoding.
        
        Args:
            d_model: Embedding dimension (number of bits)
            max_len: Maximum sequence length (2^d_model by default)
        """
        self.d_model = d_model
        
        if max_len is None:
            # Maximum representable: 2^d_model
            self.max_len = 2 ** d_model
        else:
            self.max_len = max_len
            
            # Check if we have enough bits
            required_bits = math.ceil(math.log2(max(max_len, 1)))
            if required_bits > d_model:
                print(f"Warning: {d_model} bits may not be enough for {max_len} positions")
    
    def _position_to_binary(self, position: int) -> np.ndarray:
        """
        Convert position to binary representation.
        
        Args:
            position: Position index
            
        Returns:
            Binary array (d_model,)
        """
        binary = np.zeros(self.d_model, dtype=np.float32)
        
        for i in range(self.d_model):
            # Extract i-th bit: (position // 2^i) % 2
            binary[i] = (position >> i) & 1
        
        return binary
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate binary positional encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        pe = np.zeros((seq_len, self.d_model))
        
        for pos in range(seq_len):
            pe[pos] = self._position_to_binary(pos)
        
        return pe
    
    def visualize(self, seq_len: int = 16):
        """Visualize binary positional encoding."""
        pe = self.forward(seq_len)
        
        print(f"Binary Positional Encoding")
        print("=" * 80)
        print(f"Bits: {self.d_model}, Positions shown: {seq_len}")
        print()
        
        # Show all bits for visualization
        show_pos = min(seq_len, 20)
        show_bits = min(self.d_model, 16)
        
        print("Pos | Binary  |", end="")
        for b in range(show_bits):
            print(f" B{b} |", end="")
        print()
        print("-" * 80)
        
        for pos in range(show_pos):
            # Show decimal and binary representation
            binary_str = format(pos, f'0{show_bits}b')[::-1]  # LSB first
            print(f"{pos:3d} | {binary_str} |", end="")
            
            for b in range(show_bits):
                print(f"  {int(pe[pos, b])} |", end="")
            print()


class NormalizedBinaryEncoding:
    """
    Binary encoding with normalization to [-1, +1] or [0, 1].
    """
    
    def __init__(self, d_model: int, max_len: int = None, 
                 output_range: str = "centered"):
        """
        Initialize normalized binary encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            output_range: "centered" for [-1,+1] or "positive" for [0,1]
        """
        self.binary_encoder = BinaryPositionalEncoding(d_model, max_len)
        self.output_range = output_range
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate normalized binary encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Normalized positional encodings (seq_len, d_model)
        """
        # Get binary encoding
        binary_pe = self.binary_encoder.forward(seq_len)
        
        if self.output_range == "centered":
            # Map {0, 1} to {-1, +1}
            normalized_pe = 2 * binary_pe - 1
        else:
            # Keep as {0, 1}
            normalized_pe = binary_pe
        
        return normalized_pe


class SmoothBinaryEncoding:
    """
    Binary encoding with sigmoid smoothing.
    """
    
    def __init__(self, d_model: int, max_len: int = None, temperature: float = 5.0):
        """
        Initialize smooth binary encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            temperature: Temperature for sigmoid (higher = more binary)
        """
        self.binary_encoder = BinaryPositionalEncoding(d_model, max_len)
        self.temperature = temperature
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate smooth binary encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Smooth positional encodings (seq_len, d_model)
        """
        # Get binary encoding
        binary_pe = self.binary_encoder.forward(seq_len)
        
        # Map to {-1, +1} then apply sigmoid
        centered = 2 * binary_pe - 1
        smooth_pe = 1.0 / (1.0 + np.exp(-self.temperature * centered))
        
        return smooth_pe


class GrayCodeEncoding:
    """
    Gray code positional encoding.
    
    Adjacent positions differ by exactly 1 bit.
    """
    
    def __init__(self, d_model: int, max_len: int = None):
        """
        Initialize Gray code encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len if max_len else 2 ** d_model
    
    def _binary_to_gray(self, n: int) -> int:
        """Convert binary number to Gray code."""
        return n ^ (n >> 1)
    
    def _position_to_gray(self, position: int) -> np.ndarray:
        """
        Convert position to Gray code representation.
        
        Args:
            position: Position index
            
        Returns:
            Gray code array (d_model,)
        """
        gray = self._binary_to_gray(position)
        
        gray_array = np.zeros(self.d_model, dtype=np.float32)
        for i in range(self.d_model):
            gray_array[i] = (gray >> i) & 1
        
        return gray_array
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate Gray code positional encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        pe = np.zeros((seq_len, self.d_model))
        
        for pos in range(seq_len):
            pe[pos] = self._position_to_gray(pos)
        
        return pe
    
    def compare_with_binary(self, seq_len: int = 16):
        """Compare Gray code with standard binary."""
        print(f"Gray Code vs. Standard Binary Comparison")
        print("=" * 80)
        
        binary_encoder = BinaryPositionalEncoding(self.d_model)
        binary_pe = binary_encoder.forward(seq_len)
        gray_pe = self.forward(seq_len)
        
        show_pos = min(seq_len, 16)
        show_bits = min(self.d_model, 8)
        
        print("Pos | Standard Binary | Gray Code    | Hamming Dist from Prev")
        print("-" * 80)
        
        for pos in range(show_pos):
            binary_str = ''.join([str(int(binary_pe[pos, i])) for i in range(show_bits)])
            gray_str = ''.join([str(int(gray_pe[pos, i])) for i in range(show_bits)])
            
            if pos > 0:
                # Calculate Hamming distance from previous position
                gray_diff = np.sum(np.abs(gray_pe[pos] - gray_pe[pos-1]))
            else:
                gray_diff = 0
            
            print(f"{pos:3d} | {binary_str:>16s} | {gray_str:>12s} | {int(gray_diff)}")


class MultiLevelBinaryEncoding:
    """
    Multi-level binary encoding.
    
    Encodes position at multiple scales simultaneously.
    """
    
    def __init__(self, d_model: int, num_levels: int = 3):
        """
        Initialize multi-level binary encoding.
        
        Args:
            d_model: Total embedding dimension
            num_levels: Number of scale levels
        """
        assert d_model % num_levels == 0, "d_model must be divisible by num_levels"
        
        self.d_model = d_model
        self.num_levels = num_levels
        self.bits_per_level = d_model // num_levels
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate multi-level binary encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        pe = np.zeros((seq_len, self.d_model))
        
        for pos in range(seq_len):
            offset = 0
            
            for level in range(self.num_levels):
                # Scale position for this level
                scale = 2 ** level
                scaled_pos = pos // scale
                
                # Convert to binary for this level
                for bit in range(self.bits_per_level):
                    pe[pos, offset + bit] = (scaled_pos >> bit) & 1
                
                offset += self.bits_per_level
        
        return pe


# ============================================================================
# PART 3: HYBRID ENCODINGS
# ============================================================================

class HybridIntegerBinaryEncoding:
    """
    Hybrid encoding combining integer and binary.
    
    First half: smooth integer encoding
    Second half: discrete binary encoding
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize hybrid encoding.
        
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
        """
        assert d_model % 2 == 0, "d_model must be even for hybrid encoding"
        
        self.d_model = d_model
        self.max_len = max_len
        
        self.d_integer = d_model // 2
        self.d_binary = d_model // 2
        
        self.integer_encoder = IntegerPositionalEncoding(self.d_integer, max_len)
        self.binary_encoder = BinaryPositionalEncoding(self.d_binary, max_len)
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Generate hybrid encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        # Get integer and binary parts
        integer_pe = self.integer_encoder.forward(seq_len, normalize=True)
        binary_pe = self.binary_encoder.forward(seq_len)
        
        # Concatenate
        hybrid_pe = np.concatenate([integer_pe, binary_pe], axis=1)
        
        return hybrid_pe


# ============================================================================
# PART 4: COMPARISON AND ANALYSIS
# ============================================================================

def compare_encodings(seq_len: int = 32, d_model: int = 8):
    """
    Compare different positional encoding methods.
    """
    print("\n" + "=" * 80)
    print("POSITIONAL ENCODING COMPARISON")
    print("=" * 80)
    print(f"Sequence length: {seq_len}, Embedding dimension: {d_model}")
    print()
    
    # Create encoders
    integer_enc = IntegerPositionalEncoding(d_model)
    binary_enc = BinaryPositionalEncoding(d_model)
    gray_enc = GrayCodeEncoding(d_model)
    
    # Generate encodings
    integer_pe = integer_enc.forward(seq_len, normalize=True)
    binary_pe = binary_enc.forward(seq_len)
    gray_pe = gray_enc.forward(seq_len)
    
    # Show first few positions
    show_pos = min(8, seq_len)
    show_dim = min(8, d_model)
    
    print("\nInteger Encoding (first 8 dims):")
    print("-" * 80)
    for pos in range(show_pos):
        print(f"Pos {pos}: {integer_pe[pos, :show_dim]}")
    
    print("\nBinary Encoding (first 8 dims):")
    print("-" * 80)
    for pos in range(show_pos):
        print(f"Pos {pos}: {binary_pe[pos, :show_dim]}")
    
    print("\nGray Code Encoding (first 8 dims):")
    print("-" * 80)
    for pos in range(show_pos):
        print(f"Pos {pos}: {gray_pe[pos, :show_dim]}")


def analyze_frequency_content(seq_len: int = 64):
    """
    Analyze frequency content of different encodings.
    """
    print("\n" + "=" * 80)
    print("FREQUENCY ANALYSIS")
    print("=" * 80)
    
    d_model = 8
    binary_enc = BinaryPositionalEncoding(d_model)
    binary_pe = binary_enc.forward(seq_len)
    
    print(f"\nBinary encoding bit frequencies:")
    print("-" * 80)
    
    for dim in range(d_model):
        # Count transitions (0->1 or 1->0)
        bit_sequence = binary_pe[:, dim]
        transitions = np.sum(np.abs(np.diff(bit_sequence)))
        frequency = transitions / seq_len
        
        print(f"Bit {dim}: {transitions:3d} transitions, frequency = {frequency:.4f}")
    
    print("\nObservation:")
    print("- Lower bits change more frequently (high frequency)")
    print("- Higher bits change less frequently (low frequency)")
    print("- Similar to sinusoidal encoding's frequency decomposition!")


def hamming_distance_analysis(seq_len: int = 16):
    """
    Analyze Hamming distances between adjacent positions.
    """
    print("\n" + "=" * 80)
    print("HAMMING DISTANCE ANALYSIS")
    print("=" * 80)
    
    d_model = 8
    
    # Standard binary
    binary_enc = BinaryPositionalEncoding(d_model)
    binary_pe = binary_enc.forward(seq_len)
    
    # Gray code
    gray_enc = GrayCodeEncoding(d_model)
    gray_pe = gray_enc.forward(seq_len)
    
    print("\nHamming distances between adjacent positions:")
    print("-" * 80)
    print("Position | Standard Binary | Gray Code")
    print("-" * 80)
    
    for pos in range(1, seq_len):
        binary_dist = np.sum(np.abs(binary_pe[pos] - binary_pe[pos-1]))
        gray_dist = np.sum(np.abs(gray_pe[pos] - gray_pe[pos-1]))
        
        print(f"{pos:3d}->{pos-1:3d}  |        {int(binary_dist)}        |     {int(gray_dist)}")
    
    print("\nObservation:")
    print("- Standard binary: distance varies (1 to multiple bits)")
    print("- Gray code: distance is always exactly 1 bit")


def visualize_encoding_patterns(seq_len: int = 64, d_model: int = 8):
    """
    Visualize encoding patterns as heatmaps.
    """
    print("\n" + "=" * 80)
    print("ENCODING PATTERN VISUALIZATION")
    print("=" * 80)
    
    # Create encodings
    integer_enc = IntegerPositionalEncoding(d_model)
    binary_enc = BinaryPositionalEncoding(d_model)
    multi_enc = MultiScaleIntegerEncoding(d_model)
    
    integer_pe = integer_enc.forward(seq_len, normalize=True)
    binary_pe = binary_enc.forward(seq_len)
    multi_pe = multi_enc.forward(seq_len)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Integer encoding
    im1 = axes[0].imshow(integer_pe.T, aspect='auto', cmap='viridis', 
                         interpolation='nearest')
    axes[0].set_title('Integer Encoding')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0])
    
    # Binary encoding
    im2 = axes[1].imshow(binary_pe.T, aspect='auto', cmap='binary', 
                         interpolation='nearest')
    axes[1].set_title('Binary Encoding')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Dimension')
    plt.colorbar(im2, ax=axes[1])
    
    # Multi-scale encoding
    im3 = axes[2].imshow(multi_pe.T, aspect='auto', cmap='plasma', 
                         interpolation='nearest')
    axes[2].set_title('Multi-Scale Integer Encoding')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Dimension')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/home/claude/encoding_patterns.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: encoding_patterns.png")
    plt.close()


def memory_and_computation_analysis():
    """
    Analyze memory and computational requirements.
    """
    print("\n" + "=" * 80)
    print("MEMORY AND COMPUTATION ANALYSIS")
    print("=" * 80)
    
    seq_len = 512
    d_model = 512
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {d_model}")
    
    # Memory requirements
    memory_per_position = d_model * 4  # 4 bytes per float32
    total_memory = seq_len * memory_per_position
    
    print(f"\nMemory Requirements:")
    print(f"  Per position: {memory_per_position} bytes = {memory_per_position/1024:.2f} KB")
    print(f"  Total: {total_memory} bytes = {total_memory/1024:.2f} KB")
    
    # Computational complexity
    print(f"\nComputational Complexity (per position):")
    print(f"  Integer encoding: O({d_model}) - just assignment")
    print(f"  Binary encoding: O({d_model}) - bit operations")
    print(f"  Sinusoidal: O({d_model}) - trigonometric functions")
    
    print(f"\nRelative speed (approximate):")
    print(f"  Integer: 1.0x (fastest)")
    print(f"  Binary: 1.2x (very fast, bit ops)")
    print(f"  Sinusoidal: 5-10x (slower, sin/cos)")


# ============================================================================
# PART 5: DEMONSTRATIONS
# ============================================================================

def demo_integer_encoding():
    """Demonstrate integer positional encoding."""
    print("\n" + "=" * 80)
    print("INTEGER POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 80)
    
    # Basic integer encoding
    print("\n1. Basic Integer Encoding")
    print("-" * 80)
    int_enc = IntegerPositionalEncoding(d_model=8)
    int_enc.visualize(seq_len=10)
    
    # Multi-scale integer encoding
    print("\n2. Multi-Scale Integer Encoding")
    print("-" * 80)
    multi_enc = MultiScaleIntegerEncoding(d_model=8, alpha=1.0)
    pe = multi_enc.forward(10)
    
    print("Position | D0    | D1    | D2    | D3    | D4    | D5    | D6    | D7")
    print("-" * 80)
    for pos in range(10):
        print(f"   {pos:2d}    |", end="")
        for d in range(8):
            print(f" {pe[pos, d]:.3f} |", end="")
        print()


def demo_binary_encoding():
    """Demonstrate binary positional encoding."""
    print("\n" + "=" * 80)
    print("BINARY POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 80)
    
    # Standard binary
    print("\n1. Standard Binary Encoding")
    print("-" * 80)
    binary_enc = BinaryPositionalEncoding(d_model=8)
    binary_enc.visualize(seq_len=16)
    
    # Gray code
    print("\n2. Gray Code Encoding")
    print("-" * 80)
    gray_enc = GrayCodeEncoding(d_model=8)
    gray_enc.compare_with_binary(seq_len=16)
    
    # Smooth binary
    print("\n3. Smooth Binary Encoding")
    print("-" * 80)
    smooth_enc = SmoothBinaryEncoding(d_model=8, temperature=5.0)
    pe = smooth_enc.forward(8)
    
    print("Position | B0    | B1    | B2    | B3    | B4    | B5    | B6    | B7")
    print("-" * 80)
    for pos in range(8):
        print(f"   {pos:2d}    |", end="")
        for d in range(8):
            print(f" {pe[pos, d]:.3f} |", end="")
        print()


def demo_hybrid_encoding():
    """Demonstrate hybrid encoding."""
    print("\n" + "=" * 80)
    print("HYBRID ENCODING DEMONSTRATION")
    print("=" * 80)
    
    hybrid_enc = HybridIntegerBinaryEncoding(d_model=8)
    pe = hybrid_enc.forward(8)
    
    print("\nHybrid Integer-Binary Encoding")
    print("First 4 dims: Integer | Last 4 dims: Binary")
    print("-" * 80)
    print("Pos | Integer part  | Binary part")
    print("-" * 80)
    
    for pos in range(8):
        int_part = pe[pos, :4]
        bin_part = pe[pos, 4:]
        
        int_str = ' '.join([f'{v:.2f}' for v in int_part])
        bin_str = ' '.join([f'{int(v)}' for v in bin_part])
        
        print(f"{pos:3d} | {int_str} | {bin_str}")


def demo_practical_application():
    """Demonstrate practical application."""
    print("\n" + "=" * 80)
    print("PRACTICAL APPLICATION: TEXT SEQUENCE")
    print("=" * 80)
    
    # Simulate a text sequence
    text = "The quick brown fox jumps"
    tokens = text.split()
    seq_len = len(tokens)
    
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Sequence length: {seq_len}")
    
    # Create encodings
    d_model = 8
    integer_enc = IntegerPositionalEncoding(d_model)
    binary_enc = BinaryPositionalEncoding(d_model)
    
    integer_pe = integer_enc.forward(seq_len, normalize=True)
    binary_pe = binary_enc.forward(seq_len)
    
    print("\n" + "-" * 80)
    print("Position | Token  | Integer PE (first 4) | Binary PE (first 4)")
    print("-" * 80)
    
    for pos, token in enumerate(tokens):
        int_str = ' '.join([f'{v:.2f}' for v in integer_pe[pos, :4]])
        bin_str = ' '.join([f'{int(v)}' for v in binary_pe[pos, :4]])
        
        print(f"   {pos:2d}    | {token:6s} | {int_str} | {bin_str}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTEGER AND BINARY POSITIONAL ENCODING")
    print("Complete Implementation and Analysis")
    print("=" * 80)
    
    # Run demonstrations
    demo_integer_encoding()
    demo_binary_encoding()
    demo_hybrid_encoding()
    demo_practical_application()
    
    # Run analyses
    compare_encodings(seq_len=16, d_model=8)
    analyze_frequency_content(seq_len=64)
    hamming_distance_analysis(seq_len=16)
    memory_and_computation_analysis()
    
    # Create visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    visualize_encoding_patterns(seq_len=64, d_model=16)
    
    print("\n" + "=" * 80)
    print("All demonstrations and analyses completed!")
    print("=" * 80)
