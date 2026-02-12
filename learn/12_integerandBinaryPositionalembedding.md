# Integer and Binary Positional Encodings: A Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Integer Positional Encoding](#integer-positional-encoding)
3. [Binary Positional Encoding](#binary-positional-encoding)
4. [Comparison with Other Methods](#comparison-with-other-methods)
5. [Advanced Variants](#advanced-variants)
6. [Practical Applications](#practical-applications)
7. [Implementation Considerations](#implementation-considerations)

---

## Introduction

### Why Alternative Positional Encodings?

While sinusoidal and learned positional encodings are popular, **integer** and **binary** positional encodings offer unique advantages:

1. **Simplicity**: Easier to understand and implement
2. **Interpretability**: Direct correspondence to position
3. **Efficiency**: Computationally lightweight
4. **Deterministic**: No learning required
5. **Structured**: Clear mathematical properties

### The Challenge

Neural networks need positional information because:
- Self-attention is **permutation invariant**
- Word order matters: "Dog bites man" ≠ "Man bites dog"
- Solution: Encode position into input embeddings

### Encoding Requirements

A good positional encoding should:
1. Be **unique** for each position
2. Be **bounded** (not grow infinitely)
3. Allow the model to learn **relative positions**
4. **Generalize** to longer sequences than seen in training

---

## Integer Positional Encoding

### 2.1 Basic Concept

**Core Idea**: Directly use the integer position index as a feature.

**Mathematical Definition**:

For position $\text{pos} \in \{0, 1, 2, ..., L-1\}$ and embedding dimension $d$:

$$\text{PE}_{\text{int}}(\text{pos}, i) = \frac{\text{pos}}{C} \quad \text{for all } i \in \{0, 1, ..., d-1\}$$

where:
- $C$ is a scaling constant (typically $L$ or $L-1$)
- The same value is repeated across all dimensions

**Example**:

Sequence length: $L = 100$
Embedding dimension: $d = 512$
Scaling: $C = 100$

```
Position 0:  PE = [0.00, 0.00, 0.00, ..., 0.00]  (512 dimensions)
Position 1:  PE = [0.01, 0.01, 0.01, ..., 0.01]
Position 2:  PE = [0.02, 0.02, 0.02, ..., 0.02]
...
Position 99: PE = [0.99, 0.99, 0.99, ..., 0.99]
```

---

### 2.2 Normalized Integer Encoding

**Problem with basic integer encoding**: Values grow linearly with position.

**Solution**: Normalize to $[0, 1]$ range.

$$\text{PE}_{\text{norm}}(\text{pos}) = \frac{\text{pos}}{L - 1} \cdot \mathbf{1}_d$$

where $\mathbf{1}_d$ is a vector of ones with dimension $d$.

**Properties**:

1. **Bounded**: Always in $[0, 1]$
2. **Uniform scaling**: Each position is equally spaced
3. **Simple**: Just one division operation

**Example Computation**:

Sequence length: $L = 10$
Dimension: $d = 4$

```
Position 0: [0/9, 0/9, 0/9, 0/9] = [0.000, 0.000, 0.000, 0.000]
Position 1: [1/9, 1/9, 1/9, 1/9] = [0.111, 0.111, 0.111, 0.111]
Position 2: [2/9, 2/9, 2/9, 2/9] = [0.222, 0.222, 0.222, 0.222]
Position 3: [3/9, 3/9, 3/9, 3/9] = [0.333, 0.333, 0.333, 0.333]
...
Position 9: [9/9, 9/9, 9/9, 9/9] = [1.000, 1.000, 1.000, 1.000]
```

---

### 2.3 Multi-Scale Integer Encoding

**Limitation of uniform scaling**: Only one "frequency" of position information.

**Solution**: Use different scales across dimensions.

$$\text{PE}_{\text{multi}}(\text{pos}, i) = \frac{\text{pos}}{L - 1} \cdot s_i$$

where $s_i$ is a dimension-specific scale factor:

$$s_i = \left(\frac{i}{d-1}\right)^\alpha$$

with $\alpha$ controlling the progression (typically $\alpha = 1$ or $\alpha = 2$).

**Example** ($\alpha = 1$):

Position: $\text{pos} = 5$
Sequence length: $L = 10$
Dimension: $d = 4$

Scale factors:
```
s_0 = 0/3 = 0.000
s_1 = 1/3 = 0.333
s_2 = 2/3 = 0.667
s_3 = 3/3 = 1.000
```

Positional encoding:
```
PE(5, 0) = (5/9) × 0.000 = 0.000
PE(5, 1) = (5/9) × 0.333 = 0.185
PE(5, 2) = (5/9) × 0.667 = 0.370
PE(5, 3) = (5/9) × 1.000 = 0.556
```

Result: $\text{PE}(5) = [0.000, 0.185, 0.370, 0.556]$

**Advantage**: Different dimensions capture position at different "resolutions."

---

### 2.4 Learnable Integer Encoding

**Hybrid approach**: Start with integer encoding, make it learnable.

$$\text{PE}_{\text{learn}}(\text{pos}, i) = w_i \cdot \frac{\text{pos}}{L - 1} + b_i$$

where:
- $w_i$ is learnable weight for dimension $i$
- $b_i$ is learnable bias for dimension $i$

**Initialization**:
```
w_i = 1.0  (identity)
b_i = 0.0  (no bias)
```

**Training**: Weights and biases are updated via backpropagation.

**Advantages**:
1. Start with interpretable baseline
2. Allow model to adapt scaling
3. Learn dimension-specific importance

---

### 2.5 Advantages of Integer Encoding

**Pros:**

1. **Simplicity**: Easiest to understand and implement
2. **Interpretability**: Direct mapping from position to encoding
3. **Deterministic**: No randomness, fully reproducible
4. **No parameters**: (for non-learnable version)
5. **Fast computation**: Just arithmetic operations

**Cons:**

1. **Limited expressiveness**: Same value across all dimensions (basic version)
2. **Linear relationships only**: Cannot capture complex patterns
3. **No frequency diversity**: Unlike sinusoidal encoding
4. **Fixed maximum length**: Cannot extrapolate beyond training length

---

## Binary Positional Encoding

### 3.1 Core Concept

**Inspiration**: Represent position as a binary number.

**Key Insight**: Binary representation naturally provides multi-scale positional information:
- Least significant bit (LSB): changes every position (frequency = 1)
- Next bit: changes every 2 positions (frequency = 0.5)
- Next bit: changes every 4 positions (frequency = 0.25)
- And so on...

This is analogous to how sinusoidal encoding uses different frequencies!

---

### 3.2 Mathematical Formulation

For position $\text{pos}$ and embedding dimension $d$:

**Binary representation**:

$$\text{pos} = \sum_{i=0}^{d-1} b_i \cdot 2^i$$

where $b_i \in \{0, 1\}$ is the $i$-th bit.

**Binary positional encoding**:

$$\text{PE}_{\text{bin}}(\text{pos}, i) = b_i = \left\lfloor \frac{\text{pos}}{2^i} \right\rfloor \bmod 2$$

**In words**: The $i$-th dimension gets the $i$-th bit of the position's binary representation.

---

### 3.3 Step-by-Step Example

**Setup**:
- Maximum position: 15 (requires 4 bits: $2^4 = 16$)
- Embedding dimension: $d = 4$

**Positions in binary**:
```
Position  0: 0000
Position  1: 0001
Position  2: 0010
Position  3: 0011
Position  4: 0100
Position  5: 0101
Position  6: 0110
Position  7: 0111
Position  8: 1000
Position  9: 1001
Position 10: 1010
Position 11: 1011
Position 12: 1100
Position 13: 1101
Position 14: 1110
Position 15: 1111
```

**Positional Encodings**:

Reading bits from right to left (LSB to MSB):

```
Position  0: [0, 0, 0, 0]  (binary: 0000)
Position  1: [1, 0, 0, 0]  (binary: 0001)
Position  2: [0, 1, 0, 0]  (binary: 0010)
Position  3: [1, 1, 0, 0]  (binary: 0011)
Position  4: [0, 0, 1, 0]  (binary: 0100)
Position  5: [1, 0, 1, 0]  (binary: 0101)
Position  6: [0, 1, 1, 0]  (binary: 0110)
Position  7: [1, 1, 1, 0]  (binary: 0111)
Position  8: [0, 0, 0, 1]  (binary: 1000)
Position  9: [1, 0, 0, 1]  (binary: 1001)
Position 10: [0, 1, 0, 1]  (binary: 1010)
Position 11: [1, 1, 0, 1]  (binary: 1011)
Position 12: [0, 0, 1, 1]  (binary: 1100)
Position 13: [1, 0, 1, 1]  (binary: 1101)
Position 14: [0, 1, 1, 1]  (binary: 1110)
Position 15: [1, 1, 1, 1]  (binary: 1111)
```

**Pattern observation**:

- **Dimension 0** (LSB): alternates every position (1, 0, 1, 0, 1, 0, ...)
- **Dimension 1**: alternates every 2 positions (0, 0, 1, 1, 0, 0, 1, 1, ...)
- **Dimension 2**: alternates every 4 positions (0, 0, 0, 0, 1, 1, 1, 1, ...)
- **Dimension 3** (MSB): alternates every 8 positions

Each dimension captures position information at a different frequency!

---

### 3.4 Mathematical Properties

**Property 1: Uniqueness**

Each position has a unique binary encoding (up to $2^d$ positions).

**Proof**: Binary representation is bijective for integers in $[0, 2^d - 1]$.

**Property 2: Bounded**

$$\text{PE}_{\text{bin}}(\text{pos}, i) \in \{0, 1\}$$

All values are in discrete binary space.

**Property 3: Hamming Distance**

The Hamming distance between encodings of positions $p_1$ and $p_2$ equals:

$$H(p_1, p_2) = \text{popcount}(p_1 \oplus p_2)$$

where $\oplus$ is XOR and popcount counts the number of 1s.

**Example**:
```
Position 3: [1, 1, 0, 0]
Position 5: [1, 0, 1, 0]
XOR:        [0, 1, 1, 0]
Hamming distance: 2
```

Positions that are close numerically don't always have small Hamming distance!

**Property 4: Maximum Length**

Can represent positions up to $2^d - 1$.

For $d = 512$ dimensions: maximum position = $2^{512} - 1$ (astronomically large!)

---

### 3.5 Normalized Binary Encoding

**Problem**: Binary values {0, 1} may not be ideal for neural networks.

**Solution 1: Map to [-1, +1]**

$$\text{PE}_{\text{bin-norm}}(\text{pos}, i) = 2 \cdot b_i - 1$$

where $b_i \in \{0, 1\}$ is the binary bit.

**Example**:
```
Binary:     [1, 0, 1, 1]
Normalized: [+1, -1, +1, +1]
```

**Solution 2: Map to [0, 1] with smoothing**

$$\text{PE}_{\text{bin-smooth}}(\text{pos}, i) = \sigma(c \cdot (2b_i - 1))$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is sigmoid
- $c$ is temperature parameter (controls smoothness)

**Properties**:
- $c \to \infty$: approaches hard binary {0, 1}
- $c \to 0$: approaches uniform 0.5
- Typical: $c = 5$ gives smooth but distinct values

**Example** ($c = 5$):
```
Binary bit 0: σ(5×(-1)) = σ(-5) = 0.007
Binary bit 1: σ(5×(+1)) = σ(+5) = 0.993
```

---

### 3.6 Gray Code Encoding

**Problem with standard binary**: Adjacent positions can have large Hamming distance.

**Example**:
```
Position 3: 0011 (binary)
Position 4: 0100 (binary)
Hamming distance: 3 (all bits differ!)
```

**Solution: Gray Code**

Gray code ensures adjacent positions differ by exactly 1 bit.

**Gray code formula**:

$$\text{gray}(\text{pos}) = \text{pos} \oplus \left\lfloor \frac{\text{pos}}{2} \right\rfloor$$

**Example**:

```
Standard Binary    Gray Code
Position  0: 0000      0000
Position  1: 0001      0001  (differ by 1 bit)
Position  2: 0010      0011  (differ by 1 bit)
Position  3: 0011      0010  (differ by 1 bit)
Position  4: 0100      0110  (differ by 1 bit)
Position  5: 0101      0111  (differ by 1 bit)
Position  6: 0110      0101  (differ by 1 bit)
Position  7: 0111      0100  (differ by 1 bit)
Position  8: 1000      1100  (differ by 1 bit)
```

**Advantage**: Adjacent positions always have Hamming distance = 1.

**Gray code positional encoding**:

$$\text{PE}_{\text{gray}}(\text{pos}, i) = \text{bit}_i(\text{gray}(\text{pos}))$$

---

### 3.7 Multi-Level Binary Encoding

**Idea**: Use multiple binary representations at different scales.

**Formulation**:

For dimension group $g$ and position $\text{pos}$:

$$\text{pos}_g = \left\lfloor \frac{\text{pos}}{2^g} \right\rfloor$$

$$\text{PE}_{\text{multi-bin}}(\text{pos}, g \cdot k + i) = \text{bit}_i(\text{pos}_g)$$

**Example**:

Position: 13
Dimension: 12 (3 groups of 4 bits each)

```
Group 0 (scale 1):  pos_0 = 13 = 1101
Group 1 (scale 2):  pos_1 = 6  = 0110
Group 2 (scale 4):  pos_2 = 3  = 0011

PE(13) = [1,1,0,1, 0,1,1,0, 1,1,0,0]
          └─group 0─┘ └─group 1─┘ └─group 2─┘
```

**Advantage**: Captures position at multiple resolutions simultaneously.

---

### 3.8 Advantages and Disadvantages

**Advantages**:

1. **Multi-scale representation**: Different bits = different frequencies
2. **Deterministic**: No randomness or learning required
3. **Efficient**: Simple bit operations
4. **Unique**: Every position has distinct encoding (up to $2^d$)
5. **Structured**: Clear mathematical properties
6. **Extrapolation**: Can handle sequences longer than training (up to $2^d$)

**Disadvantages**:

1. **Discrete**: Hard 0/1 values may be suboptimal for continuous optimization
2. **Hamming distance issue**: Adjacent positions may differ in many bits
3. **Non-smooth**: Sudden changes at power-of-2 boundaries
4. **Maximum length**: Limited to $2^d$ positions (but usually not a problem)
5. **Less interpretable**: Than simple integer encoding

---

## Comparison with Other Methods

### 4.1 Feature Comparison Table

| Feature | Integer | Binary | Sinusoidal | Learned | RoPE |
|---------|---------|--------|------------|---------|------|
| **Uniqueness** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Bounded** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-scale** | ❌ | ✅ | ✅ | ⚠️ | ✅ |
| **Smooth** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Extrapolation** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Parameters** | 0 | 0 | 0 | O(L×d) | 0 |
| **Interpretability** | ✅✅ | ✅ | ⚠️ | ❌ | ⚠️ |

---

### 4.2 Mathematical Comparison

**Integer Encoding**:
$$\text{PE}_{\text{int}}(\text{pos}) = \frac{\text{pos}}{L-1} \cdot \mathbf{1}_d$$

- **Simple**: Single value repeated
- **Linear**: Direct proportionality to position
- **Uniform**: Same across all dimensions

**Binary Encoding**:
$$\text{PE}_{\text{bin}}(\text{pos}, i) = \left\lfloor \frac{\text{pos}}{2^i} \right\rfloor \bmod 2$$

- **Multi-scale**: Different frequencies per dimension
- **Discrete**: Binary values {0, 1}
- **Exponential**: Powers of 2

**Sinusoidal Encoding**:
$$\text{PE}_{\text{sin}}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

- **Multi-scale**: Different frequencies per dimension
- **Continuous**: Smooth values in [-1, 1]
- **Periodic**: Repeating patterns

---

### 4.3 Visualization Example

Position = 10, Dimension = 8

**Integer** (normalized):
```
[0.909, 0.909, 0.909, 0.909, 0.909, 0.909, 0.909, 0.909]
```

**Binary** (position 10 = 1010 in binary):
```
[0, 1, 0, 1, 0, 0, 0, 0]
```

**Sinusoidal** (approximate):
```
[-0.544, 0.839, 0.010, 1.000, 0.000, 1.000, 0.000, 1.000]
```

---

### 4.4 When to Use Each

**Use Integer Encoding when**:
- Simplicity is paramount
- You want maximum interpretability
- Position is naturally scalar (e.g., time series)
- Short sequences (< 100 tokens)

**Use Binary Encoding when**:
- You want multi-scale without computation
- Memory is limited (discrete values)
- Maximum sequence length known and reasonable
- You can tolerate discrete representations

**Use Sinusoidal Encoding when**:
- Smooth gradients are important
- You need unlimited sequence length
- Multi-scale representation crucial
- Standard transformer architecture

**Use Learned Encoding when**:
- You have lots of data
- Fixed maximum sequence length
- Want model to determine best encoding
- Task-specific position patterns

---

## Advanced Variants

### 5.1 Hybrid Integer-Binary Encoding

**Idea**: Combine smooth integer with discrete binary.

$$\text{PE}_{\text{hybrid}}(\text{pos}, i) = \begin{cases}
\frac{\text{pos}}{L-1} & \text{if } i < d/2 \\
\text{bit}_{i-d/2}(\text{pos}) & \text{if } i \geq d/2
\end{cases}$$

**Example** ($d = 8$, $\text{pos} = 5$, $L = 10$):

First 4 dimensions (integer):
```
[5/9, 5/9, 5/9, 5/9] = [0.556, 0.556, 0.556, 0.556]
```

Last 4 dimensions (binary, 5 = 0101):
```
[1, 0, 1, 0]
```

Combined:
```
[0.556, 0.556, 0.556, 0.556, 1, 0, 1, 0]
```

**Advantage**: Smooth + discrete information.

---

### 5.2 Fourier-Binary Encoding

**Idea**: Apply Fourier transform to binary encoding.

$$\text{PE}_{\text{fourier-bin}}(\text{pos}, i) = \sum_{j=0}^{d-1} b_j \cdot e^{2\pi i j / d}$$

where $b_j$ are binary bits.

**Properties**:
- Complex-valued encoding
- Smooth in frequency domain
- Preserves uniqueness

---

### 5.3 Learned Binary Gates

**Idea**: Use binary encoding with learned gates.

$$\text{PE}_{\text{gated}}(\text{pos}, i) = g_i \cdot b_i(\text{pos}) + (1 - g_i) \cdot \frac{\text{pos}}{L-1}$$

where:
- $g_i \in [0, 1]$ is learned gate for dimension $i$
- $b_i(\text{pos})$ is binary bit
- Linear interpolation between binary and integer

**Training**: Gates learned via backpropagation.

---

### 5.4 Hierarchical Binary Encoding

**Idea**: Encode position at multiple hierarchical levels.

For a sentence with words, subwords, and characters:

$$\text{PE}_{\text{hier}}(\text{pos}_{\text{word}}, \text{pos}_{\text{subword}}, \text{pos}_{\text{char}}) = [\text{PE}_{\text{bin}}(\text{pos}_{\text{word}}) \,|\, \text{PE}_{\text{bin}}(\text{pos}_{\text{subword}}) \,|\, \text{PE}_{\text{bin}}(\text{pos}_{\text{char}})]$$

**Example**:

"hello" is:
- Word position 0
- Subword position 0 
- Character positions 0,1,2,3,4

```
Character 'h': [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0]
Character 'e': [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0,0]
Character 'l': [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0,0]
...
```

**Advantage**: Captures structure at multiple granularities.

---

## Practical Applications

### 6.1 Time Series Forecasting

**Integer encoding** is natural for time indices.

$$\text{PE}_{\text{time}}(t) = \frac{t - t_{\min}}{t_{\max} - t_{\min}}$$

**Example**: Stock prices
```
Day 0:  PE = 0.0
Day 1:  PE = 0.01
...
Day 99: PE = 0.99
```

---

### 6.2 Image Positional Encoding

For 2D positions $(x, y)$ in image:

**Binary 2D encoding**:

$$\text{PE}_{\text{2D}}(x, y, i) = \begin{cases}
\text{bit}_{i/2}(x) & \text{if } i \text{ is even} \\
\text{bit}_{(i-1)/2}(y) & \text{if } i \text{ is odd}
\end{cases}$$

**Example**: Position (5, 3) with $d = 8$

$x = 5 = 0101$ (binary)
$y = 3 = 0011$ (binary)

```
PE(5, 3) = [0,0, 1,0, 0,1, 1,1]
            ↑ ↑  ↑ ↑  ↑ ↑  ↑ ↑
            x y  x y  x y  x y
            bit bit bit bit
            0 0  1 1  2 2  3 3
```

---

### 6.3 Graph Neural Networks

**Node positional encoding** using binary:

For node at distance $d$ from root:

$$\text{PE}_{\text{graph}}(d) = \text{binary}(d)$$

Encodes tree depth or graph distance.

---

### 6.4 Document Structure Encoding

Hierarchical documents (section, paragraph, sentence):

$$\text{PE}_{\text{doc}}(\text{sec}, \text{par}, \text{sent}) = \text{concat}(\text{bin}(\text{sec}), \text{bin}(\text{par}), \text{bin}(\text{sent}))$$

**Example**: Section 2, Paragraph 3, Sentence 5

```
Section 2:   [0, 1]      (2 = 10 binary)
Paragraph 3: [1, 1]      (3 = 11 binary)
Sentence 5:  [1, 0, 1]   (5 = 101 binary)

Combined: [0, 1, 1, 1, 1, 0, 1]
```

---

## Implementation Considerations

### 7.1 Dimension Requirements

**Integer encoding**: Any dimension $d$ works.

**Binary encoding**: 
- Minimum: $d \geq \lceil \log_2(L) \rceil$
- For sequence length $L = 512$: need $d \geq 9$ bits
- Typical transformer ($d = 512$): can encode up to $2^{512}$ positions!

**Practical recommendation**: Use binary for first $k$ dimensions, other encodings for rest.

---

### 7.2 Numerical Stability

**Integer encoding**: 
- Always normalize to [0, 1]
- Avoid large unnormalized values

**Binary encoding**:
- Consider smoothing hard 0/1 values
- Use temperature-scaled sigmoid for gradients

---

### 7.3 Gradient Flow

**Integer encoding**: 
- ✅ Smooth gradients
- ✅ Easy to optimize

**Binary encoding**:
- ⚠️ Discrete values may hinder gradient flow
- Solution: Use smooth approximations during training
- Use straight-through estimator for backprop

---

### 7.4 Combining with Token Embeddings

**Addition** (most common):

$$\mathbf{x}_{\text{pos}} = \mathbf{x}_{\text{token}} + \mathbf{x}_{\text{PE}}$$

**Concatenation**:

$$\mathbf{x}_{\text{pos}} = [\mathbf{x}_{\text{token}} \,|\, \mathbf{x}_{\text{PE}}]$$

**Multiplication** (less common):

$$\mathbf{x}_{\text{pos}} = \mathbf{x}_{\text{token}} \odot (1 + \mathbf{x}_{\text{PE}})$$

---

### 7.5 Computational Complexity

**Integer Encoding**:
- Time: $O(d)$ per position
- Space: $O(d)$ per position
- Very fast: just arithmetic

**Binary Encoding**:
- Time: $O(d)$ per position
- Space: $O(d)$ per position  
- Very fast: bit operations

**Comparison to Sinusoidal**:
- Sinusoidal: Requires trigonometric functions (slower)
- Binary: Just bit shifts and masks (faster)
- Integer: Just division (fastest)

---

## Key Takeaways

### Summary Table

| Aspect | Integer | Binary |
|--------|---------|--------|
| **Complexity** | Simplest | Simple |
| **Expressiveness** | Low | Medium |
| **Multi-scale** | No | Yes |
| **Smoothness** | Yes | No |
| **Max length** | Fixed | $2^d$ |
| **Interpretability** | Highest | High |
| **Computation** | Fastest | Very fast |
| **Use case** | Time series, simple tasks | Moderate tasks, discrete preference |

---

### Best Practices

1. **Start simple**: Try integer encoding first
2. **Add complexity**: Move to binary if needed
3. **Consider hybrids**: Combine multiple methods
4. **Normalize**: Always scale to reasonable range
5. **Smooth if discrete**: Use temperature scaling for binary
6. **Benchmark**: Compare empirically for your task

---

### Research Directions

1. **Learnable hybrid encodings**: Combine multiple types
2. **Task-adaptive selection**: Learn which encoding to use
3. **Dynamic encodings**: Change based on sequence properties
4. **Multi-modal encodings**: Different modalities, different encodings

---

## References

- Vaswani et al. (2017). "Attention Is All You Need" - Introduced sinusoidal PE
- While integer and binary encodings are less studied in papers, they represent fundamental mathematical principles
- Binary encoding relates to Walsh-Hadamard transforms and Fourier analysis
- Gray code from Gray, F. (1953). "Pulse Code Communication"
- Multi-scale binary relates to wavelets and multi-resolution analysis

---

## Mathematical Appendix

### A.1 Binary Operations Reference

**Integer to binary**:
$$b_i = \left\lfloor \frac{n}{2^i} \right\rfloor \bmod 2$$

**Gray code**:
$$\text{gray}(n) = n \oplus \left\lfloor \frac{n}{2} \right\rfloor$$

**Hamming distance**:
$$H(a, b) = \text{popcount}(a \oplus b)$$

**Population count (number of 1 bits)**:
$$\text{popcount}(n) = \sum_{i=0}^{\lceil \log_2 n \rceil} \left(\left\lfloor \frac{n}{2^i} \right\rfloor \bmod 2\right)$$

---

### A.2 Frequency Analysis

Binary encoding creates square waves with frequencies:

$$f_i = \frac{1}{2^{i+1}}$$

- Dimension 0: frequency = 1/2 (alternates every position)
- Dimension 1: frequency = 1/4 (alternates every 2 positions)
- Dimension 2: frequency = 1/8 (alternates every 4 positions)

This mirrors the frequency decomposition in sinusoidal encoding!

---

*This document provides a complete mathematical treatment of integer and binary positional encodings. For implementation, see the accompanying code file.*
