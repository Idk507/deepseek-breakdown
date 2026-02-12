# Rotary Positional Encoding (RoPE): Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concept and Intuition](#core-concept-and-intuition)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Detailed Mathematical Derivations](#detailed-mathematical-derivations)
5. [Properties and Proofs](#properties-and-proofs)
6. [Implementation Details](#implementation-details)
7. [Comparison with Other Methods](#comparison-with-other-methods)
8. [Applications and Variations](#applications-and-variations)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

### What is RoPE?

**Rotary Positional Encoding (RoPE)** is a method for encoding positional information in Transformers by applying a rotation to the query and key vectors in the self-attention mechanism.

**Proposed by**: Su et al. (2021) in "RoFormer: Enhanced Transformer with Rotary Position Embedding"

**Used by**: 
- LLaMA (Meta)
- PaLM (Google)
- GPT-NeoX (EleutherAI)
- Mistral
- Qwen
- Many other modern LLMs

### Why RoPE?

**Traditional problem**: How to encode position such that:
1. ✅ Preserves semantic meaning
2. ✅ Encodes relative distances
3. ✅ Generalizes to unseen sequence lengths
4. ✅ Doesn't require additional parameters

**RoPE's solution**: Encode position by **rotating** embeddings in 2D subspaces.

---

## Core Concept and Intuition

### The Rotation Metaphor

Imagine embedding vectors as points in 2D space. To encode position:
- **Position 0**: No rotation (0°)
- **Position 1**: Rotate by θ (e.g., 30°)
- **Position 2**: Rotate by 2θ (60°)
- **Position 3**: Rotate by 3θ (90°)

Each position gets a unique angle of rotation!

### Visual Intuition (2D Example)

```
Original vector: x = [x₀, x₁]

Position 0:  [x₀, x₁]                    (0° rotation)
Position 1:  [x₀cos(θ)-x₁sin(θ), ...]   (θ rotation)
Position 2:  [x₀cos(2θ)-x₁sin(2θ), ...] (2θ rotation)
Position 3:  [x₀cos(3θ)-x₁sin(3θ), ...] (3θ rotation)
```

### Key Insight: Dot Product of Rotations

For vectors rotated by different amounts, their dot product depends only on the **relative angle difference**:

$$\text{Rot}(m\theta) \cdot \text{Rot}(n\theta) = \text{Rot}((n-m)\theta) = f(n-m)$$

This is exactly what we want for relative position encoding!

---

## Mathematical Formulation

### Rotation Matrix in 2D

A rotation by angle $\theta$ in 2D:

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Applied to vector $\mathbf{x} = [x_0, x_1]^T$:

$$R(\theta) \mathbf{x} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix} = \begin{pmatrix} x_0\cos\theta - x_1\sin\theta \\ x_0\sin\theta + x_1\cos\theta \end{pmatrix}$$

### Extension to Higher Dimensions

For $d$-dimensional space (where $d$ is even), treat as $d/2$ independent 2D rotations:

$$\mathbf{x} = [x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1}]$$

Pair up dimensions:
- Pair 0: $(x_0, x_1)$ → rotate by $\theta_0$
- Pair 1: $(x_2, x_3)$ → rotate by $\theta_1$
- ...
- Pair $d/2-1$: $(x_{d-2}, x_{d-1})$ → rotate by $\theta_{d/2-1}$

### Rotation Matrix in d-Dimensions

$$\mathbf{R}_{\Theta} = \begin{pmatrix}
R(\theta_0) & 0 & 0 & \cdots & 0 \\
0 & R(\theta_1) & 0 & \cdots & 0 \\
0 & 0 & R(\theta_2) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R(\theta_{d/2-1})
\end{pmatrix}$$

where each $R(\theta_i)$ is a 2×2 rotation matrix.

### Position-Dependent Rotation

For position $m$ in the sequence:

$$\mathbf{R}_{\Theta, m} = \begin{pmatrix}
R(m\theta_0) & 0 & \cdots & 0 \\
0 & R(m\theta_1) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & R(m\theta_{d/2-1})
\end{pmatrix}$$

Rotation angle for each pair scales with position $m$.

### Frequency Selection

The rotation frequencies are chosen similarly to sinusoidal encoding:

$$\theta_i = \frac{1}{10000^{2i/d}} = 10000^{-2i/d}$$

where $i \in \{0, 1, 2, ..., d/2-1\}$.

**This gives**:
- Lower dimension pairs: Higher frequency (rotate more per position)
- Higher dimension pairs: Lower frequency (rotate less per position)

### Application to Queries and Keys

For a query at position $m$:

$$\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m$$

For a key at position $n$:

$$\tilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n} \mathbf{k}_n$$

**Attention score** between query at $m$ and key at $n$:

$$\text{score}(m, n) = \tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n$$

---

## Detailed Mathematical Derivations

### Derivation 1: Why Rotation Encodes Relative Position

**Goal**: Show that attention score depends only on $n - m$.

**Starting point**:

$$\text{score}(m, n) = \tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n = (\mathbf{R}_{\Theta, m} \mathbf{q}_m)^T (\mathbf{R}_{\Theta, n} \mathbf{k}_n)$$

**Step 1**: Transpose property

$$= \mathbf{q}_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \mathbf{k}_n$$

**Step 2**: Rotation matrix property

Recall that for rotation matrices:

$$\mathbf{R}^T(\theta) = \mathbf{R}(-\theta) = \mathbf{R}(\theta)^{-1}$$

So:

$$\mathbf{R}_{\Theta, m}^T = \mathbf{R}_{\Theta, -m}$$

**Step 3**: Composition of rotations

$$\mathbf{R}_{\Theta, -m} \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, n-m}$$

**Proof for 2D case**:

$$R(-m\theta) R(n\theta) = \begin{pmatrix} \cos(-m\theta) & -\sin(-m\theta) \\ \sin(-m\theta) & \cos(-m\theta) \end{pmatrix} \begin{pmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{pmatrix}$$

Using angle addition formulas:

$$\cos(-m\theta)\cos(n\theta) - \sin(-m\theta)\sin(n\theta) = \cos(-m\theta + n\theta) = \cos((n-m)\theta)$$

Similarly for other elements, we get:

$$R(-m\theta) R(n\theta) = R((n-m)\theta)$$

**Step 4**: Final result

$$\text{score}(m, n) = \mathbf{q}_m^T \mathbf{R}_{\Theta, n-m} \mathbf{k}_n$$

**The attention score depends only on the relative position $(n-m)$!**

### Derivation 2: Explicit Element-Wise Formula

For dimension pair $(2i, 2i+1)$ at position $m$:

**Original values**: $x_{2i}, x_{2i+1}$

**After rotation**:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

**Expanding**:

$$x'_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)$$

$$x'_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)$$

**For all pairs**:

For $j = 0, 1, 2, ..., d/2-1$:

$$x'_{2j} = x_{2j} \cos(m\theta_j) - x_{2j+1} \sin(m\theta_j)$$

$$x'_{2j+1} = x_{2j} \sin(m\theta_j) + x_{2j+1} \cos(m\theta_j)$$

### Derivation 3: Complex Number Representation

RoPE can be elegantly expressed using complex numbers.

**Represent each dimension pair as complex number**:

$$z_j = x_{2j} + i \cdot x_{2j+1}$$

**Rotation by angle $\theta$ in complex plane**:

$$z'_j = z_j \cdot e^{i\theta_j} = z_j \cdot (\cos\theta_j + i\sin\theta_j)$$

**For position $m$**:

$$z'_j = z_j \cdot e^{im\theta_j}$$

**Expanding back to real coordinates**:

$$z'_j = (x_{2j} + ix_{2j+1})(\cos(m\theta_j) + i\sin(m\theta_j))$$

$$= x_{2j}\cos(m\theta_j) - x_{2j+1}\sin(m\theta_j) + i(x_{2j}\sin(m\theta_j) + x_{2j+1}\cos(m\theta_j))$$

Real part: $x'_{2j} = x_{2j}\cos(m\theta_j) - x_{2j+1}\sin(m\theta_j)$

Imaginary part: $x'_{2j+1} = x_{2j}\sin(m\theta_j) + x_{2j+1}\cos(m\theta_j)$

Same as rotation matrix form!

---

## Properties and Proofs

### Property 1: Relative Position Encoding

**Statement**: The inner product of rotated vectors depends only on their relative position difference.

**Proof**: Already shown in Derivation 1.

$$\langle \mathbf{R}_{\Theta,m}\mathbf{q}, \mathbf{R}_{\Theta,n}\mathbf{k} \rangle = \langle \mathbf{q}, \mathbf{R}_{\Theta,n-m}\mathbf{k} \rangle$$

### Property 2: Boundedness

**Statement**: Rotated vectors have the same norm as original vectors.

**Proof**: Rotation matrices are orthogonal, so they preserve norms.

$$\|\mathbf{R}_{\Theta,m}\mathbf{x}\|^2 = (\mathbf{R}_{\Theta,m}\mathbf{x})^T(\mathbf{R}_{\Theta,m}\mathbf{x}) = \mathbf{x}^T \mathbf{R}_{\Theta,m}^T \mathbf{R}_{\Theta,m} \mathbf{x}$$

Since $\mathbf{R}_{\Theta,m}^T \mathbf{R}_{\Theta,m} = \mathbf{I}$:

$$= \mathbf{x}^T \mathbf{x} = \|\mathbf{x}\|^2$$

**Implication**: RoPE doesn't amplify or diminish the magnitude of embeddings.

### Property 3: Uniqueness

**Statement**: Different positions produce different rotation patterns (up to periodicity).

**Proof**: For positions $m \neq n$:

$$\mathbf{R}_{\Theta,m} \neq \mathbf{R}_{\Theta,n}$$

unless $m - n$ is a multiple of the period.

**Period for dimension pair $j$**:

$$T_j = \frac{2\pi}{\theta_j} = 2\pi \cdot 10000^{2j/d}$$

- Dimension 0: Period ≈ 6.28
- Dimension $d/2-1$: Period ≈ 62,832

Different dimension pairs have different periods, making collisions unlikely.

### Property 4: Linear Complexity

**Statement**: Can be computed in $O(d)$ time per position.

**Proof**: For each dimension pair, we perform:
- 2 multiplications (for cos term)
- 2 multiplications (for sin term)
- 2 additions/subtractions

Total: 6 operations per pair × $d/2$ pairs = $3d$ operations.

### Property 5: Long-Range Decay

**Statement**: Attention between distant tokens naturally decays for high-frequency dimensions.

**Mathematical insight**:

For dimension pair $j$ with frequency $\theta_j$:

$$\text{similarity}(\text{pos } m, \text{pos } n) \propto \cos((n-m)\theta_j)$$

- If $|n-m|$ is large and $\theta_j$ is large (high frequency):
  - $(n-m)\theta_j$ cycles through many periods
  - Average similarity approaches 0

- If $|n-m|$ is large and $\theta_j$ is small (low frequency):
  - $(n-m)\theta_j$ is still relatively small
  - Similarity remains high

**This creates multi-scale attention naturally!**

---

## Implementation Details

### Step-by-Step Algorithm

**Input**: 
- Query/Key embeddings: $\mathbf{x} \in \mathbb{R}^{d}$
- Position: $m$
- Frequencies: $\theta_0, \theta_1, ..., \theta_{d/2-1}$

**Step 1: Precompute sin and cos values**

For position $m$ and all frequencies:

```
cos_values[j] = cos(m × θ_j)  for j = 0, 1, ..., d/2-1
sin_values[j] = sin(m × θ_j)  for j = 0, 1, ..., d/2-1
```

**Step 2: Reshape embeddings into pairs**

```
x_pairs = reshape(x, [d/2, 2])
# x_pairs[j] = [x_{2j}, x_{2j+1}]
```

**Step 3: Apply rotation to each pair**

For each pair $j$:

```
x0 = x_pairs[j, 0]
x1 = x_pairs[j, 1]

x0_new = x0 × cos_values[j] - x1 × sin_values[j]
x1_new = x0 × sin_values[j] + x1 × cos_values[j]

x_pairs[j] = [x0_new, x1_new]
```

**Step 4: Reshape back to original shape**

```
x_rotated = reshape(x_pairs, [d])
```

### Efficient Batch Implementation

**For batch of sequences**:

Input shape: `(batch, seq_len, d_model)`

```python
# Precompute all positions
positions = np.arange(seq_len)  # (seq_len,)

# Compute all frequencies
freqs = positions[:, None] * theta[None, :]  # (seq_len, d/2)

# Compute sin and cos
cos_vals = np.cos(freqs)  # (seq_len, d/2)
sin_vals = np.sin(freqs)  # (seq_len, d/2)

# Reshape inputs
x_pairs = x.reshape(batch, seq_len, d//2, 2)  # (batch, seq_len, d/2, 2)

# Extract even and odd dimensions
x_even = x_pairs[..., 0]  # (batch, seq_len, d/2)
x_odd = x_pairs[..., 1]   # (batch, seq_len, d/2)

# Apply rotation
x_even_rot = x_even * cos_vals - x_odd * sin_vals
x_odd_rot = x_even * sin_vals + x_odd * cos_vals

# Combine back
x_rotated = np.stack([x_even_rot, x_odd_rot], axis=-1)
x_rotated = x_rotated.reshape(batch, seq_len, d)
```

### Frequency Computation

$$\theta_j = 10000^{-2j/d}$$

**Numerical stability consideration**:

Instead of computing as written, use:

$$\theta_j = \exp\left(-\frac{2j}{d} \ln(10000)\right)$$

```python
inv_freq = np.exp(-np.arange(0, d, 2) * (np.log(10000.0) / d))
```

This avoids potential overflow/underflow for large $d$.

### Caching for Autoregressive Generation

During autoregressive generation, we generate one token at a time.

**Without caching**: Recompute rotations for all previous positions.

**With caching**: 
1. Store previous rotated keys and values
2. Only compute rotation for new position
3. Concatenate with cached values

```python
# First call (position 0 to L-1)
rotated_k = apply_rope(k, positions=range(L))
cache_k = rotated_k

# Generation step (position L)
new_k = apply_rope(k_new, positions=[L])
cache_k = concatenate([cache_k, new_k], axis=1)
```

---

## Comparison with Other Methods

### RoPE vs. Sinusoidal

**Similarities**:
- Both use same frequency base (10000)
- Both have multi-scale properties
- Both are deterministic (no learned parameters)

**Differences**:

| Aspect | Sinusoidal | RoPE |
|--------|-----------|------|
| **Application** | Added to embeddings | Applied via rotation |
| **Affects** | All vectors equally | Only Q and K |
| **Position type** | Absolute | Relative |
| **Mechanism** | Addition | Matrix multiplication |
| **Dot product** | Doesn't encode relative | Encodes relative naturally |

**Mathematical difference**:

Sinusoidal adds position:
$$\mathbf{q}_m' = \mathbf{q}_m + \text{PE}(m)$$

RoPE rotates:
$$\mathbf{q}_m' = \mathbf{R}_{\Theta,m} \mathbf{q}_m$$

### RoPE vs. Learned Positions

| Aspect | Learned | RoPE |
|--------|---------|------|
| **Parameters** | $O(L \times d)$ | 0 |
| **Max length** | Fixed $L$ | Unlimited |
| **Extrapolation** | Poor | Excellent |
| **Training time** | Slower (more params) | Faster |
| **Task adaptation** | Yes | No |

### RoPE vs. ALiBi

| Aspect | ALiBi | RoPE |
|--------|-------|------|
| **Mechanism** | Attention bias | Embedding rotation |
| **Affects** | Attention scores | Q and K vectors |
| **Extrapolation** | Excellent | Excellent |
| **Direction** | Symmetric | Can encode direction |
| **Complexity** | $O(L^2)$ per head | $O(Ld)$ total |

**Performance comparison** (perplexity on language modeling):

```
Sequence Length    | Learned | RoPE  | ALiBi
-------------------|---------|-------|-------
512 (train)        |   23.5  | 23.2  | 23.4
1024 (test)        |   24.1  | 23.8  | 23.6
2048 (extrapolate) |   31.2  | 24.5  | 23.9
4096 (extrapolate) |   45.8  | 25.8  | 24.2
```

RoPE: Strong extrapolation, second best
ALiBi: Best extrapolation overall

---

## Applications and Variations

### 1. Standard RoPE (RoFormer)

Original implementation as described above.

**Used in**: RoFormer, GPT-NeoX

### 2. RoPE with Varying Base

Some models use different base values instead of 10000.

**Base = 500,000** (CodeGen):
- Larger base → longer wavelengths
- Better for very long sequences
- Used in code generation

**Formula**:
$$\theta_j = B^{-2j/d}$$

where $B$ is the base (e.g., 500,000).

### 3. Partial RoPE

Only apply RoPE to a fraction of dimensions.

**Example (LLaMA)**:
- Apply RoPE to first 25% of dimensions
- Keep remaining 75% without rotation

**Motivation**: 
- Preserve some non-positional information
- Allow model to learn position-independent patterns

**Implementation**:
```python
d_rope = d // 4  # 25% of dimensions

# Apply RoPE to first d_rope dimensions
x[:, :d_rope] = apply_rope(x[:, :d_rope])

# Keep remaining dimensions unchanged
# x[:, d_rope:] stays the same
```

### 4. RoPE Scaling

For handling sequences longer than training length.

**Linear scaling** (simple):
$$\theta_j' = \theta_j / s$$

where $s > 1$ is scaling factor.

**Example**: If trained on length 2048, use $s=2$ for length 4096.

**NTK-aware scaling** (better):

$$\theta_j' = \theta_j \cdot s^{-2j/d}$$

Different frequencies scale differently.

### 5. RoPE for 2D Positions (Vision)

For images, positions are 2D: $(x, y)$.

**Separate RoPE for each dimension**:

Split $d$ into two groups:
- First $d/2$ dimensions: encode $x$ position
- Last $d/2$ dimensions: encode $y$ position

$$\mathbf{R}_{\Theta}(x, y) = \begin{pmatrix} \mathbf{R}_{\Theta_x}(x) & 0 \\ 0 & \mathbf{R}_{\Theta_y}(y) \end{pmatrix}$$

### 6. RoPE in Multi-Query Attention

RoPE works perfectly with MQA!

**Standard MHA**: Apply RoPE to all Q and K heads independently.

**MQA**: 
- Apply RoPE to all Q heads (different rotations)
- Apply RoPE to single shared K (one rotation)

**Memory savings maintained**: KV cache still $h$ times smaller.

---

## Advanced Topics

### Mathematical Analysis: Why Does It Work?

**Theorem**: RoPE creates a valid kernel for measuring similarity that incorporates relative position.

**Proof sketch**:

Define similarity kernel:
$$K(m, n, \mathbf{q}, \mathbf{k}) = \langle \mathbf{R}_{\Theta,m}\mathbf{q}, \mathbf{R}_{\Theta,n}\mathbf{k} \rangle$$

This can be written as:
$$K(m, n, \mathbf{q}, \mathbf{k}) = \sum_{j=0}^{d/2-1} \text{Re}(q_j^* k_j e^{i(n-m)\theta_j})$$

where $q_j, k_j$ are complex representations of dimension pairs.

This is a **positive definite kernel** combining:
1. Content similarity: $q_j^* k_j$
2. Position similarity: $e^{i(n-m)\theta_j}$

### Fourier Analysis Perspective

RoPE can be viewed as applying position-dependent phase shifts in Fourier space.

**Each dimension pair** acts as:
- A complex exponential at frequency $\theta_j$
- Phase advances by $m\theta_j$ at position $m$

**This is analogous to**:
- Modulation in signal processing
- Phase encoding in quantum mechanics

### Connection to Attention Weights

**Standard attention** (no position encoding):
$$A_{mn} \propto \exp(\mathbf{q}_m^T \mathbf{k}_n)$$

**With RoPE**:
$$A_{mn} \propto \exp((\mathbf{R}_{\Theta,m}\mathbf{q}_m)^T (\mathbf{R}_{\Theta,n}\mathbf{k}_n))$$
$$= \exp(\mathbf{q}_m^T \mathbf{R}_{\Theta,n-m} \mathbf{k}_n)$$

The rotation matrix $\mathbf{R}_{\Theta,n-m}$ modulates attention based on relative position!

### Limitations and Open Questions

**1. Fixed frequencies**:
- Cannot adapt to specific tasks
- May not be optimal for all sequence types

**2. Dimension requirements**:
- Requires even dimension
- Wastes capacity if $d$ is odd

**3. Computational overhead**:
- Slower than learned embeddings
- Requires trigonometric functions

**4. Theoretical questions**:
- Optimal choice of base frequency?
- Best dimension allocation for partial RoPE?
- Interaction with other architectural choices?

---

## Practical Recommendations

### When to Use RoPE

**✅ Use RoPE when**:
- Building autoregressive language models
- Long-context understanding is important
- Extrapolation beyond training length needed
- Using Multi-Query or Grouped-Query Attention

**❌ Consider alternatives when**:
- Very short sequences only (<128 tokens)
- Fixed maximum length is acceptable
- Computational budget is extremely tight

### Hyperparameter Choices

**Base frequency**:
- Standard: 10,000 (good default)
- Long sequences: 500,000 or higher
- Short sequences: 1,000 may suffice

**Partial RoPE ratio**:
- 25% (LLaMA): Good balance
- 50%: More position info
- 100%: Maximum position encoding

**Scaling for extrapolation**:
- Linear scaling: Simple, works reasonably
- NTK-aware: Better, more complex
- Start with linear, upgrade if needed

---

## Summary

### Key Equations

**1. Rotation for position $m$, dimension pair $j$**:

$$\begin{pmatrix} x'_{2j} \\ x'_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}$$

**2. Frequency**:

$$\theta_j = 10000^{-2j/d}$$

**3. Relative position property**:

$$\langle \mathbf{R}_{\Theta,m}\mathbf{q}, \mathbf{R}_{\Theta,n}\mathbf{k} \rangle = \langle \mathbf{q}, \mathbf{R}_{\Theta,n-m}\mathbf{k} \rangle$$

### Advantages

1. ✅ Encodes relative positions naturally
2. ✅ No additional parameters
3. ✅ Excellent extrapolation
4. ✅ Preserves vector norms
5. ✅ Multi-scale representation
6. ✅ Theoretically grounded

### Limitations

1. ❌ Requires even dimension
2. ❌ Computational overhead vs. learned
3. ❌ Fixed frequencies (not adaptive)

---

## References

1. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y.** (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv preprint arXiv:2104.09864*.

2. **Touvron, H., et al.** (2023). "LLaMA: Open and Efficient Foundation Language Models." *arXiv preprint arXiv:2302.13971*.

3. **Black, S., et al.** (2022). "GPT-NeoX-20B: An Open-Source Autoregressive Language Model." *arXiv preprint arXiv:2204.06745*.

4. **Press, O., Smith, N. A., & Lewis, M.** (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *arXiv preprint arXiv:2108.12409*.

---

*This document provides complete mathematical coverage of Rotary Positional Encoding. For implementation, see the accompanying code file.*
