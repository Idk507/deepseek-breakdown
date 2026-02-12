# Complete Guide to Positional Encodings: All Major Techniques

## Table of Contents
1. [Introduction](#introduction)
2. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
3. [Learned Positional Encoding](#learned-positional-encoding)
4. [Rotary Positional Embedding (RoPE)](#rotary-positional-embedding-rope)
5. [ALiBi (Attention with Linear Biases)](#alibi-attention-with-linear-biases)
6. [Integer Positional Encoding](#integer-positional-encoding)
7. [Binary Positional Encoding](#binary-positional-encoding)
8. [Relative Positional Encoding](#relative-positional-encoding)
9. [Advanced Techniques](#advanced-techniques)
10. [Comprehensive Comparison](#comprehensive-comparison)

---

## Introduction

### Why Positional Encoding?

**Problem**: Self-attention in Transformers is **permutation-invariant**.

For input sequences $X = [x_1, x_2, x_3]$ and $X' = [x_2, x_1, x_3]$:

$$\text{Attention}(X) = \text{Attention}(X')$$

But position matters:
- "Dog bites man" ≠ "Man bites dog"
- "I didn't say he stole the money" (7 different meanings based on emphasis)

**Solution**: Inject positional information into the input.

### Requirements for Good Positional Encoding

1. **Unique**: Each position has distinct encoding
2. **Bounded**: Values don't grow infinitely
3. **Generalizable**: Works for unseen sequence lengths
4. **Learnable**: Model can extract relative positions
5. **Efficient**: Fast to compute

---

## 1. Sinusoidal Positional Encoding

### 1.1 Introduction

**Proposed by**: Vaswani et al. (2017) in "Attention Is All You Need"

**Key Idea**: Use sine and cosine functions at different frequencies to create unique positional signatures.

### 1.2 Mathematical Formulation

For position $\text{pos}$ and dimension index $i$:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

where:
- $\text{pos} \in \{0, 1, 2, ..., L-1\}$ is position in sequence
- $i \in \{0, 1, ..., d_{\text{model}}/2 - 1\}$ is dimension index
- $d_{\text{model}}$ is embedding dimension
- Even dimensions use sine, odd dimensions use cosine

### 1.3 Intuition: Wavelengths and Frequencies

**Frequency for dimension $2i$**:

$$f_i = \frac{1}{10000^{2i/d_{\text{model}}}}$$

**Wavelength**:

$$\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$$

**Range of wavelengths**:
- Dimension 0: $\lambda_0 = 2\pi$ (period ≈ 6.28)
- Dimension $d_{\text{model}}-2$: $\lambda_{\max} = 2\pi \cdot 10000$ (period ≈ 62,832)

**Geometric progression**: Each dimension's frequency decreases by a constant factor.

$$\frac{f_i}{f_{i+1}} = 10000^{2/d_{\text{model}}}$$

For $d_{\text{model}} = 512$:
$$\frac{f_i}{f_{i+1}} = 10000^{2/512} \approx 1.046$$

### 1.4 Step-by-Step Example

**Setup**:
- Position: $\text{pos} = 0, 1, 2$
- Dimension: $d_{\text{model}} = 4$

**Dimension 0** (even, use sine):

$$i = 0, \quad \omega_0 = \frac{1}{10000^{0/4}} = 1$$

```
PE(0, 0) = sin(0 / 1) = sin(0) = 0.000
PE(1, 0) = sin(1 / 1) = sin(1) = 0.841
PE(2, 0) = sin(2 / 1) = sin(2) = 0.909
```

**Dimension 1** (odd, use cosine):

$$i = 0, \quad \omega_0 = 1$$

```
PE(0, 1) = cos(0 / 1) = cos(0) = 1.000
PE(1, 1) = cos(1 / 1) = cos(1) = 0.540
PE(2, 1) = cos(2 / 1) = cos(2) = -0.416
```

**Dimension 2** (even, use sine):

$$i = 1, \quad \omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{100} = 0.01$$

```
PE(0, 2) = sin(0 / 100) = sin(0) = 0.000
PE(1, 2) = sin(1 / 100) = sin(0.01) = 0.010
PE(2, 2) = sin(2 / 100) = sin(0.02) = 0.020
```

**Dimension 3** (odd, use cosine):

$$i = 1, \quad \omega_1 = 0.01$$

```
PE(0, 3) = cos(0 / 100) = cos(0) = 1.000
PE(1, 3) = cos(1 / 100) = cos(0.01) = 1.000
PE(2, 3) = cos(2 / 100) = cos(0.02) = 0.9998
```

**Complete encodings**:

```
Position 0: [0.000,  1.000,  0.000,  1.000]
Position 1: [0.841,  0.540,  0.010,  1.000]
Position 2: [0.909, -0.416,  0.020,  0.9998]
```

### 1.5 Key Properties

**Property 1: Bounded**

$$\text{PE}(\text{pos}, i) \in [-1, 1]$$

All values are bounded by sine and cosine range.

**Property 2: Unique Representation**

Each position creates a unique pattern across dimensions (like a "fingerprint").

**Property 3: Linear Transformation of Relative Positions**

For any fixed offset $k$:

$$\text{PE}(\text{pos} + k) = M_k \cdot \text{PE}(\text{pos})$$

where $M_k$ is a linear transformation matrix.

**Proof**: Using angle addition formulas:

$$\sin(\text{pos} + k) = \sin(\text{pos})\cos(k) + \cos(\text{pos})\sin(k)$$

This allows the model to learn to attend to relative positions!

**Property 4: Unlimited Length**

Can compute for any position (no maximum length constraint).

**Property 5: Deterministic**

No parameters to learn; same computation every time.

### 1.6 Why These Specific Frequencies?

**Geometric spacing** provides:
1. **Multi-scale representation**: Captures both local and global patterns
2. **Smooth interpolation**: Adjacent positions have similar encodings
3. **Extrapolation**: Can handle longer sequences than seen in training

**Choice of base 10000**:
- Large enough to handle long sequences (wavelength up to ~62K)
- Not too large (maintains numerical stability)
- Empirically validated in original Transformer paper

### 1.7 Advantages and Limitations

**Advantages**:
- ✅ No learned parameters (reduces overfitting)
- ✅ Deterministic and reproducible
- ✅ Unlimited sequence length
- ✅ Smooth and continuous
- ✅ Model can learn relative positions

**Limitations**:
- ❌ Fixed frequencies (cannot adapt to data)
- ❌ May not be optimal for all tasks
- ❌ Computational cost of sin/cos (though precomputable)

---

## 2. Learned Positional Encoding

### 2.1 Introduction

**Used by**: BERT, GPT-1, GPT-2

**Key Idea**: Treat positional encodings as learnable parameters, just like word embeddings.

### 2.2 Mathematical Formulation

**Position embedding matrix**:

$$\mathbf{P} \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$$

where:
- $L_{\max}$ is maximum sequence length
- Each row $\mathbf{P}[i, :]$ is the learned encoding for position $i$

**For position $\text{pos}$**:

$$\text{PE}_{\text{learned}}(\text{pos}) = \mathbf{P}[\text{pos}, :] \in \mathbb{R}^{d_{\text{model}}}$$

**Initialization**: Typically random (Xavier/Glorot or small random values).

$$\mathbf{P}[i, j] \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d_{\text{model}}}}\right)$$

**Training**: Updated via backpropagation like any other parameters.

$$\frac{\partial \mathcal{L}}{\partial \mathbf{P}[i, j]} = \frac{\partial \mathcal{L}}{\partial \text{input}[i, j]}$$

### 2.3 Example

**Configuration**:
- Maximum length: $L_{\max} = 512$
- Embedding dimension: $d_{\text{model}} = 768$ (BERT-base)

**Position embedding matrix**:

$$\mathbf{P} \in \mathbb{R}^{512 \times 768}$$

Total parameters: $512 \times 768 = 393,216$

**For a sequence** "The cat sat":

```
Position 0: P[0, :] = [0.023, -0.145, 0.089, ..., 0.234]  (768 dims)
Position 1: P[1, :] = [-0.112, 0.267, -0.045, ..., -0.156]
Position 2: P[2, :] = [0.198, -0.034, 0.123, ..., 0.078]
```

These values are learned during training!

### 2.4 Combining with Token Embeddings

**Token embeddings**: $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$

**For token at position $i$ with ID $t_i$**:

$$\mathbf{x}_i = \mathbf{E}[t_i, :] + \mathbf{P}[i, :]$$

**Example (BERT)**:

```
Token: "cat" (ID 2345)
Position: 1

Token embedding:    E[2345, :] = [0.5, -0.3, 0.8, ...]
Position embedding: P[1, :]    = [-0.1, 0.2, -0.05, ...]
Final embedding:    x_1        = [0.4, -0.1, 0.75, ...]
```

### 2.5 Advantages and Limitations

**Advantages**:
- ✅ Task-adaptive (learns optimal encoding for the task)
- ✅ Simple implementation
- ✅ Can capture complex positional patterns
- ✅ Empirically performs well

**Limitations**:
- ❌ Fixed maximum length $L_{\max}$
- ❌ Cannot extrapolate to longer sequences
- ❌ Additional parameters to learn
- ❌ May overfit on small datasets
- ❌ Requires training data

### 2.6 Variants

**2.6.1 Absolute Position Embeddings (APE)**

Standard learned embeddings as described above.

**2.6.2 Trainable Sinusoidal**

Initialize with sinusoidal pattern, then allow learning:

$$\mathbf{P}[i, j] = \text{PE}_{\text{sin}}(i, j) + \delta_{ij}$$

where $\delta_{ij}$ are learnable offsets.

**2.6.3 Conditional Position Embeddings**

Position embeddings that depend on content:

$$\mathbf{P}_{\text{cond}}(i, \mathbf{x}) = f(\mathbf{x}) \cdot \mathbf{P}[i, :]$$

where $f(\mathbf{x})$ is a learned function of the input.

---

## 3. Rotary Positional Embedding (RoPE)

### 3.1 Introduction

**Proposed by**: Su et al. (2021) in "RoFormer"

**Used by**: LLaMA, PaLM, GPT-NeoX, Mistral

**Key Innovation**: Encode position by **rotating** the query and key vectors.

### 3.2 Core Concept

Instead of adding positional information, **rotate** embeddings in 2D subspaces.

**For position $m$**, rotate each pair of dimensions by angle $m\theta$:

$$\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$$

### 3.3 Mathematical Formulation

**Rotation matrix for dimension pair $(2i, 2i+1)$**:

$$\mathbf{R}_{\Theta, m}^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

where:

$$\theta_i = 10000^{-2i/d}$$

(Same base as sinusoidal encoding!)

**Apply rotation to query and key**:

For query at position $m$:

$$\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m} \mathbf{q}_m$$

For key at position $n$:

$$\tilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n} \mathbf{k}_n$$

**Attention score**:

$$\text{score}(m, n) = \tilde{\mathbf{q}}_m^T \tilde{\mathbf{k}}_n = \mathbf{q}_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \mathbf{k}_n$$

### 3.4 Key Property: Relative Position Encoding

Using rotation matrix properties:

$$\mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, n-m}$$

Therefore:

$$\text{score}(m, n) = \mathbf{q}_m^T \mathbf{R}_{\Theta, n-m} \mathbf{k}_n$$

**The attention depends only on relative position $(n - m)$, not absolute positions $m$ and $n$!**

### 3.5 Detailed Example

**Setup**:
- Dimension: $d = 4$
- Positions: $m = 1, n = 3$

**Step 1: Compute rotation angles**

$$\theta_0 = 10000^{-0/4} = 1.0$$
$$\theta_1 = 10000^{-2/4} = 0.01$$

**Step 2: Rotation matrices**

For position $m = 1$:

$$\mathbf{R}_{\Theta, 1}^{(0)} = \begin{pmatrix} \cos(1) & -\sin(1) \\ \sin(1) & \cos(1) \end{pmatrix} = \begin{pmatrix} 0.540 & -0.841 \\ 0.841 & 0.540 \end{pmatrix}$$

$$\mathbf{R}_{\Theta, 1}^{(1)} = \begin{pmatrix} \cos(0.01) & -\sin(0.01) \\ \sin(0.01) & \cos(0.01) \end{pmatrix} = \begin{pmatrix} 1.000 & -0.010 \\ 0.010 & 1.000 \end{pmatrix}$$

For position $n = 3$:

$$\mathbf{R}_{\Theta, 3}^{(0)} = \begin{pmatrix} \cos(3) & -\sin(3) \\ \sin(3) & \cos(3) \end{pmatrix} = \begin{pmatrix} -0.990 & -0.141 \\ 0.141 & -0.990 \end{pmatrix}$$

$$\mathbf{R}_{\Theta, 3}^{(1)} = \begin{pmatrix} \cos(0.03) & -\sin(0.03) \\ \sin(0.03) & \cos(0.03) \end{pmatrix} = \begin{pmatrix} 1.000 & -0.030 \\ 0.030 & 1.000 \end{pmatrix}$$

**Step 3: Apply to query/key vectors**

Query at position 1: $\mathbf{q}_1 = [q_0, q_1, q_2, q_3]$

Rotated query:
```
[q'_0]   [cos(1)  -sin(1)] [q_0]
[q'_1] = [sin(1)   cos(1)] [q_1]

[q'_2]   [cos(0.01)  -sin(0.01)] [q_2]
[q'_3] = [sin(0.01)   cos(0.01)] [q_3]
```

### 3.6 Efficient Implementation

**Instead of explicit matrix multiplication**, use element-wise operations:

For dimension pair $(x_0, x_1)$ at position $m$:

$$x'_0 = x_0 \cos(m\theta) - x_1 \sin(m\theta)$$
$$x'_1 = x_0 \sin(m\theta) + x_1 \cos(m\theta)$$

**Vectorized for all pairs**:

```python
cos_m_theta = cos(m * theta)  # (d/2,)
sin_m_theta = sin(m * theta)  # (d/2,)

x_pairs = x.reshape(-1, 2)  # (d/2, 2)
x0 = x_pairs[:, 0]
x1 = x_pairs[:, 1]

x0_new = x0 * cos_m_theta - x1 * sin_m_theta
x1_new = x0 * sin_m_theta + x1 * cos_m_theta

x_rotated = stack([x0_new, x1_new]).reshape(d)
```

### 3.7 Advantages and Limitations

**Advantages**:
- ✅ Encodes **relative** positions naturally
- ✅ Excellent extrapolation to longer sequences
- ✅ No additional parameters
- ✅ Mathematically elegant
- ✅ Strong empirical performance

**Limitations**:
- ❌ Only applies to queries and keys (not values)
- ❌ Requires dimension to be even
- ❌ More complex implementation

---

## 4. ALiBi (Attention with Linear Biases)

### 4.1 Introduction

**Proposed by**: Press et al. (2021) in "Train Short, Test Long"

**Used by**: BLOOM, MPT, StarCoder

**Revolutionary idea**: Don't encode position in embeddings; add **bias** directly to attention scores.

### 4.2 Mathematical Formulation

**Standard attention score**:

$$\text{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

**ALiBi attention score**:

$$\text{score}_{\text{ALiBi}}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|$$

where:
- $m$ is **head-specific slope** (different for each attention head)
- $|i - j|$ is **distance** between query position $i$ and key position $j$

**No positional encoding is added to input embeddings!**

### 4.3 Head-Specific Slopes

For $h$ attention heads:

$$m_{\text{head}} = 2^{-\frac{8 \cdot \text{head}}{h}}$$

**Example with 8 heads**:

```
Head 1: m = 2^(-8/8)  = 2^(-1) = 0.5
Head 2: m = 2^(-16/8) = 2^(-2) = 0.25
Head 3: m = 2^(-24/8) = 2^(-3) = 0.125
Head 4: m = 2^(-32/8) = 2^(-4) = 0.0625
Head 5: m = 2^(-40/8) = 2^(-5) = 0.03125
Head 6: m = 2^(-48/8) = 2^(-6) = 0.015625
Head 7: m = 2^(-56/8) = 2^(-7) = 0.0078125
Head 8: m = 2^(-64/8) = 2^(-8) = 0.00390625
```

**Interpretation**:
- **Larger slope** (Head 1): Stronger penalty for distance → focuses on nearby tokens
- **Smaller slope** (Head 8): Weaker penalty → can attend farther

Different heads have different **attention spans**!

### 4.4 Detailed Example

**Setup**:
- Sequence length: 5
- Query position: $i = 2$
- Head: 1 (slope $m = 0.5$)

**Compute biases**:

```
Key position j=0: bias = -0.5 × |2-0| = -0.5 × 2 = -1.0
Key position j=1: bias = -0.5 × |2-1| = -0.5 × 1 = -0.5
Key position j=2: bias = -0.5 × |2-2| = -0.5 × 0 = 0.0  (self)
Key position j=3: bias = -0.5 × |2-3| = -0.5 × 1 = -0.5
Key position j=4: bias = -0.5 × |2-4| = -0.5 × 2 = -1.0
```

**Bias matrix for head 1** (all queries):

```
      Key: 0     1     2     3     4
Query 0: [  0,  -0.5, -1.0, -1.5, -2.0]
Query 1: [-0.5,   0,  -0.5, -1.0, -1.5]
Query 2: [-1.0, -0.5,   0,  -0.5, -1.0]
Query 3: [-1.5, -1.0, -0.5,   0,  -0.5]
Query 4: [-2.0, -1.5, -1.0, -0.5,   0 ]
```

**Attention computation**:

Before softmax:
$$\text{logits}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} + \text{bias}_{ij}$$

Then apply softmax as usual.

### 4.5 Advantages and Limitations

**Advantages**:
- ✅ **Excellent extrapolation** to longer sequences
- ✅ **No positional embeddings** → saves parameters
- ✅ Simple and efficient
- ✅ Different heads have different attention spans
- ✅ Works well in practice (BLOOM: 176B parameters)

**Limitations**:
- ❌ Only encodes **distance**, not direction
- ❌ Less flexible than learned encodings
- ❌ Symmetric (distance from A to B = B to A)

---

## 5. Integer Positional Encoding

### 5.1 Mathematical Formulation

**Normalized integer encoding**:

$$\text{PE}_{\text{int}}(\text{pos}) = \frac{\text{pos}}{L - 1} \cdot \mathbf{1}_{d_{\text{model}}}$$

where $\mathbf{1}_{d_{\text{model}}}$ is a vector of ones.

**Example** ($L = 10$, $d = 4$):

```
Position 0: [0.000, 0.000, 0.000, 0.000]
Position 5: [0.556, 0.556, 0.556, 0.556]
Position 9: [1.000, 1.000, 1.000, 1.000]
```

**Multi-scale variant**:

$$\text{PE}_{\text{multi}}(\text{pos}, i) = \frac{\text{pos}}{L-1} \cdot \left(\frac{i}{d-1}\right)^\alpha$$

Different dimensions have different scales.

### 5.2 Properties

**Advantages**:
- ✅ Simplest possible encoding
- ✅ Highly interpretable
- ✅ Fast computation

**Limitations**:
- ❌ No multi-scale representation (basic version)
- ❌ Cannot extrapolate beyond training length
- ❌ Less expressive than sinusoidal

---

## 6. Binary Positional Encoding

### 6.1 Mathematical Formulation

Convert position to binary representation:

$$\text{PE}_{\text{bin}}(\text{pos}, i) = \left\lfloor \frac{\text{pos}}{2^i} \right\rfloor \bmod 2$$

**Example** (position 13 = 1101 in binary, $d = 8$):

```
PE(13) = [1, 0, 1, 1, 0, 0, 0, 0]
          ↑  ↑  ↑  ↑
         2^0 2^1 2^2 2^3
```

### 6.2 Gray Code Variant

**Standard binary problem**: Adjacent positions can differ in multiple bits.

```
Position 3: 0011
Position 4: 0100  (3 bits changed!)
```

**Gray code solution**: Adjacent positions differ by exactly 1 bit.

$$\text{gray}(n) = n \oplus \left\lfloor \frac{n}{2} \right\rfloor$$

```
Position 3: 0010 (Gray)
Position 4: 0110 (Gray) (only 1 bit changed)
```

### 6.3 Properties

**Advantages**:
- ✅ Multi-scale naturally (different bits = different frequencies)
- ✅ Can represent up to $2^d$ positions
- ✅ Deterministic

**Limitations**:
- ❌ Discrete values (hard 0/1)
- ❌ Hamming distance issues (standard binary)

---

## 7. Relative Positional Encoding

### 7.1 Introduction

**Key Idea**: Instead of encoding absolute positions, encode **relative distances** between tokens.

**Proposed by**: Shaw et al. (2018), Huang et al. (2018)

### 7.2 Mathematical Formulation

**Modify attention to include relative position**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + R}{\sqrt{d_k}}\right)V$$

where $R_{ij}$ encodes the relative position between $i$ and $j$.

**Learnable relative position embeddings**:

$$R_{ij} = \mathbf{q}_i^T \mathbf{r}_{j-i}$$

where $\mathbf{r}_k$ is learned embedding for relative distance $k$.

**Clipped version** (T5, DeBERTa):

$$k_{\text{clipped}} = \max(-k_{\max}, \min(k_{\max}, k))$$

Only learn embeddings for distances in $[-k_{\max}, k_{\max}]$.

### 7.3 Variants

**7.3.1 Relative Position Bias (T5)**

Add scalar bias to attention:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j + b_{j-i}$$

where $b_k$ is learned bias for relative position $k$.

**7.3.2 DeBERTa**

Separate relative position attention:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j + \mathbf{q}_i \cdot \mathbf{r}_{j-i} + \mathbf{k}_j \cdot \mathbf{r}_{i-j}$$

---

## 8. Advanced Techniques

### 8.1 Complex-Valued Positional Encoding

Represent rotations in complex plane:

$$\text{PE}_{\text{complex}}(\text{pos}, i) = e^{i \cdot \text{pos} \cdot \theta_i}$$

where $\theta_i = 10000^{-2i/d}$.

### 8.2 Fourier Features

Random Fourier features for positional encoding:

$$\text{PE}_{\text{fourier}}(\text{pos}) = [\cos(2\pi \mathbf{B} \text{pos}), \sin(2\pi \mathbf{B} \text{pos})]$$

where $\mathbf{B}$ is random matrix sampled once.

### 8.3 Conditional Positional Encoding (CPE)

Position encoding that depends on content:

$$\text{PE}_{\text{CPE}}(\text{pos}, \mathbf{x}) = f_{\text{conv}}(\mathbf{x})$$

Use depth-wise convolution to generate position-aware features.

### 8.4 No Positional Encoding (NoPE)

**Recent finding**: Some models (e.g., Transformer-XL with relative attention) work well without explicit positional encoding!

Model learns position from:
- Attention patterns
- Relative position mechanisms
- Causal masking

---

## 9. Comprehensive Comparison

### 9.1 Feature Comparison Table

| Method | Type | Parameters | Max Length | Extrapolation | Multi-Scale | Used By |
|--------|------|------------|------------|---------------|-------------|---------|
| **Sinusoidal** | Absolute | 0 | ∞ | ✅ Good | ✅ Yes | Transformer |
| **Learned** | Absolute | $L \times d$ | $L$ | ❌ Poor | ⚠️ Implicit | BERT, GPT-2 |
| **RoPE** | Relative | 0 | ∞ | ✅ Excellent | ✅ Yes | LLaMA, PaLM |
| **ALiBi** | Relative | 0 | ∞ | ✅ Excellent | ✅ Yes | BLOOM |
| **Integer** | Absolute | 0 | $L$ | ❌ No | ❌ No | Simple tasks |
| **Binary** | Absolute | 0 | $2^d$ | ✅ Limited | ✅ Yes | Research |
| **Relative** | Relative | $2k \times d$ | ∞ | ✅ Good | ⚠️ Learned | T5, DeBERTa |

### 9.2 Performance Comparison

**Perplexity on language modeling** (lower is better):

```
Method          | Short (≤512) | Long (1024+) | Extrapolation
----------------|--------------|--------------|---------------
Sinusoidal      | 23.4         | 24.1         | 28.5
Learned         | 22.8         | 23.5         | 45.2 (poor!)
RoPE            | 22.5         | 23.0         | 24.8
ALiBi           | 23.1         | 23.2         | 23.9
```

**Trends**:
- Learned: Best on training lengths, worst on extrapolation
- RoPE: Consistently strong across all lengths
- ALiBi: Best extrapolation

### 9.3 Computational Complexity

| Method | Precomputation | Per Token | Memory |
|--------|---------------|-----------|--------|
| Sinusoidal | $O(Ld)$ | $O(1)$ | $O(Ld)$ |
| Learned | — | $O(1)$ | $O(Ld)$ |
| RoPE | $O(Ld)$ | $O(d)$ | $O(Ld)$ |
| ALiBi | — | $O(L)$ | $O(hL^2)$ |
| Integer | — | $O(d)$ | $O(d)$ |
| Binary | — | $O(d)$ | $O(d)$ |

---

## 10. Choosing the Right Encoding

### Decision Tree

```
Is extrapolation critical?
├─ YES
│  ├─ Want relative positions? → RoPE or ALiBi
│  └─ Want absolute positions? → Sinusoidal
└─ NO
   ├─ Have lots of data? → Learned
   ├─ Want simplicity? → Integer
   └─ Research/experiments? → Binary or Relative
```

### Recommendations by Task

**Language Modeling (GPT-style)**:
- Best: RoPE (used by LLaMA, Mistral)
- Alternative: ALiBi (used by BLOOM)

**Masked Language Modeling (BERT-style)**:
- Best: Learned (used by BERT)
- Alternative: Sinusoidal

**Long-Context Tasks**:
- Best: ALiBi or RoPE
- Avoid: Learned (cannot extrapolate)

**Short Sequences (< 512 tokens)**:
- Any method works well
- Learned slightly better

**Research/New Architectures**:
- Start with: Sinusoidal (no hyperparameters)
- Then try: RoPE (strong baseline)

---

## Key Takeaways

1. **Sinusoidal**: Classic, simple, works well
2. **Learned**: Task-adaptive but cannot extrapolate
3. **RoPE**: State-of-the-art for relative positions
4. **ALiBi**: Best extrapolation, used in large models
5. **Integer/Binary**: Simple but limited
6. **Relative**: Flexible, used in T5 and DeBERTa

**Modern trend**: Moving toward relative position encodings (RoPE, ALiBi) for better extrapolation.

---

## References

1. Vaswani et al. (2017). "Attention Is All You Need" - Sinusoidal
2. Devlin et al. (2018). "BERT" - Learned
3. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding" - RoPE
4. Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases" - ALiBi
5. Shaw et al. (2018). "Self-Attention with Relative Position Representations" - Relative
6. Raffel et al. (2020). "T5" - Relative Position Bias
7. He et al. (2020). "DeBERTa" - Disentangled Attention

---

*This document provides complete mathematical coverage of all major positional encoding techniques. For implementation, see the accompanying code file.*
