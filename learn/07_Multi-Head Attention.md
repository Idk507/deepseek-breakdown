# Multi-Head Attention: A Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Preliminaries: Scaled Dot-Product Attention](#preliminaries-scaled-dot-product-attention)
3. [Multi-Head Attention Architecture](#multi-head-attention-architecture)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Step-by-Step Computation](#step-by-step-computation)
6. [Intuition and Why It Works](#intuition-and-why-it-works)
7. [Computational Complexity](#computational-complexity)

---

## Introduction

Multi-Head Attention is a core mechanism in the Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). It allows the model to jointly attend to information from different representation subspaces at different positions, enabling richer feature extraction and better sequence modeling.

The key innovation is running multiple attention mechanisms in parallel, each focusing on different aspects of the input, then combining their outputs.

---

## Preliminaries: Scaled Dot-Product Attention

Before diving into multi-head attention, we must understand the building block: **Scaled Dot-Product Attention**.

### Mathematical Definition

Given:
- **Query** matrix: $Q \in \mathbb{R}^{n \times d_k}$
- **Key** matrix: $K \in \mathbb{R}^{m \times d_k}$
- **Value** matrix: $V \in \mathbb{R}^{m \times d_v}$

Where:
- $n$ = number of queries (sequence length for queries)
- $m$ = number of keys/values (sequence length for keys/values)
- $d_k$ = dimension of queries and keys
- $d_v$ = dimension of values

The attention output is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Step-by-Step Breakdown

1. **Compute Similarity Scores**: $S = QK^T \in \mathbb{R}^{n \times m}$
   - Each element $S_{ij}$ represents the similarity between query $i$ and key $j$

2. **Scale**: $S_{scaled} = \frac{S}{\sqrt{d_k}}$
   - Scaling prevents extremely small gradients when $d_k$ is large
   - Without scaling, dot products grow large in magnitude, pushing softmax into regions with small gradients

3. **Apply Softmax**: $A = \text{softmax}(S_{scaled}) \in \mathbb{R}^{n \times m}$
   - Normalizes scores to create attention weights
   - For each query $i$: $A_{ij} = \frac{e^{S_{scaled,ij}}}{\sum_{k=1}^{m} e^{S_{scaled,ik}}}$
   - Each row sums to 1

4. **Weighted Sum**: $\text{Output} = AV \in \mathbb{R}^{n \times d_v}$
   - Combines values based on attention weights

### Why Scaling by √d_k?

As $d_k$ increases, the variance of $QK^T$ grows proportionally to $d_k$. Assuming $Q$ and $K$ have zero mean and unit variance:

$$\text{Var}(q \cdot k) = d_k$$

Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping the softmax inputs in a reasonable range.

---

## Multi-Head Attention Architecture

Multi-Head Attention runs $h$ parallel attention mechanisms (called "heads"), each with learned linear projections of $Q$, $K$, and $V$.

### Visual Overview

```
Input: Q, K, V (d_model dimensional)
         |
         ├─────┬─────┬─────┬─────┐
         |     |     |     |     |
       Head1 Head2 Head3 ... Headh
         |     |     |     |     |
         └─────┴─────┴─────┴─────┘
                    |
              Concatenate
                    |
            Linear Projection
                    |
                 Output
```

---

## Mathematical Formulation

### Input Dimensions

Given:
- Input queries: $Q \in \mathbb{R}^{n \times d_{model}}$
- Input keys: $K \in \mathbb{R}^{m \times d_{model}}$
- Input values: $V \in \mathbb{R}^{m \times d_{model}}$
- Number of heads: $h$
- Dimension per head: $d_k = d_v = \frac{d_{model}}{h}$

### Learnable Parameters

For each head $i \in \{1, 2, ..., h\}$:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ (query projection)
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ (key projection)
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ (value projection)

Additionally:
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$ (output projection)

### Complete Formulation

**Step 1: Project inputs for each head**

For head $i$:
$$Q_i = QW_i^Q \in \mathbb{R}^{n \times d_k}$$
$$K_i = KW_i^K \in \mathbb{R}^{m \times d_k}$$
$$V_i = VW_i^V \in \mathbb{R}^{m \times d_v}$$

**Step 2: Compute attention for each head**

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i \in \mathbb{R}^{n \times d_v}$$

**Step 3: Concatenate all heads**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

Where:
$$\text{Concat}(\text{head}_1, ..., \text{head}_h) \in \mathbb{R}^{n \times hd_v}$$

**Step 4: Final linear projection**

The concatenated output is projected back to $d_{model}$ dimensions:
$$\text{Output} = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \in \mathbb{R}^{n \times d_{model}}$$

---

## Step-by-Step Computation

Let's work through a concrete example with actual numbers.

### Example Setup

**Parameters:**
- Sequence length: $n = m = 4$ (self-attention)
- Model dimension: $d_{model} = 512$
- Number of heads: $h = 8$
- Dimension per head: $d_k = d_v = 512/8 = 64$

**Input:**
- $X \in \mathbb{R}^{4 \times 512}$ (input sequence)
- For self-attention: $Q = K = V = X$

### Detailed Computation

#### Head 1 Computation

**1. Linear Projections:**

$$Q_1 = XW_1^Q = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$
$$K_1 = XW_1^K = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$
$$V_1 = XW_1^V = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$

**2. Compute Scores:**

$$S_1 = Q_1K_1^T = \mathbb{R}^{4 \times 64} \times \mathbb{R}^{64 \times 4} = \mathbb{R}^{4 \times 4}$$

Example (with random values):
$$S_1 = \begin{bmatrix}
20.5 & 15.2 & 8.3 & 12.1 \\
16.8 & 22.3 & 10.5 & 14.2 \\
9.2 & 11.5 & 19.8 & 7.6 \\
13.4 & 15.1 & 9.9 & 21.2
\end{bmatrix}$$

**3. Scale:**

$$S_{scaled} = \frac{S_1}{\sqrt{64}} = \frac{S_1}{8}$$

$$S_{scaled} = \begin{bmatrix}
2.56 & 1.90 & 1.04 & 1.51 \\
2.10 & 2.79 & 1.31 & 1.78 \\
1.15 & 1.44 & 2.48 & 0.95 \\
1.68 & 1.89 & 1.24 & 2.65
\end{bmatrix}$$

**4. Apply Softmax (row-wise):**

For row 1: 
$$e^{2.56} = 12.94, \quad e^{1.90} = 6.69, \quad e^{1.04} = 2.83, \quad e^{1.51} = 4.53$$
$$\text{sum} = 26.99$$

$$A_1[0,:] = \begin{bmatrix} 0.48 & 0.25 & 0.10 & 0.17 \end{bmatrix}$$

Similarly for all rows:
$$A_1 = \begin{bmatrix}
0.48 & 0.25 & 0.10 & 0.17 \\
0.18 & 0.51 & 0.10 & 0.21 \\
0.12 & 0.16 & 0.62 & 0.10 \\
0.16 & 0.19 & 0.10 & 0.55
\end{bmatrix}$$

**5. Weighted Sum:**

$$\text{head}_1 = A_1V_1 = \mathbb{R}^{4 \times 4} \times \mathbb{R}^{4 \times 64} = \mathbb{R}^{4 \times 64}$$

#### Repeat for All 8 Heads

Each head produces an output of shape $\mathbb{R}^{4 \times 64}$.

#### Concatenation

$$\text{Concat} = [\text{head}_1 \,|\, \text{head}_2 \,|\, ... \,|\, \text{head}_8] \in \mathbb{R}^{4 \times 512}$$

This is achieved by stacking the 64-dimensional outputs horizontally:
$$[8 \times (4 \times 64)] \rightarrow (4 \times 512)$$

#### Final Projection

$$\text{Output} = \text{Concat} \cdot W^O = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 512} = \mathbb{R}^{4 \times 512}$$

---

## Intuition and Why It Works

### Why Multiple Heads?

**1. Multiple Representation Subspaces:**
- Each head can learn to focus on different aspects of the input
- One head might focus on syntactic relationships
- Another might focus on semantic relationships
- Another might capture long-range dependencies

**2. Ensemble Effect:**
- Multiple heads provide different "views" of the data
- Combining them creates a richer representation
- Similar to ensemble learning in traditional ML

**3. Gradient Flow:**
- Multiple heads provide multiple gradient pathways
- Helps with optimization and reduces risk of getting stuck

### Example: Sentence Processing

Consider the sentence: **"The animal didn't cross the street because it was too tired."**

- **Head 1** might learn: "it" refers to "animal" (coreference resolution)
- **Head 2** might learn: "tired" relates to "animal" (semantic relationship)
- **Head 3** might learn: "didn't cross" is the main action (syntactic structure)
- **Head 4** might learn: "because" introduces causation (logical relationship)

Each head specializes, and together they capture the full meaning.

### Mathematical Benefits

**1. Parameter Efficiency:**
- Total parameters: $h \times (3d_{model}d_k + d_{model}^2)$ where $d_k = d_{model}/h$
- Versus single head with same total dimension: much fewer parameters for similar capacity

**2. Parallel Computation:**
- All heads can be computed in parallel
- Efficient GPU utilization

**3. Dimensionality:**
- By using $d_k = d_{model}/h$, we keep total computation similar to single-head attention
- Get representational benefits without computational explosion

---

## Computational Complexity

### Time Complexity

For sequence length $n$ and model dimension $d_{model}$:

**Per Head:**
1. Linear projections: $O(n \cdot d_{model} \cdot d_k) = O(n \cdot d_{model}^2 / h)$
2. $QK^T$: $O(n^2 \cdot d_k) = O(n^2 \cdot d_{model} / h)$
3. Softmax: $O(n^2)$
4. Attention·V: $O(n^2 \cdot d_v) = O(n^2 \cdot d_{model} / h)$

**All Heads (parallel):**
- Projections: $O(n \cdot d_{model}^2)$
- Attention computation: $O(n^2 \cdot d_{model})$
- Output projection: $O(n \cdot d_{model}^2)$

**Total: $O(n^2 \cdot d_{model} + n \cdot d_{model}^2)$**

### Space Complexity

**Storage:**
- Parameters: $O(d_{model}^2)$
- Activations per layer: $O(n \cdot d_{model})$
- Attention matrices: $O(h \cdot n^2)$

**Total: $O(n^2 \cdot h + n \cdot d_{model} + d_{model}^2)$**

### Comparison with Single-Head Attention

Multi-head attention with $h$ heads and dimension $d_k = d_{model}/h$ per head has:
- **Same computational complexity** as single-head attention with dimension $d_{model}$
- **Better representation learning** due to multiple subspaces
- **More parameters** by a factor proportional to $h$

---

## Key Takeaways

1. **Multi-head attention = parallel attention mechanisms** with different learned projections
2. **Each head has dimension $d_k = d_{model}/h$** to keep computation manageable
3. **Heads specialize** in different aspects of the input relationships
4. **Concatenation + projection** combines insights from all heads
5. **Quadratic in sequence length** $O(n^2)$ - the main bottleneck for long sequences
6. **Core building block** of Transformers, used in encoder, decoder, and cross-attention

---

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- The scaling factor $\sqrt{d_k}$ is derived from variance analysis of dot products.
- Modern implementations often use optimizations like Flash Attention for efficiency.

---

