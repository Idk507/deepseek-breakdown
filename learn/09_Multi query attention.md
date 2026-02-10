# Multi-Query Attention: A Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Background: Multi-Head Attention Review](#background-multi-head-attention-review)
3. [The Problem with Multi-Head Attention](#the-problem-with-multi-head-attention)
4. [Multi-Query Attention Architecture](#multi-query-attention-architecture)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Step-by-Step Computation](#step-by-step-computation)
7. [Comparison: MHA vs MQA](#comparison-mha-vs-mqa)
8. [Why Multi-Query Attention Works](#why-multi-query-attention-works)
9. [Computational Analysis](#computational-analysis)
10. [Use Cases and Applications](#use-cases-and-applications)

---

## Introduction

**Multi-Query Attention (MQA)** is a variant of multi-head attention introduced in the paper "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019). It's designed to significantly reduce memory bandwidth requirements and improve inference speed, especially in autoregressive generation tasks.

### Key Insight

Instead of having separate Key and Value projections for each attention head (as in Multi-Head Attention), MQA uses:
- **Multiple query heads** (like MHA)
- **Single shared key projection** across all heads
- **Single shared value projection** across all heads

This dramatically reduces the KV cache size during inference, which is the primary bottleneck in large language model deployment.

---

## Background: Multi-Head Attention Review

### Standard Multi-Head Attention (MHA)

In MHA, for $h$ heads:

**Parameters:**
- Query projections: $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$ for $i = 1, ..., h$
- Key projections: $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$ for $i = 1, ..., h$
- Value projections: $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ for $i = 1, ..., h$
- Output projection: $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

**Computation for head $i$:**
$$Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

**Total parameters for Q, K, V:** $3hd_{model}d_k = 3d_{model}^2$ (when $d_k = d_{model}/h$)

---

## The Problem with Multi-Head Attention

### Memory Bottleneck in Inference

During autoregressive generation (e.g., GPT models), we use **KV caching**:

1. **Without KV cache:** For each new token, recompute K and V for all previous tokens (wasteful)
2. **With KV cache:** Store computed K and V values, only compute for new token

**KV Cache Size for MHA:**
$$\text{Memory} = 2 \times \text{batch\_size} \times \text{num\_layers} \times h \times \text{seq\_len} \times d_k$$

For typical models:
- Layers: 32-96
- Heads: 32-128
- Sequence length: 2048-32768
- $d_k$: 64-128

**Example (GPT-3 scale):**
- 96 layers × 96 heads × 8192 seq_len × 128 $d_k$ × 2 bytes (FP16)
- ≈ **19 GB** just for KV cache per sample!

### The Bandwidth Problem

During generation:
1. Load KV cache from memory (slow)
2. Compute attention (fast, compute-bound)
3. The memory bandwidth becomes the bottleneck, not computation

**Goal:** Reduce KV cache size without significantly hurting model quality.

---

## Multi-Query Attention Architecture

### Core Idea

Share the Key and Value projections across all query heads:
- Multiple query heads: $Q_1, Q_2, ..., Q_h$ (each with own projection)
- Single key head: $K$ (shared across all query heads)
- Single value head: $V$ (shared across all query heads)

### Visual Comparison

**Multi-Head Attention:**
```
Input X
├─ Q₁ ─┐       ├─ K₁ ─┐       ├─ V₁ ─┐
├─ Q₂ ─┤       ├─ K₂ ─┤       ├─ V₂ ─┤
├─ Q₃ ─┤  →    ├─ K₃ ─┤  →    ├─ V₃ ─┤  → Concat → Output
├─ ...─┤       ├─ ...─┤       ├─ ...─┤
└─ Qₕ ─┘       └─ Kₕ ─┘       └─ Vₕ ─┘

h different Q projections
h different K projections  ← Memory intensive!
h different V projections  ← Memory intensive!
```

**Multi-Query Attention:**
```
Input X
├─ Q₁ ─┐              ┌─ K (shared)
├─ Q₂ ─┤              │
├─ Q₃ ─┤  →    ───────┤  →    ───────┐
├─ ...─┤              │              │
└─ Qₕ ─┘              └─ V (shared)  │  → Concat → Output

h different Q projections
1 shared K projection   ← Huge memory savings!
1 shared V projection   ← Huge memory savings!
```

---

## Mathematical Formulation

### Parameters

Given:
- Input dimension: $d_{model}$
- Number of query heads: $h$
- Dimension per query head: $d_k = d_{model} / h$
- Key/Value dimension: $d_k$ (same as query head dimension)

### Weight Matrices

**Query projections (one per head):**
$$W_i^Q \in \mathbb{R}^{d_{model} \times d_k} \quad \text{for } i = 1, 2, ..., h$$

**Shared Key projection:**
$$W^K \in \mathbb{R}^{d_{model} \times d_k}$$

**Shared Value projection:**
$$W^V \in \mathbb{R}^{d_{model} \times d_k}$$

**Output projection:**
$$W^O \in \mathbb{R}^{hd_k \times d_{model}}$$

### Forward Pass

**Input:**
- Query input: $X_Q \in \mathbb{R}^{n \times d_{model}}$ (n = sequence length)
- Key input: $X_K \in \mathbb{R}^{m \times d_{model}}$ (m = key sequence length)
- Value input: $X_V \in \mathbb{R}^{m \times d_{model}}$

**Step 1: Compute projections**

For each query head $i$:
$$Q_i = X_Q W_i^Q \in \mathbb{R}^{n \times d_k}$$

Shared key and value (computed once):
$$K = X_K W^K \in \mathbb{R}^{m \times d_k}$$
$$V = X_V W^V \in \mathbb{R}^{m \times d_k}$$

**Step 2: Scaled dot-product attention per head**

For each head $i$:
$$\text{head}_i = \text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right)V$$

Breaking this down:
$$S_i = \frac{Q_i K^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$$
$$A_i = \text{softmax}(S_i) \in \mathbb{R}^{n \times m}$$
$$\text{head}_i = A_i V \in \mathbb{R}^{n \times d_k}$$

**Step 3: Concatenate heads**
$$\text{Concat} = [\text{head}_1 \,|\, \text{head}_2 \,|\, ... \,|\, \text{head}_h] \in \mathbb{R}^{n \times hd_k}$$

**Step 4: Output projection**
$$\text{Output} = \text{Concat} \cdot W^O \in \mathbb{R}^{n \times d_{model}}$$

### Key Difference from Multi-Head Attention

The crucial difference is:
- **MHA:** Each head $i$ uses its own $K_i$ and $V_i$
- **MQA:** All heads share the same $K$ and $V$

This means:
- Different query heads can still attend to different patterns
- But they all look at the same key-value representations
- Each head computes different attention weights $A_i$ based on its unique $Q_i$

---

## Step-by-Step Computation

Let's work through a concrete example.

### Setup

**Hyperparameters:**
- Sequence length: $n = 4$
- Model dimension: $d_{model} = 512$
- Number of heads: $h = 8$
- Head dimension: $d_k = d_{model}/h = 64$

**Input:**
- $X \in \mathbb{R}^{4 \times 512}$ (for self-attention: $X_Q = X_K = X_V = X$)

### Step 1: Linear Projections

**Query projections (8 different ones):**

For head 1:
$$Q_1 = X W_1^Q = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$

For head 2:
$$Q_2 = X W_2^Q = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$

... and so on for all 8 heads.

**Shared key and value (computed once):**

$$K = X W^K = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$

$$V = X W^V = \mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 64} = \mathbb{R}^{4 \times 64}$$

**Example values (randomly initialized):**

$$Q_1 = \begin{bmatrix}
0.2 & -0.5 & ... & 0.3 \\
0.1 & 0.8 & ... & -0.2 \\
-0.4 & 0.3 & ... & 0.6 \\
0.7 & -0.1 & ... & 0.4
\end{bmatrix}_{4 \times 64}$$

$$K = \begin{bmatrix}
0.5 & 0.2 & ... & -0.3 \\
-0.1 & 0.6 & ... & 0.4 \\
0.3 & -0.4 & ... & 0.1 \\
0.8 & 0.1 & ... & -0.2
\end{bmatrix}_{4 \times 64}$$

$$V = \begin{bmatrix}
0.3 & 0.1 & ... & 0.5 \\
0.6 & -0.2 & ... & 0.3 \\
-0.1 & 0.4 & ... & -0.2 \\
0.2 & 0.7 & ... & 0.1
\end{bmatrix}_{4 \times 64}$$

### Step 2: Attention for Head 1

**Compute similarity scores:**
$$S_1 = Q_1 K^T = \mathbb{R}^{4 \times 64} \times \mathbb{R}^{64 \times 4} = \mathbb{R}^{4 \times 4}$$

Example result:
$$S_1 = \begin{bmatrix}
18.2 & 12.4 & 15.6 & 10.8 \\
14.5 & 20.1 & 11.3 & 16.7 \\
11.8 & 13.2 & 19.4 & 9.5 \\
16.3 & 15.8 & 12.1 & 21.6
\end{bmatrix}$$

**Scale by √d_k:**
$$S_{scaled} = \frac{S_1}{\sqrt{64}} = \frac{S_1}{8}$$

$$S_{scaled} = \begin{bmatrix}
2.275 & 1.550 & 1.950 & 1.350 \\
1.813 & 2.513 & 1.413 & 2.088 \\
1.475 & 1.650 & 2.425 & 1.188 \\
2.038 & 1.975 & 1.513 & 2.700
\end{bmatrix}$$

**Apply softmax (row-wise):**

For row 1:
$$e^{2.275} = 9.73, \quad e^{1.550} = 4.71, \quad e^{1.950} = 7.03, \quad e^{1.350} = 3.86$$
$$\text{sum} = 25.33$$

$$A_1[0, :] = \begin{bmatrix}
0.384 & 0.186 & 0.278 & 0.152
\end{bmatrix}$$

Complete attention matrix:
$$A_1 = \begin{bmatrix}
0.384 & 0.186 & 0.278 & 0.152 \\
0.161 & 0.462 & 0.134 & 0.243 \\
0.149 & 0.182 & 0.516 & 0.153 \\
0.202 & 0.182 & 0.125 & 0.491
\end{bmatrix}$$

**Compute weighted sum:**
$$\text{head}_1 = A_1 V = \mathbb{R}^{4 \times 4} \times \mathbb{R}^{4 \times 64} = \mathbb{R}^{4 \times 64}$$

### Step 3: Attention for Heads 2-8

**Important:** All heads use the **same K and V**, but different $Q_i$:

For head 2:
$$S_2 = Q_2 K^T$$ (different from $S_1$ because $Q_2 \neq Q_1$)
$$A_2 = \text{softmax}(S_2 / \sqrt{d_k})$$
$$\text{head}_2 = A_2 V$$ (same $V$ as head 1!)

This repeats for all 8 heads. Each head produces:
$$\text{head}_i \in \mathbb{R}^{4 \times 64}$$

### Step 4: Concatenation

$$\text{Concat} = [\text{head}_1 \,|\, \text{head}_2 \,|\, ... \,|\, \text{head}_8]$$

$$\text{Concat} \in \mathbb{R}^{4 \times (8 \times 64)} = \mathbb{R}^{4 \times 512}$$

### Step 5: Output Projection

$$\text{Output} = \text{Concat} \cdot W^O$$

$$\mathbb{R}^{4 \times 512} \times \mathbb{R}^{512 \times 512} = \mathbb{R}^{4 \times 512}$$

---

## Comparison: MHA vs MQA

### Parameter Count

**Multi-Head Attention:**
- Query weights: $h \times d_{model} \times d_k = d_{model}^2$
- Key weights: $h \times d_{model} \times d_k = d_{model}^2$
- Value weights: $h \times d_{model} \times d_k = d_{model}^2$
- Output weights: $d_{model}^2$
- **Total:** $4d_{model}^2$

**Multi-Query Attention:**
- Query weights: $h \times d_{model} \times d_k = d_{model}^2$
- Key weights: $d_{model} \times d_k = d_{model}^2 / h$
- Value weights: $d_{model} \times d_k = d_{model}^2 / h$
- Output weights: $d_{model}^2$
- **Total:** $2d_{model}^2 + 2d_{model}^2/h \approx 2d_{model}^2$ (for large $h$)

**Reduction:** ~50% fewer parameters

### KV Cache Size

**Multi-Head Attention:**
$$\text{Cache}_{\text{MHA}} = 2 \times \text{layers} \times h \times \text{seq\_len} \times d_k$$

**Multi-Query Attention:**
$$\text{Cache}_{\text{MQA}} = 2 \times \text{layers} \times 1 \times \text{seq\_len} \times d_k$$

**Reduction factor:** $h$ (number of heads)

For typical models with $h = 32$ or $h = 96$:
- **32x smaller** cache for $h=32$
- **96x smaller** cache for $h=96$

### Example: GPT-3 Scale Model

**Configuration:**
- Layers: 96
- Heads: 96
- $d_k$: 128
- Sequence length: 2048
- Precision: FP16 (2 bytes)

**MHA KV Cache:**
$$2 \times 96 \times 96 \times 2048 \times 128 \times 2 = 19.2 \text{ GB}$$

**MQA KV Cache:**
$$2 \times 96 \times 1 \times 2048 \times 128 \times 2 = 200 \text{ MB}$$

**Reduction:** 96x smaller (19.2 GB → 200 MB)

### Inference Speed

**Memory bandwidth saved:**
- Loading KV cache is the bottleneck during generation
- With 96x smaller cache, memory bandwidth requirement drops 96x
- This directly translates to faster generation

**Typical speedups:**
- 1.5-3x faster inference for standard batch sizes
- Even larger speedups for long sequences or large batch sizes

### Quality Comparison

**Empirical findings:**
- Small quality degradation (typically <2% on downstream tasks)
- Can be recovered with:
  - Slightly longer training
  - Uptraining (continue training MHA model with MQA)
  - Grouped-Query Attention (GQA) - middle ground

---

## Why Multi-Query Attention Works

### Intuition

**Query Diversity is More Important than Key/Value Diversity:**

1. **Queries determine "what to look for"**
   - Different query heads can learn to identify different patterns
   - Example: syntactic relations, semantic relations, positional patterns

2. **Keys/Values provide "what information exists"**
   - The underlying information doesn't need multiple representations
   - Different queries can extract different aspects from the same K/V

### Analogy: Library Search

Think of attention as searching a library:

**Multi-Head Attention (MHA):**
- 8 different librarians (query heads)
- 8 different catalogs (key heads)  
- 8 different book collections (value heads)
- Each librarian uses their own catalog and collection

**Multi-Query Attention (MQA):**
- 8 different librarians (query heads)
- 1 shared catalog (key head)
- 1 shared book collection (value head)
- Each librarian searches the same catalog differently and retrieves from the same collection

The key insight: Having different search strategies (queries) is more important than having different catalogs (keys) and collections (values).

### Mathematical Perspective

The attention output for head $i$ is:
$$\text{head}_i = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

Even with shared $K$ and $V$:
- Each $Q_i$ produces different attention patterns (via $Q_i K^T$)
- Different attention patterns extract different information from the same $V$
- The diversity in queries is sufficient for most tasks

### When Does It Hurt?

MQA may underperform when:
- Very complex reasoning tasks requiring subtle distinctions
- Tasks where different heads need fundamentally different value representations
- Small models where every parameter counts

In practice, the performance gap is small for most applications.

---

## Computational Analysis

### Time Complexity

**Multi-Head Attention:**
$$\text{Time}_{\text{MHA}} = O(n^2 d_{model} + n d_{model}^2)$$

**Multi-Query Attention:**
$$\text{Time}_{\text{MQA}} = O(n^2 d_{model} + n d_{model}^2)$$

**Training time:** Same asymptotic complexity

However, in practice:
- MQA has fewer parameters to update
- Slightly faster training (5-10% typical)

### Space Complexity

**Training:**
- Similar memory for activations
- ~50% fewer parameters for MQA

**Inference (with KV caching):**
- **MHA:** $O(h \times \text{seq\_len} \times d_k)$ per layer
- **MQA:** $O(1 \times \text{seq\_len} \times d_k)$ per layer
- **Reduction:** $h$ times smaller

### Memory Bandwidth Analysis

During autoregressive generation:

**Per token generated:**
1. Load KV cache from memory
2. Compute attention
3. Update KV cache

**MHA bandwidth per token:**
$$\text{Bandwidth}_{\text{MHA}} = 2 \times h \times d_k \times \text{seq\_len} \times \text{bytes}$$

**MQA bandwidth per token:**
$$\text{Bandwidth}_{\text{MQA}} = 2 \times 1 \times d_k \times \text{seq\_len} \times \text{bytes}$$

For long sequences, memory bandwidth dominates computation, so MQA provides near-linear speedup in $h$.

---

## Use Cases and Applications

### Where MQA Excels

**1. Autoregressive Language Models**
- GPT-style models
- Long-context generation
- Real-time applications (chatbots, code generation)

**2. Resource-Constrained Deployment**
- Mobile devices
- Edge computing
- Serving many concurrent users

**3. Long-Context Tasks**
- Document summarization
- Long-form QA
- Code analysis
- Where KV cache becomes massive

**4. Batch Inference**
- Serving multiple requests simultaneously
- Limited GPU memory budget

### Models Using MQA

**Production Models:**
- **PaLM** (Google): Uses MQA for efficient scaling
- **Falcon** (TII): Uses MQA for open-source efficiency
- **StarCoder**: Code generation with MQA
- Various open-source models prioritizing inference speed

### Grouped-Query Attention (GQA)

A middle ground between MHA and MQA:

**Idea:** Instead of 1 KV head (MQA) or $h$ KV heads (MHA), use $g$ KV heads where $1 < g < h$

**Configuration:**
- $h$ query heads
- $g$ key-value head groups
- Each group of $h/g$ query heads shares one KV head

**Example:** $h=32$ query heads, $g=8$ KV groups
- 4x memory reduction vs MHA
- Better quality than pure MQA
- Used in LLaMA-2, Mistral

---

## Key Takeaways

1. **MQA reduces KV cache by $h$ times** by sharing key and value projections across query heads

2. **Parameter reduction:** ~50% fewer parameters compared to MHA

3. **Inference speedup:** 1.5-3x faster generation, more for long sequences

4. **Quality tradeoff:** Small degradation (<2% typical) recoverable with training

5. **Best for:** Autoregressive models, long contexts, resource-constrained deployment

6. **Design principle:** Query diversity matters more than key/value diversity

7. **Evolution:** MHA → MQA → GQA (grouped-query attention for balance)

---

## References

- Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv preprint*.
- Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv preprint*.
- Used in: PaLM, Falcon, StarCoder, and many modern LLMs

---

