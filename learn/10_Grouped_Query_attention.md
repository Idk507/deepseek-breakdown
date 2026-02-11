# Grouped Query Attention (GQA): Complete Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem with Multi-Head Attention](#the-problem-with-multi-head-attention)
3. [Evolution of Attention Mechanisms](#evolution-of-attention-mechanisms)
4. [Grouped Query Attention Explained](#grouped-query-attention-explained)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
7. [Detailed Examples](#detailed-examples)
8. [Memory and Computation Analysis](#memory-and-computation-analysis)
9. [Comparison with Other Variants](#comparison-with-other-variants)
10. [Implementation Details](#implementation-details)
11. [Practical Applications](#practical-applications)
12. [Advanced Topics](#advanced-topics)

---

## Introduction

### What is Grouped Query Attention (GQA)?

**Grouped Query Attention (GQA)** is a memory-efficient variant of multi-head attention that reduces the Key-Value (KV) cache size during inference while maintaining model quality.

**Key Innovation:**
> Instead of having separate Key and Value projections for each attention head, GQA shares K and V across groups of Query heads.

### Historical Context

```
Timeline of Attention Evolution:

2017: Multi-Head Attention (MHA)
      - Separate Q, K, V for each head
      - Best quality, highest memory

2019: Multi-Query Attention (MQA)
      - Single K, V shared across all heads
      - Lowest memory, some quality loss

2023: Grouped Query Attention (GQA)
      - K, V shared within groups
      - Middle ground: good quality + efficiency
```

### Why GQA Matters

**Problem:** Large language models are expensive to serve

```
Example: 70B parameter model with 8K context

Multi-Head Attention (32 heads):
- KV cache: ~180 GB per batch
- Limits batch size and throughput

Grouped Query Attention (8 groups):
- KV cache: ~45 GB per batch
- 4× memory reduction
- Similar quality to MHA
```

### Visual Comparison

```
Multi-Head Attention (MHA):
Q heads: [Q₁] [Q₂] [Q₃] [Q₄] [Q₅] [Q₆] [Q₇] [Q₈]
K heads: [K₁] [K₂] [K₃] [K₄] [K₅] [K₆] [K₇] [K₈]
V heads: [V₁] [V₂] [V₃] [V₄] [V₅] [V₆] [V₇] [V₈]

Each Q head has its own K, V

Multi-Query Attention (MQA):
Q heads: [Q₁] [Q₂] [Q₃] [Q₄] [Q₅] [Q₆] [Q₇] [Q₈]
K heads: [K₁] shared across all
V heads: [V₁] shared across all

All Q heads share single K, V

Grouped Query Attention (GQA):
Q heads: [Q₁] [Q₂] [Q₃] [Q₄] [Q₅] [Q₆] [Q₇] [Q₈]
         └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
K heads:   [K₁]   [K₂]   [K₃]   [K₄]
V heads:   [V₁]   [V₂]   [V₃]   [V₄]

Groups of Q heads share K, V
```

---

## The Problem with Multi-Head Attention

### Memory Bottleneck During Inference

**Standard Multi-Head Attention:**

```
Configuration:
- num_heads = 32
- head_dim = 128
- total_dim = 32 × 128 = 4096
- sequence_length = 2048
- batch_size = 8

KV Cache Memory:
= 2 (K and V) 
  × num_layers (e.g., 80)
  × num_heads (32)
  × head_dim (128)
  × seq_len (2048)
  × batch_size (8)
  × bytes (2 for FP16)

= 2 × 80 × 32 × 128 × 2048 × 8 × 2
= ~171 GB per batch!
```

**This is a huge problem:**

```
GPU Memory (A100): 80 GB
KV Cache needed: 171 GB for single batch!

Result:
- Can't fit in memory
- Limits batch size to 1 or less
- Reduces throughput
- Increases cost per token
```

### Why KV Cache is Needed

**During generation:**

```
Step 1: Generate token 1
Process: "The" → compute K₁, V₁

Step 2: Generate token 2
Process: "The cat" → recompute K₁, V₁ + compute K₂, V₂

Step 3: Generate token 3
Process: "The cat sat" → recompute K₁, K₂, V₁, V₂ + compute K₃, V₃
```

**Problem:** Recomputing Keys and Values for past tokens is wasteful!

**Solution: KV Caching:**

```
Step 1: Compute and cache K₁, V₁
Step 2: Use cached K₁, V₁ + compute and cache K₂, V₂
Step 3: Use cached K₁, K₂, V₁, V₂ + compute K₃, V₃

Speedup: 10-20× faster generation
Cost: Memory to store all cached K, V
```

### The Trade-off

```
Without KV Cache:
✓ Low memory
✗ Very slow generation (recompute everything)

With KV Cache (MHA):
✓ Fast generation
✗ High memory (limits batch size)

Ideal:
✓ Fast generation
✓ Lower memory
✓ Maintain quality

→ This is what GQA achieves!
```

---

## Evolution of Attention Mechanisms

### 1. Multi-Head Attention (MHA) - Baseline

**Architecture:**

```
num_heads = h (e.g., 32)

For each head i:
  Q_i = X W_i^Q  ∈ ℝ^(n×d_k)
  K_i = X W_i^K  ∈ ℝ^(n×d_k)
  V_i = X W_i^V  ∈ ℝ^(n×d_k)

Total KV parameters: 2 × h × d_k × d_model
Total KV cache: 2 × h × d_k × seq_len
```

**Properties:**
- ✓ Best quality (full expressiveness)
- ✓ Each head specializes independently
- ✗ Highest memory usage
- ✗ Most parameters

### 2. Multi-Query Attention (MQA) - Maximum Sharing

**Architecture:**

```
num_heads = h (e.g., 32)

For each head i:
  Q_i = X W_i^Q  ∈ ℝ^(n×d_k)

Shared across all heads:
  K = X W^K  ∈ ℝ^(n×d_k)
  V = X W^V  ∈ ℝ^(n×d_k)

Total KV parameters: 2 × d_k × d_model
Total KV cache: 2 × d_k × seq_len

Reduction: h× fewer KV parameters and cache
```

**Properties:**
- ✓ Minimal memory (h× reduction)
- ✓ Fast inference
- △ Quality degradation (5-10%)
- ✗ Less expressive

### 3. Grouped Query Attention (GQA) - Middle Ground

**Architecture:**

```
num_heads = h (e.g., 32)
num_kv_heads = g (e.g., 8)
group_size = h / g (e.g., 4)

For each head i:
  Q_i = X W_i^Q  ∈ ℝ^(n×d_k)

For each group j (j = 1 to g):
  K_j = X W_j^K  ∈ ℝ^(n×d_k)
  V_j = X W_j^V  ∈ ℝ^(n×d_k)

Heads in group j share K_j, V_j

Total KV parameters: 2 × g × d_k × d_model
Total KV cache: 2 × g × d_k × seq_len

Reduction: (h/g)× fewer KV parameters and cache
```

**Properties:**
- ✓ Good quality (close to MHA)
- ✓ Memory efficient (h/g× reduction)
- ✓ Configurable trade-off
- ✓ Best of both worlds

---

## Grouped Query Attention Explained

### Core Concept

**Key Idea:** Share Keys and Values across groups of Query heads

```
Example: 8 Query heads, 4 KV heads (2 queries per KV)

Group 1: Q₁, Q₂ → share K₁, V₁
Group 2: Q₃, Q₄ → share K₂, V₂
Group 3: Q₅, Q₆ → share K₃, V₃
Group 4: Q₇, Q₈ → share K₄, V₄
```

### Grouping Strategy

**Configuration parameters:**

```
num_query_heads (h): Total number of query heads
num_kv_heads (g): Number of key-value heads
group_size (s): h / g queries per KV head

Constraint: h must be divisible by g
```

**Common configurations:**

```
Model Size    h (queries)  g (kv)  group_size  Reduction
-----------------------------------------------------------
7B params     32           4       8           8×
13B params    40           5       8           8×
70B params    64           8       8           8×
LLaMA-2-7B    32           8       4           4×
LLaMA-2-70B   64           8       8           8×
```

### Information Flow

**Multi-Head Attention (MHA):**

```
Input X
  ↓
┌─────────────────────────────────┐
│ Head 1: Q₁ ← K₁ ← V₁           │
│ Head 2: Q₂ ← K₂ ← V₂           │
│ Head 3: Q₃ ← K₃ ← V₃           │
│ ...                             │
│ Head h: Qₕ ← Kₕ ← Vₕ           │
└─────────────────────────────────┘
  ↓
Concat & Project
  ↓
Output

Independent K, V for each head
```

**Grouped Query Attention (GQA):**

```
Input X
  ↓
┌─────────────────────────────────┐
│ Group 1: Q₁, Q₂ ← K₁ ← V₁      │
│ Group 2: Q₃, Q₄ ← K₂ ← V₂      │
│ Group 3: Q₅, Q₆ ← K₃ ← V₃      │
│ ...                             │
│ Group g: Q_{h-1}, Qₕ ← Kᵍ ← Vᵍ │
└─────────────────────────────────┘
  ↓
Concat & Project
  ↓
Output

Shared K, V within each group
```

---

## Mathematical Foundation

### Standard Multi-Head Attention (Review)

```
For each head i (i = 1 to h):

Q_i = X W_i^Q    where W_i^Q ∈ ℝ^(d_model × d_k)
K_i = X W_i^K    where W_i^K ∈ ℝ^(d_model × d_k)
V_i = X W_i^V    where W_i^V ∈ ℝ^(d_model × d_k)

head_i = Attention(Q_i, K_i, V_i)
       = softmax(Q_i K_i^T / √d_k) V_i

Output = Concat(head_1, ..., head_h) W^O
```

### Grouped Query Attention Formula

```
Given:
- num_query_heads = h
- num_kv_heads = g
- group_size = s = h / g

For each query head i (i = 1 to h):
  Q_i = X W_i^Q    where W_i^Q ∈ ℝ^(d_model × d_k)

For each KV head j (j = 1 to g):
  K_j = X W_j^K    where W_j^K ∈ ℝ^(d_model × d_k)
  V_j = X W_j^V    where W_j^V ∈ ℝ^(d_model × d_k)

Group assignment:
  Query head i uses KV head j where j = ⌊i / s⌋

Attention computation:
  For query head i:
    j = ⌊i / s⌋  (which KV group)
    head_i = Attention(Q_i, K_j, V_j)
           = softmax(Q_i K_j^T / √d_k) V_j

Output = Concat(head_1, ..., head_h) W^O
```

### Detailed Mathematical Steps

**Step 1: Input Processing**

```
Input: X ∈ ℝ^(n×d_model)
where:
  n = sequence length
  d_model = model dimension
```

**Step 2: Query Projections**

```
For i = 1 to h:
  Q_i = X W_i^Q ∈ ℝ^(n×d_k)

Total Q parameters: h × d_model × d_k
```

**Step 3: Key-Value Projections**

```
For j = 1 to g:
  K_j = X W_j^K ∈ ℝ^(n×d_k)
  V_j = X W_j^V ∈ ℝ^(n×d_k)

Total KV parameters: 2 × g × d_model × d_k
```

**Step 4: Group Assignment**

```
Define mapping: query_head → kv_head

group(i) = ⌊(i-1) / s⌋ + 1

Example (h=8, g=4, s=2):
Q₁, Q₂ → K₁, V₁  (group 1)
Q₃, Q₄ → K₂, V₂  (group 2)
Q₅, Q₆ → K₃, V₃  (group 3)
Q₇, Q₈ → K₄, V₄  (group 4)
```

**Step 5: Attention Computation**

```
For each query head i:
  j = group(i)
  
  # Compute attention scores
  S_i = Q_i K_j^T / √d_k ∈ ℝ^(n×n)
  
  # Apply softmax
  A_i = softmax(S_i) ∈ ℝ^(n×n)
  
  # Apply to values
  head_i = A_i V_j ∈ ℝ^(n×d_k)
```

**Step 6: Concatenation and Projection**

```
Concat all heads:
  H = [head_1 | head_2 | ... | head_h] ∈ ℝ^(n×(h×d_k))
  
Output projection:
  Output = H W^O ∈ ℝ^(n×d_model)
  
where W^O ∈ ℝ^((h×d_k)×d_model)
```

### Parameter Count Analysis

```
Multi-Head Attention (h heads):
  Q params: h × d_model × d_k
  K params: h × d_model × d_k
  V params: h × d_model × d_k
  O params: (h × d_k) × d_model
  
  Total: 4 × h × d_model × d_k

Grouped Query Attention (h query heads, g KV heads):
  Q params: h × d_model × d_k
  K params: g × d_model × d_k
  V params: g × d_model × d_k
  O params: (h × d_k) × d_model
  
  Total: (2h + 2g) × d_model × d_k

Reduction:
  (4h - 2h - 2g) / 4h = (2h - 2g) / 4h = (h - g) / 2h
  
Example (h=32, g=8):
  Reduction = (32 - 8) / 64 = 24/64 = 37.5%
```

---

## Step-by-Step Walkthrough

### Example 1: Small GQA (8 queries, 4 KV heads)

**Configuration:**

```
num_query_heads = 8
num_kv_heads = 4
group_size = 2
d_k = 4 (simplified)
sequence_length = 3
```

**Input:**

```
X = [[1, 0, 1, 0],    ← token 1
     [0, 1, 0, 1],    ← token 2
     [1, 1, 0, 0]]    ← token 3

Shape: (3, 4)
```

#### Step 1: Create Query Projections

```
For simplicity, use identity-like projections:

Q₁ = X  (head 1)
Q₂ = X  (head 2)
Q₃ = X  (head 3)
Q₄ = X  (head 4)
Q₅ = X  (head 5)
Q₆ = X  (head 6)
Q₇ = X  (head 7)
Q₈ = X  (head 8)

Each Q_i ∈ ℝ^(3×4)
```

#### Step 2: Create Key-Value Projections

```
K₁ = X     (for group 1: Q₁, Q₂)
K₂ = X*2   (for group 2: Q₃, Q₄)
K₃ = X     (for group 3: Q₅, Q₆)
K₄ = X*2   (for group 4: Q₇, Q₈)

V₁ = X     (for group 1)
V₂ = X     (for group 2)
V₃ = X*2   (for group 3)
V₄ = X*2   (for group 4)

Only 4 K and 4 V (not 8 of each!)
```

#### Step 3: Group Assignment

```
Group 1: Heads 1, 2 → use K₁, V₁
Group 2: Heads 3, 4 → use K₂, V₂
Group 3: Heads 5, 6 → use K₃, V₃
Group 4: Heads 7, 8 → use K₄, V₄
```

#### Step 4: Compute Attention for Head 1

```
Head 1 (in group 1):
  Q₁ = [[1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]]
  
  K₁ = [[1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]]
  
  V₁ = [[1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]]

Compute Q₁K₁^T:
  Q₁K₁^T = [[1,0,1,0],      [[1,0,1],
            [0,1,0,1],   ×   [0,1,1],
            [1,1,0,0]]       [1,0,0],
                             [0,1,0]]
  
         = [[2, 0, 1],
            [0, 2, 1],
            [1, 1, 2]]

Scale by √4 = 2:
  Scaled = [[1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0]]

Softmax (row-wise):
  Row 1: softmax([1.0, 0.0, 0.5])
       = [0.432, 0.158, 0.289]
  
  Row 2: softmax([0.0, 1.0, 0.5])
       = [0.158, 0.432, 0.289]
  
  Row 3: softmax([0.5, 0.5, 1.0])
       = [0.244, 0.244, 0.512]

Attention weights:
  A₁ = [[0.432, 0.158, 0.289],
        [0.158, 0.432, 0.289],
        [0.244, 0.244, 0.512]]

Apply to V₁:
  head₁ = A₁ V₁
  
  For position 1:
  = 0.432×[1,0,1,0] + 0.158×[0,1,0,1] + 0.289×[1,1,0,0]
  = [0.432,0,0.432,0] + [0,0.158,0,0.158] + [0.289,0.289,0,0]
  = [0.721, 0.447, 0.432, 0.158]

  Similarly for positions 2, 3...
```

#### Step 5: Compute Attention for Head 2

```
Head 2 (also in group 1):
  Uses same K₁, V₁ as head 1!
  
  Q₂ might be different projection of X
  But still uses K₁, V₁
  
  head₂ = softmax(Q₂K₁^T / √d_k) V₁
```

#### Step 6: Concatenate All Heads

```
All 8 heads computed:
  head₁, head₂ (used K₁, V₁)
  head₃, head₄ (used K₂, V₂)
  head₅, head₆ (used K₃, V₃)
  head₇, head₈ (used K₄, V₄)

Concatenate:
  H = [head₁ | head₂ | head₃ | head₄ | head₅ | head₆ | head₇ | head₈]
  
  Shape: (3, 8×4) = (3, 32)
```

#### Step 7: Output Projection

```
Output = H W^O

where W^O ∈ ℝ^(32×d_model)

Final output ∈ ℝ^(3×d_model)
```

### Key Observations

```
1. Memory Savings:
   - MHA would need: 8 K + 8 V = 16 matrices
   - GQA needs: 4 K + 4 V = 8 matrices
   - Reduction: 50%

2. Computation:
   - Still compute attention for all 8 heads
   - But reuse K, V across heads in same group
   - Slight computation savings

3. Expressiveness:
   - Less expressive than MHA (shared K, V)
   - More expressive than MQA (multiple K, V)
   - Good quality-efficiency trade-off
```

---

## Detailed Examples

### Example 2: Memory Calculation

**Model Configuration:**

```
d_model = 4096
num_layers = 80
num_query_heads = 64
num_kv_heads = 8
head_dim = d_model / num_query_heads = 64
sequence_length = 8192
batch_size = 4
dtype = FP16 (2 bytes)
```

**Multi-Head Attention KV Cache:**

```
Memory = 2 (K and V)
       × num_layers
       × num_heads
       × head_dim
       × seq_len
       × batch_size
       × bytes

     = 2 × 80 × 64 × 64 × 8192 × 4 × 2
     = 2 × 80 × 64 × 64 × 8192 × 8
     = 2,147,483,648 bytes
     = ~2.15 GB per batch... wait, let me recalculate

Actually:
     = 2 × 80 × 64 × 64 × 8192 × 4 × 2 bytes
     = 4,294,967,296 bytes
     = 4.29 GB... still seems off

Let me recalculate properly:
     = 2 (K,V) × 80 (layers) × 64 (heads) × 64 (dim) × 8192 (seq) × 4 (batch) × 2 (bytes)
     = 2 × 80 × 64 × 64 × 8192 × 4 × 2
     = 2 × 80 × 64 × 64 × 65536
     = 2 × 80 × 16,777,216
     = 2,684,354,560 bytes
     = ~2.68 GB

No wait, 8192 × 4 = 32768, not 65536. Let me be more careful:

Memory = 2 × 80 × 64 × 64 × 8192 × 4 × 2
       = Let me compute step by step:
       = 2 × 80 = 160
       = 160 × 64 = 10,240
       = 10,240 × 64 = 655,360
       = 655,360 × 8,192 = 5,368,709,120
       = 5,368,709,120 × 4 = 21,474,836,480
       = 21,474,836,480 × 2 = 42,949,672,960 bytes
       = ~43 GB for batch of 4
       = ~10.7 GB per sample
```

**Grouped Query Attention KV Cache:**

```
Memory = 2 (K and V)
       × num_layers
       × num_kv_heads  ← Changed!
       × head_dim
       × seq_len
       × batch_size
       × bytes

     = 2 × 80 × 8 × 64 × 8192 × 4 × 2
     = 160 × 8 × 64 × 8192 × 4 × 2
     = 1,280 × 64 × 8192 × 4 × 2
     = 81,920 × 8192 × 4 × 2
     = 671,088,640 × 4 × 2
     = 5,368,709,120 bytes
     = ~5.37 GB for batch of 4
     = ~1.34 GB per sample
```

**Savings:**

```
Reduction = (MHA - GQA) / MHA
          = (43 - 5.37) / 43
          = 87.5%

With GQA:
- 8× reduction in KV cache
- Can fit 8× larger batch
- Or support 8× longer sequences
```

### Example 3: Quality vs Efficiency Trade-off

**Experimental Results (typical):**

```
Configuration Comparison on 7B model:

Multi-Head (h=32, g=32):
  Perplexity: 8.45
  KV Cache: 100%
  Training Speed: 1.0×
  Inference Speed: 1.0×

Grouped (h=32, g=8):
  Perplexity: 8.52  (+0.07, negligible)
  KV Cache: 25%
  Training Speed: 1.05× (slightly faster)
  Inference Speed: 1.3× (faster)

Grouped (h=32, g=4):
  Perplexity: 8.61  (+0.16, small loss)
  KV Cache: 12.5%
  Training Speed: 1.08×
  Inference Speed: 1.5×

Multi-Query (h=32, g=1):
  Perplexity: 9.12  (+0.67, noticeable loss)
  KV Cache: 3.125%
  Training Speed: 1.1×
  Inference Speed: 1.8×

Sweet spot: g = h/4 or h/8
- Minimal quality loss
- Significant memory savings
- Good speedup
```

---

## Memory and Computation Analysis

### Training Time Complexity

```
Multi-Head Attention:
  Q computation: O(n × d_model × h × d_k)
  K computation: O(n × d_model × h × d_k)
  V computation: O(n × d_model × h × d_k)
  Attention: O(n² × h × d_k)
  Output: O(n × h × d_k × d_model)

Grouped Query Attention:
  Q computation: O(n × d_model × h × d_k)
  K computation: O(n × d_model × g × d_k)  ← Reduced
  V computation: O(n × d_model × g × d_k)  ← Reduced
  Attention: O(n² × h × d_k)  (same, all Q heads compute)
  Output: O(n × h × d_k × d_model)

Training speedup: ~5-10% (modest)
Main benefit is in inference!
```

### Inference Time Analysis

**Without KV Cache:**

```
Both MHA and GQA: Same complexity
- Must recompute everything each step
- O(n × d_model × d_k) per step
```

**With KV Cache:**

```
Multi-Head Attention:
  Store: h KV heads × seq_len × d_k
  Memory: O(h × n × d_k)

Grouped Query Attention:
  Store: g KV heads × seq_len × d_k
  Memory: O(g × n × d_k)

Memory reduction: h/g ×

For typical h=32, g=8:
  Reduction: 4×
```

### Bandwidth Analysis

```
During each generation step:

Multi-Head Attention:
  Load from KV cache: 2 × h × seq_len × d_k × bytes
  
  For h=64, seq_len=2048, d_k=64, FP16:
  = 2 × 64 × 2048 × 64 × 2
  = 33,554,432 bytes
  = ~33.5 MB per step

Grouped Query Attention (g=8):
  Load from KV cache: 2 × g × seq_len × d_k × bytes
  
  = 2 × 8 × 2048 × 64 × 2
  = 4,194,304 bytes
  = ~4.2 MB per step

Bandwidth reduction: 8×
- Less memory traffic
- Faster on memory-bound GPUs
```

### Parameter Count

```
Example: d_model = 4096, h = 32, g = 8, d_k = 128

Multi-Head Attention:
  W^Q: 32 × 4096 × 128 = 16,777,216
  W^K: 32 × 4096 × 128 = 16,777,216
  W^V: 32 × 4096 × 128 = 16,777,216
  W^O: (32×128) × 4096 = 16,777,216
  Total: 67,108,864 (~67M)

Grouped Query Attention:
  W^Q: 32 × 4096 × 128 = 16,777,216
  W^K: 8 × 4096 × 128 = 4,194,304     ← Reduced
  W^V: 8 × 4096 × 128 = 4,194,304     ← Reduced
  W^O: (32×128) × 4096 = 16,777,216
  Total: 41,943,040 (~42M)

Parameter reduction: 37.5%
Model size reduction: ~10-15% overall (attention is part of model)
```

---

## Comparison with Other Variants

### Feature Comparison Table

| Feature | MHA | GQA | MQA |
|---------|-----|-----|-----|
| Query heads | h | h | h |
| KV heads | h | g (g < h) | 1 |
| KV cache size | 100% | 100%/s | 100%/h |
| Parameters | 100% | ~60-80% | ~50% |
| Quality | Best | ~99% | ~90-95% |
| Training speed | 1× | 1.05-1.1× | 1.1-1.15× |
| Inference speed | 1× | 1.2-1.5× | 1.5-2× |
| Memory | Highest | Medium | Lowest |

Where s = h/g (group size)

### When to Use Each

```
Multi-Head Attention (MHA):
✓ Use when:
  - Memory is not constrained
  - Maximum quality needed
  - Research/experimentation
  - Small models (<3B params)
  
✗ Avoid when:
  - Limited GPU memory
  - Need high throughput
  - Large batch inference

Grouped Query Attention (GQA):
✓ Use when:
  - Balancing quality and efficiency
  - Production deployments
  - Medium-large models (7B-70B)
  - Batch inference important
  
✗ Avoid when:
  - Extreme memory constraints (use MQA)
  - Tiny models (overhead not worth it)

Multi-Query Attention (MQA):
✓ Use when:
  - Extreme memory constraints
  - Maximum throughput critical
  - Quality loss acceptable
  - Very long context (100K+ tokens)
  
✗ Avoid when:
  - Quality is critical
  - Tasks requiring nuanced understanding
  - Small batch sizes (benefits minimal)
```

### Empirical Quality Comparison

```
Task: Language Modeling (Perplexity, lower is better)

Model: 7B parameters

Configuration        Perplexity    Relative
------------------------------------------------
MHA (h=32, g=32)     8.45          100%
GQA (h=32, g=16)     8.47          99.8%
GQA (h=32, g=8)      8.52          99.2%
GQA (h=32, g=4)      8.61          98.1%
MQA (h=32, g=1)      9.12          92.7%

Observation:
- GQA with g=8 loses only 0.8% quality
- But saves 75% KV cache memory
- Sweet spot for most applications
```

---

## Implementation Details

### Key Implementation Considerations

#### 1. Group Assignment

```python
def assign_query_to_kv_group(query_head_idx, num_query_heads, num_kv_heads):
    """
    Determine which KV head a query head should use.
    
    Args:
        query_head_idx: Index of query head (0 to num_query_heads-1)
        num_query_heads: Total number of query heads
        num_kv_heads: Total number of KV heads
    
    Returns:
        kv_head_idx: Index of KV head to use
    """
    group_size = num_query_heads // num_kv_heads
    kv_head_idx = query_head_idx // group_size
    return kv_head_idx

# Example: 8 query heads, 4 KV heads
# Q0, Q1 → K0 (group 0)
# Q2, Q3 → K1 (group 1)
# Q4, Q5 → K2 (group 2)
# Q6, Q7 → K3 (group 3)
```

#### 2. Efficient Tensor Operations

```python
# Instead of looping over heads, use broadcasting

# Query heads: [batch, num_q_heads, seq_len, head_dim]
# KV heads: [batch, num_kv_heads, seq_len, head_dim]

# Expand KV heads to match query heads
# [batch, num_kv_heads, seq_len, head_dim]
# → [batch, num_kv_heads, 1, seq_len, head_dim]
# → [batch, num_kv_heads, group_size, seq_len, head_dim]
# → [batch, num_q_heads, seq_len, head_dim]

K_expanded = K.unsqueeze(2).expand(
    batch, num_kv_heads, group_size, seq_len, head_dim
).reshape(batch, num_q_heads, seq_len, head_dim)
```

#### 3. Memory Layout

```
Optimal memory layout for inference:

KV Cache structure:
  - Shape: [num_kv_heads, batch, seq_len, head_dim]
  - Contiguous in memory
  - Efficient for repeated access

Query projection:
  - Shape: [batch, num_q_heads, seq_len, head_dim]
  - Compute on-the-fly
  - Don't cache (changes each step)
```

#### 4. Backward Pass Considerations

```
During training, gradients must be accumulated:

For Multi-Head:
  ∂L/∂K_i from only head i

For Grouped Query:
  ∂L/∂K_j from all heads in group j
  
  ∂L/∂K_j = Σ (i in group j) ∂L/∂head_i × ∂head_i/∂K_j

Requires gradient accumulation across group
```

---

## Practical Applications

### Application 1: LLaMA 2 Models

```
LLaMA 2 uses GQA across all model sizes:

LLaMA-2-7B:
  num_query_heads = 32
  num_kv_heads = 8
  group_size = 4
  KV reduction: 4×

LLaMA-2-13B:
  num_query_heads = 40
  num_kv_heads = 8
  group_size = 5
  KV reduction: 5×

LLaMA-2-70B:
  num_query_heads = 64
  num_kv_heads = 8
  group_size = 8
  KV reduction: 8×

Benefit:
- Enables larger batch sizes
- Faster inference
- Better throughput
- Minimal quality loss
```

### Application 2: Long Context Models

```
For 100K+ token context:

Without GQA (MHA):
  32 heads × 100K tokens × 128 dim × 2 bytes
  = 819 MB per layer per sample
  × 80 layers = 65.5 GB per sample!

With GQA (g=8):
  8 heads × 100K tokens × 128 dim × 2 bytes
  = 205 MB per layer per sample
  × 80 layers = 16.4 GB per sample

Can fit 4× more samples in memory
Enables practical long-context inference
```

### Application 3: Edge Deployment

```
Deploying 7B model on device:

Available memory: 8 GB

MHA (32 heads):
  Model: 14 GB
  KV cache (4K ctx): 2 GB
  Total: 16 GB → Doesn't fit!

GQA (8 KV heads):
  Model: 13 GB (saved 1 GB on parameters)
  KV cache (4K ctx): 0.5 GB
  Total: 13.5 GB → Still doesn't fit :(

GQA + Quantization (INT4):
  Model: 4 GB
  KV cache: 0.5 GB
  Total: 4.5 GB → Fits with room to spare!

GQA makes edge deployment feasible
```

---

## Advanced Topics

### 1. Uptraining MHA to GQA

**Problem:** Convert pre-trained MHA model to GQA

**Method: Mean Pooling**

```
Given trained MHA with h KV heads
Want GQA with g KV heads (g < h)

For each KV group j:
  Heads in group: [j×s, j×s+1, ..., j×s+s-1]
  
  K_j^new = mean(K_{j×s}, K_{j×s+1}, ..., K_{j×s+s-1})
  V_j^new = mean(V_{j×s}, V_{j×s+1}, ..., V_{j×s+s-1})

Then fine-tune for small number of steps

Result:
- Preserves most knowledge
- Fast convergence (few steps)
- ~1-2% perplexity degradation
```

### 2. Adaptive Grouping

**Idea:** Different layers use different group sizes

```
Strategy:
  Early layers: Smaller groups (more KV heads)
    - Capture low-level patterns
    - Need more diversity
  
  Middle layers: Medium groups
    - Balanced representation
  
  Late layers: Larger groups (fewer KV heads)
    - High-level abstract features
    - Can share more

Example (32 query heads):
  Layers 0-20: g = 16 (group size 2)
  Layers 21-60: g = 8 (group size 4)
  Layers 61-80: g = 4 (group size 8)

Benefit:
- Better quality-memory trade-off
- Tailored to layer needs
```

### 3. Dynamic Grouping

**Idea:** Adjust grouping based on input

```python
def dynamic_grouping(x, context_length):
    """
    Use more KV heads for short contexts
    Fewer KV heads for long contexts
    """
    if context_length < 1024:
        num_kv_heads = 16  # More quality
    elif context_length < 4096:
        num_kv_heads = 8   # Balanced
    else:
        num_kv_heads = 4   # More efficiency
    
    return apply_gqa(x, num_kv_heads)
```

### 4. Grouped Query Attention with Mixture of Experts

**Combination:** GQA + MoE

```
Benefits:
- GQA reduces KV cache
- MoE reduces computation
- Combined: 10× efficiency gain

Architecture:
  For each group:
    Expert 1: K₁, V₁
    Expert 2: K₂, V₂
    ...
  
  Router selects expert per token
  
Memory: O(g × num_experts × seq_len)
Computation: O(g × active_experts × seq_len)
```

### 5. Position-Aware Grouping

**Idea:** Group assignment based on position

```
Different positions may need different KV heads

Example:
  Positions 0-512: Use KV heads 0-3
  Positions 513-1024: Use KV heads 4-7
  ...

Benefit:
- Capture position-specific patterns
- Better for long sequences
- Trade-off: More complex
```

---

## Best Practices

### 1. Choosing Group Size

```
Guidelines:

Small models (<3B):
  Use MHA or GQA with small groups (g = h/2)
  Quality matters more than memory

Medium models (3B-30B):
  GQA with g = h/4 or h/8
  Sweet spot for quality-efficiency

Large models (30B+):
  GQA with g = h/8 or h/16
  Memory savings critical

Very large models (100B+):
  Consider MQA for maximum efficiency
  Or GQA with very large groups
```

### 2. Training from Scratch vs Uptraining

```
Training from scratch with GQA:
✓ Optimal quality for given group size
✓ No conversion needed
✗ Requires full training compute

Uptraining MHA → GQA:
✓ Leverages existing MHA checkpoint
✓ Fast (few additional steps)
△ Slight quality loss vs scratch
✓ Good for experimentation
```

### 3. Hyperparameter Selection

```
Key hyperparameters:

num_kv_heads:
  - Must divide num_query_heads evenly
  - Common: 8, 16 for large models
  - Start with h/8, adjust based on memory

Learning rate for uptraining:
  - Lower than original training
  - Typical: 1e-5 to 1e-4
  - Warm up for stability

Steps for uptraining:
  - 5-10% of original training
  - Monitor perplexity convergence
  - Early stopping when stable
```

---

## Summary

### Key Concepts

```
1. Core Idea:
   - Share K, V across groups of Q heads
   - Reduces memory while maintaining quality
   - Middle ground between MHA and MQA

2. Configuration:
   - h query heads, g KV heads
   - group_size = h / g
   - Typical: g = h/8 for large models

3. Benefits:
   - (h/g)× reduction in KV cache
   - ~5-10% faster inference
   - <1% quality loss (with proper g)
   - Enables larger batch sizes

4. Trade-offs:
   - Slightly less expressive than MHA
   - More complex than standard MHA
   - Requires careful group size selection
```

### Mathematical Summary

```
Standard MHA:
  head_i = Attention(Q_i, K_i, V_i) for i = 1 to h

Grouped Query Attention:
  head_i = Attention(Q_i, K_j, V_j)
  where j = ⌊i / (h/g)⌋

Memory reduction:
  MHA: 2 × h × d_k × seq_len
  GQA: 2 × g × d_k × seq_len
  Ratio: g/h
```

### When to Use GQA

```
✓ Use GQA when:
  - Memory is constrained
  - Serving large models
  - Long context needed
  - Batch inference important
  - Quality loss <1% acceptable

✗ Don't use GQA when:
  - Small models (<1B)
  - Memory abundant
  - Research requiring MHA baseline
  - Extreme quality requirements
```

### Impact on AI Industry

```
GQA has become standard for large models:
- LLaMA 2 (Meta)
- Mistral
- Many open-source models

Enables:
- Larger models on same hardware
- Longer contexts (100K+)
- Better inference economics
- Democratization of large models

Future:
- Standard in all large models
- Further optimizations (adaptive, dynamic)
- Combined with other techniques (MoE, quantization)
```

---

*Last Updated: February 2026*
*Document Version: 1.0*
