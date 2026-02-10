# Key-Value (KV) Cache: Complete Mathematical Explanation

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem: Autoregressive Generation](#the-problem)
3. [Understanding Attention Mechanism](#understanding-attention)
4. [The Inefficiency Without Cache](#the-inefficiency)
5. [The KV Cache Solution](#the-solution)
6. [Mathematical Derivation](#mathematical-derivation)
7. [Step-by-Step Example](#step-by-step-example)
8. [Memory Analysis](#memory-analysis)
9. [Implementation Details](#implementation-details)
10. [Practical Impact](#practical-impact)

---

## Introduction

**Key-Value (KV) Cache** is an optimization technique used in transformer-based language models during **autoregressive text generation**. It dramatically speeds up inference by avoiding redundant computations.

**Main Idea:** Store previously computed Key (K) and Value (V) matrices so we don't have to recompute them for every new token.

**Impact:**
- **Speed:** 2-10× faster text generation
- **Efficiency:** Reduces computation from O(n²) to O(n) per new token
- **Trade-off:** Uses more memory to store the cache

---

## The Problem: Autoregressive Generation

### What is Autoregressive Generation?

Autoregressive generation means generating text **one token at a time**, where each new token depends on all previous tokens.

**Example:**
```
Prompt: "The cat sat on the"

Step 1: Generate "mat"     → "The cat sat on the mat"
Step 2: Generate "and"     → "The cat sat on the mat and"
Step 3: Generate "fell"    → "The cat sat on the mat and fell"
Step 4: Generate "asleep"  → "The cat sat on the mat and fell asleep"
...
```

### The Challenge

At each step, the model must:
1. Look at ALL previous tokens
2. Understand their relationships (via attention)
3. Generate the next token

**Problem:** Without caching, we recompute attention for ALL previous tokens at EVERY step!

---

## Understanding Attention Mechanism

### Self-Attention Formula

For a sequence of tokens, self-attention is computed as:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What information do I contain?"
- **V (Value):** "What should I output?"
- **d_k:** Dimension of key vectors (for scaling)

### Matrix Dimensions

For a sequence of length `n` with embedding dimension `d`:

```
Input:  X       [n × d]

Q = X·W_Q       [n × d_k]
K = X·W_K       [n × d_k]
V = X·W_V       [n × d_v]

Attention = softmax(Q·K^T / √d_k)·V
          = softmax([n×d_k]·[d_k×n] / √d_k)·[n×d_v]
          = softmax([n×n] / √d_k)·[n×d_v]
          = [n×n]·[n×d_v]
          = [n×d_v]
```

### Step-by-Step Attention Computation

**Step 1: Compute Attention Scores**
```
Scores = Q·K^T / √d_k
```
This gives an `[n × n]` matrix where `Scores[i,j]` represents how much token `i` should attend to token `j`.

**Step 2: Apply Softmax**
```
Weights = softmax(Scores)  [n × n]
```
Each row becomes a probability distribution (sums to 1).

**Step 3: Weighted Sum of Values**
```
Output = Weights·V  [n × n]·[n × d_v] = [n × d_v]
```
Each token's output is a weighted combination of all value vectors.

---

## The Inefficiency Without Cache

### Scenario: Generating 3 New Tokens

Let's say we have:
- Initial prompt: 5 tokens
- We want to generate 3 more tokens

**Step 1: Generate token 6**
```
Tokens: [1, 2, 3, 4, 5, ?]
Process ALL 5 tokens through attention
Compute Q, K, V for all 5 tokens
Generate token 6
```

**Step 2: Generate token 7**
```
Tokens: [1, 2, 3, 4, 5, 6, ?]
Process ALL 6 tokens through attention  ← Recomputing for tokens 1-5!
Compute Q, K, V for all 6 tokens        ← Wasteful!
Generate token 7
```

**Step 3: Generate token 8**
```
Tokens: [1, 2, 3, 4, 5, 6, 7, ?]
Process ALL 7 tokens through attention  ← Recomputing for tokens 1-6!
Compute Q, K, V for all 7 tokens        ← Very wasteful!
Generate token 8
```

### Computational Complexity

For generating `m` new tokens starting from a prompt of length `n`:

**Total computations:**
```
Step 1: Process n tokens
Step 2: Process (n+1) tokens
Step 3: Process (n+2) tokens
...
Step m: Process (n+m-1) tokens

Total: n + (n+1) + (n+2) + ... + (n+m-1)
     = m·n + (0+1+2+...+(m-1))
     = m·n + m(m-1)/2
     = O(m·n + m²)
```

For long sequences, this becomes **very expensive**!

---

## The KV Cache Solution

### Key Insight

Notice that in the attention formula:
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
```

When we generate a new token:
- We need **new Q** for the new token (to query what it should attend to)
- But the **K and V** for all previous tokens **don't change**!

**Idea:** Store K and V from previous computations and reuse them!

### How KV Cache Works

**Step 1: Generate token 6**
```
Tokens: [1, 2, 3, 4, 5, ?]
Compute Q, K, V for tokens 1-5
Cache K and V
Generate token 6
```

**Step 2: Generate token 7**
```
Tokens: [1, 2, 3, 4, 5, 6, ?]
Retrieve cached K, V for tokens 1-5  ← No recomputation!
Compute only new Q, K, V for token 6
Append new K, V to cache
Generate token 7
```

**Step 3: Generate token 8**
```
Tokens: [1, 2, 3, 4, 5, 6, 7, ?]
Retrieve cached K, V for tokens 1-6  ← No recomputation!
Compute only new Q, K, V for token 7
Append new K, V to cache
Generate token 8
```

### Computational Complexity with Cache

With KV cache, for each new token we only process **1 token** instead of all previous tokens.

```
Step 1: Process n tokens (build initial cache)
Step 2: Process 1 token
Step 3: Process 1 token
...
Step m: Process 1 token

Total: n + m  = O(n + m)
```

**Speedup:** From O(m·n + m²) to O(n + m) — **massive improvement** for large m!

---

## Mathematical Derivation

### Standard Attention (No Cache)

At step `t` with sequence length `t`:

**Input:**
```
X_t = [x₁, x₂, ..., x_t]  [t × d]
```

**Compute Q, K, V:**
```
Q_t = X_t·W_Q  [t × d_k]
K_t = X_t·W_K  [t × d_k]
V_t = X_t·W_V  [t × d_v]
```

**Attention:**
```
Attention_t = softmax(Q_t·K_t^T / √d_k)·V_t
```

**Problem:** At step t+1, we recompute everything!

### Attention with KV Cache

**Key Observation:** K and V are linear projections of input tokens.

```
K_t = X_t·W_K = [x₁·W_K, x₂·W_K, ..., x_t·W_K]
                 ↑        ↑              ↑
                k₁       k₂             k_t
```

Each row `k_i` depends only on input `x_i`, which doesn't change!

**At step t:**
```
K_t = [k₁, k₂, ..., k_t]
V_t = [v₁, v₂, ..., v_t]
```

**At step t+1:**
```
K_{t+1} = [k₁, k₂, ..., k_t, k_{t+1}]
          └─────────────┘    └──────┘
          From cache!        Compute new

V_{t+1} = [v₁, v₂, ..., v_t, v_{t+1}]
          └─────────────┘    └──────┘
          From cache!        Compute new
```

**Query (Q) is different:**
- At step t, we compute Q for all tokens 1..t
- At step t+1, we only need Q for the new token t+1

### Attention Computation with Cache

**Step t+1:**

1. **Retrieve cached K and V:**
   ```
   K_cached = [k₁, k₂, ..., k_t]  [t × d_k]
   V_cached = [v₁, v₂, ..., v_t]  [t × d_v]
   ```

2. **Compute new Q, K, V for token t+1:**
   ```
   q_{t+1} = x_{t+1}·W_Q  [1 × d_k]
   k_{t+1} = x_{t+1}·W_K  [1 × d_k]
   v_{t+1} = x_{t+1}·W_V  [1 × d_v]
   ```

3. **Concatenate:**
   ```
   K_{t+1} = concat(K_cached, k_{t+1})  [(t+1) × d_k]
   V_{t+1} = concat(V_cached, v_{t+1})  [(t+1) × d_v]
   ```

4. **Compute attention (only for new token):**
   ```
   Scores = q_{t+1}·K_{t+1}^T / √d_k  [1 × (t+1)]
   Weights = softmax(Scores)           [1 × (t+1)]
   Output = Weights·V_{t+1}            [1 × d_v]
   ```

**Key Point:** We only compute attention for the **new token**, not all tokens!

---

## Step-by-Step Example

Let's work through a concrete example with **actual numbers**.

### Setup

```
Vocabulary: {cat, sat, mat, on, the}
Sequence: "the cat sat"
d_model = 4 (embedding dimension)
d_k = d_v = 4 (for simplicity)
```

### Initial State (Prompt Processing)

**Input tokens:** [the, cat]

**Embeddings:**
```
x₁ (the) = [1.0, 0.5, 0.2, 0.1]
x₂ (cat) = [0.5, 1.0, 0.3, 0.2]
```

**Weight matrices (simplified):**
```
W_Q = I₄ (identity, for simplicity)
W_K = I₄
W_V = I₄
```

**Compute Q, K, V:**
```
Q = [x₁·W_Q]  = [1.0, 0.5, 0.2, 0.1]  ← q₁
    [x₂·W_Q]    [0.5, 1.0, 0.3, 0.2]  ← q₂

K = [x₁·W_K]  = [1.0, 0.5, 0.2, 0.1]  ← k₁
    [x₂·W_K]    [0.5, 1.0, 0.3, 0.2]  ← k₂

V = [x₁·W_V]  = [1.0, 0.5, 0.2, 0.1]  ← v₁
    [x₂·W_V]    [0.5, 1.0, 0.3, 0.2]  ← v₂
```

**Attention computation:**
```
Scores = Q·K^T / √4
       = [1.0, 0.5, 0.2, 0.1] · [1.0, 0.5]^T  / 2
         [0.5, 1.0, 0.3, 0.2]   [0.5, 1.0]
                                [0.2, 0.3]
                                [0.1, 0.2]
       
       = [1.0×1.0 + 0.5×0.5 + 0.2×0.2 + 0.1×0.1,  1.0×0.5 + 0.5×1.0 + 0.2×0.3 + 0.1×0.2] / 2
         [0.5×1.0 + 1.0×0.5 + 0.3×0.2 + 0.2×0.1,  0.5×0.5 + 1.0×1.0 + 0.3×0.3 + 0.2×0.2]
       
       = [1.00 + 0.25 + 0.04 + 0.01,  0.50 + 0.50 + 0.06 + 0.02] / 2
         [0.50 + 0.50 + 0.06 + 0.02,  0.25 + 1.00 + 0.09 + 0.04]
       
       = [1.30,  1.08] / 2
         [1.08,  1.38]
       
       = [0.65,  0.54]
         [0.54,  0.69]
```

**Softmax:**
```
Row 1: exp([0.65, 0.54]) / sum = [1.916, 1.716] / 3.632 = [0.527, 0.473]
Row 2: exp([0.54, 0.69]) / sum = [1.716, 1.994] / 3.710 = [0.463, 0.537]

Weights = [0.527,  0.473]
          [0.463,  0.537]
```

**Output:**
```
Output = Weights·V
       = [0.527,  0.473] · [1.0, 0.5, 0.2, 0.1]
         [0.463,  0.537]   [0.5, 1.0, 0.3, 0.2]
       
       = [0.527×1.0 + 0.473×0.5,  0.527×0.5 + 0.473×1.0,  ...]
         [0.463×1.0 + 0.537×0.5,  0.463×0.5 + 0.537×1.0,  ...]
       
       = [0.764, 0.737, 0.247, 0.147]  ← Output for "the"
         [0.732, 0.768, 0.254, 0.153]  ← Output for "cat"
```

**Cache K and V:**
```
K_cache = [1.0, 0.5, 0.2, 0.1]  ← k₁
          [0.5, 1.0, 0.3, 0.2]  ← k₂

V_cache = [1.0, 0.5, 0.2, 0.1]  ← v₁
          [0.5, 1.0, 0.3, 0.2]  ← v₂
```

### Step 1: Generate "sat" (token 3)

**New input:**
```
x₃ (sat) = [0.3, 0.8, 1.0, 0.4]
```

**WITHOUT KV Cache (inefficient):**
```
Would recompute Q, K, V for ALL 3 tokens:
Q = [q₁, q₂, q₃]  [3 × 4]
K = [k₁, k₂, k₃]  [3 × 4]  ← Recomputing k₁, k₂!
V = [v₁, v₂, v₃]  [3 × 4]  ← Recomputing v₁, v₂!
```

**WITH KV Cache (efficient):**

1. **Retrieve cached K, V:**
   ```
   K_cached = [k₁, k₂]  [2 × 4]
   V_cached = [v₁, v₂]  [2 × 4]
   ```

2. **Compute only new q, k, v:**
   ```
   q₃ = x₃·W_Q = [0.3, 0.8, 1.0, 0.4]
   k₃ = x₃·W_K = [0.3, 0.8, 1.0, 0.4]
   v₃ = x₃·W_V = [0.3, 0.8, 1.0, 0.4]
   ```

3. **Append to cache:**
   ```
   K = [k₁]  = [1.0, 0.5, 0.2, 0.1]
       [k₂]    [0.5, 1.0, 0.3, 0.2]
       [k₃]    [0.3, 0.8, 1.0, 0.4]
   
   V = [v₁]  = [1.0, 0.5, 0.2, 0.1]
       [v₂]    [0.5, 1.0, 0.3, 0.2]
       [v₃]    [0.3, 0.8, 1.0, 0.4]
   ```

4. **Compute attention (only for token 3):**
   ```
   Scores = q₃·K^T / √4
          = [0.3, 0.8, 1.0, 0.4] · [1.0, 0.5, 0.3]^T / 2
                                    [0.5, 1.0, 0.8]
                                    [0.2, 0.3, 1.0]
                                    [0.1, 0.2, 0.4]
          
          = [0.3×1.0 + 0.8×0.5 + 1.0×0.2 + 0.4×0.1,  ...]  / 2
          
          = [0.74,  1.12,  1.36] / 2
          = [0.37,  0.56,  0.68]
   
   Weights = softmax([0.37, 0.56, 0.68])
           = [0.281, 0.343, 0.376]
   
   Output₃ = [0.281, 0.343, 0.376] · [v₁]
                                      [v₂]
                                      [v₃]
           = [0.281, 0.343, 0.376] · [1.0, 0.5, 0.2, 0.1]
                                      [0.5, 1.0, 0.3, 0.2]
                                      [0.3, 0.8, 1.0, 0.4]
           = [0.534, 0.784, 0.607, 0.281]
   ```

**Key observation:** We only computed attention for 1 token, not 3!

### Comparison

**Without Cache:**
- Computed Q, K, V for 3 tokens
- Matrix operations: 3×4 matrices

**With Cache:**
- Retrieved K, V for 2 tokens (no computation)
- Computed q, k, v for 1 token
- Matrix operations: 1×4 vectors

**Speedup:** ~3× for this small example, grows with sequence length!

---

## Memory Analysis

### Memory Requirements

For a sequence of length `n` with:
- `d_k` = dimension of keys
- `d_v` = dimension of values
- `L` = number of layers
- `H` = number of attention heads
- `h` = head dimension (d_k = H × h for multi-head)

**KV Cache size per layer:**
```
K cache: [n × H × h] values
V cache: [n × H × h] values
Total: 2 × n × H × h values
```

**For L layers:**
```
Total KV cache: 2 × L × n × H × h values
```

**In bytes (using fp16):**
```
Memory = 2 × L × n × H × h × 2 bytes
       = 4 × L × n × H × h bytes
```

### Example: GPT-3 Style Model

Parameters:
- L = 96 layers
- H = 96 heads
- h = 128 dimensions per head
- n = 2048 tokens (sequence length)
- Data type: fp16 (2 bytes)

**KV Cache size:**
```
Memory = 4 × 96 × 2048 × 96 × 128 bytes
       = 4 × 96 × 2048 × 12,288 bytes
       = 9,663,676,416 bytes
       ≈ 9.66 GB per sequence!
```

For a batch of 8 sequences:
```
Total memory ≈ 77 GB just for KV cache!
```

This is why optimizations like Multi-Query Attention (MQA) are important!

### Memory vs. Computation Trade-off

**Without Cache:**
- Memory: O(L × d) for model weights only
- Computation: O(L × n² × d) per generation step

**With Cache:**
- Memory: O(L × d + L × n × d) 
- Computation: O(L × n × d) per generation step

**Trade-off:** Use O(n × d) more memory to reduce computation from O(n²) to O(n)

For long sequences, the speed gain far outweighs the memory cost!

---

## Implementation Details

### KV Cache Data Structure

The cache is typically stored as a tuple or list of tensors:

```python
# For each layer
past_key_values = [
    (K_layer_0, V_layer_0),  # Layer 0
    (K_layer_1, V_layer_1),  # Layer 1
    ...
    (K_layer_L, V_layer_L),  # Layer L
]

# Where each K and V has shape:
# K: [batch_size, num_heads, seq_len, head_dim]
# V: [batch_size, num_heads, seq_len, head_dim]
```

### Cache Operations

**1. Initialize (first forward pass):**
```python
# Process all tokens in prompt
Q, K, V = compute_qkv(input_ids)
attention_output = attention(Q, K, V)
cache = (K, V)  # Store K and V
```

**2. Update (subsequent steps):**
```python
# Process only new token
q_new, k_new, v_new = compute_qkv(new_token)

# Retrieve and concatenate
K_full = torch.cat([K_cache, k_new], dim=2)  # Concat along sequence dimension
V_full = torch.cat([V_cache, v_new], dim=2)

# Compute attention (q_new attends to all K_full)
attention_output = attention(q_new, K_full, V_full)

# Update cache
cache = (K_full, V_full)
```

**3. Causal Masking:**
```python
# Ensure token i can't attend to tokens j > i
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
scores = scores + mask
```

### Optimization Techniques

**1. Pre-allocation:**
```python
# Allocate max size upfront
max_seq_len = 2048
K_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
V_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)

# Track current position
current_pos = 0

# Update
K_cache[:, :, current_pos] = k_new
V_cache[:, :, current_pos] = v_new
current_pos += 1
```

**2. Dynamic shapes (HuggingFace approach):**
```python
# Cache grows dynamically
if past_key_values is None:
    cache_k = k
    cache_v = v
else:
    cache_k = torch.cat([past_key_values[0], k], dim=2)
    cache_v = torch.cat([past_key_values[1], v], dim=2)
```

**3. Quantization:**
```python
# Store cache in int8 instead of fp16
# Reduces memory by 50%
K_cache_int8 = quantize(K_cache)
V_cache_int8 = quantize(V_cache)

# Dequantize when needed
K_full = dequantize(K_cache_int8)
```

---

## Practical Impact

### Speed Improvements

Real-world benchmarks for text generation:

| Sequence Length | Without Cache | With Cache | Speedup |
|----------------|---------------|------------|---------|
| 128 tokens     | 1.2s          | 0.4s       | 3.0×    |
| 512 tokens     | 4.8s          | 0.6s       | 8.0×    |
| 2048 tokens    | 19.2s         | 1.0s       | 19.2×   |

**Note:** Speedup increases linearly with sequence length!

### Use Cases

**1. Chatbots and Assistants:**
- Multi-turn conversations with long context
- Need to process entire conversation history for each response
- KV cache makes this feasible in real-time

**2. Code Generation:**
- Generate long code files
- Each line depends on all previous lines
- KV cache enables fast incremental generation

**3. Document Continuation:**
- Continue writing from long documents
- Process entire document context once
- Generate new content efficiently

**4. Interactive Applications:**
- Real-time text completion
- Low latency requirements
- KV cache reduces latency from seconds to milliseconds

### Limitations

**1. Memory Constraints:**
- Long sequences require large caches (GBs of memory)
- Batch size limited by available GPU memory
- Solutions: Quantization, Sliding Window, PagedAttention

**2. Static Context:**
- Cache assumes context doesn't change
- If earlier tokens need revision, cache must be invalidated
- Partial solutions: Chunked caching, Recomputation windows

**3. Multi-head Attention:**
- Each head needs its own K and V cache
- 32+ heads = 32× memory per sequence
- Solution: Multi-Query Attention (MQA), Grouped-Query Attention (GQA)

---

## Summary

### Key Concepts

1. **Problem:** Autoregressive generation recomputes attention for all previous tokens at each step → O(n²) complexity

2. **Solution:** Cache K and V matrices from previous steps, only compute new token → O(n) complexity

3. **Trade-off:** Use O(n × d) additional memory to reduce O(n²) computation to O(n)

4. **Impact:** 2-20× speedup depending on sequence length, essential for production inference

### When to Use KV Cache

✅ **Use KV Cache:**
- Text generation (autoregressive)
- Long sequence inference
- Real-time applications
- Production deployment
- Batch inference with similar prompts

❌ **Don't Need KV Cache:**
- Single forward pass (no generation)
- Classification tasks
- Training (where you process full sequences at once)
- Very short sequences (<10 tokens)

### Mathematical Essence

The core mathematical insight:

```
At step t:   Attention = softmax(Q_t · K_t^T) · V_t
At step t+1: Attention = softmax(Q_{t+1} · K_{t+1}^T) · V_{t+1}

Where: K_{t+1} = [K_t, k_{t+1}]  ← Append, don't recompute K_t
       V_{t+1} = [V_t, v_{t+1}]  ← Append, don't recompute V_t
       Q_{t+1} = [q_{t+1}]       ← Only compute new query
```

This simple observation leads to massive computational savings!

---

## Further Reading

- **Multi-Query Attention (MQA):** Reduces cache size by 95% by sharing K/V across heads
- **Grouped-Query Attention (GQA):** Balance between MHA and MQA (used in Llama 2)
- **PagedAttention (vLLM):** Manages KV cache like OS memory paging
- **Sliding Window Attention:** Limits cache to recent tokens for infinite sequences
- **Flash Attention:** Optimizes attention computation with cache

---
