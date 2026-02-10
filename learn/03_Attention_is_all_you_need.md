# Attention Is All You Need: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem with RNNs](#the-problem-with-rnns)
3. [Core Concept: Attention Mechanism](#core-concept-attention-mechanism)
4. [Self-Attention in Detail](#self-attention-in-detail)
5. [Multi-Head Attention](#multi-head-attention)
6. [Positional Encoding](#positional-encoding)
7. [Complete Transformer Architecture](#complete-transformer-architecture)
8. [Training Process](#training-process)
9. [Practical Examples](#practical-examples)
10. [Mathematical Deep Dive](#mathematical-deep-dive)

---

## Introduction

### What Problem Does This Paper Solve?

Before Transformers (2017), sequence modeling relied on:
- **RNNs (Recurrent Neural Networks)**: Process sequences step-by-step
- **LSTMs/GRUs**: Better RNNs that handle long sequences
- **Attention mechanisms**: Used WITH RNNs to focus on important parts

**The Revolutionary Idea:**
> "What if we could use ONLY attention mechanisms, without RNNs at all?"

This is what the paper "Attention Is All You Need" proposed, introducing the **Transformer**.

### Key Innovations

1. **Self-Attention**: Let each word look at all other words simultaneously
2. **Parallelization**: Process entire sequence at once (not step-by-step)
3. **Positional Encoding**: Inject position information without recurrence
4. **Multi-Head Attention**: Attend to different aspects simultaneously

---

## The Problem with RNNs

### How RNNs Work

```
Sequential Processing:
Time step 1: h₁ = f(x₁, h₀)
Time step 2: h₂ = f(x₂, h₁)
Time step 3: h₃ = f(x₃, h₂)
...

Each step depends on previous step → Cannot parallelize
```

### Problems

**1. Sequential Computation**
```
Must wait for step t-1 to complete before computing step t
For sequence length n: O(n) sequential operations
```

**2. Long-Range Dependencies**
```
Information from position 1 must pass through all intermediate steps
to reach position 100

Signal degrades over long distances (vanishing gradient)
```

**3. Limited Context**
```
At position t, model has limited "memory" of early positions
Even with LSTM/GRU, context window is effectively limited
```

### The Transformer Solution

```
✗ RNN:     Sequential processing, limited parallelization
✓ Transformer: Parallel processing, direct connections
```

**Key Insight:**
Every position can directly attend to every other position in ONE step!

```
Position 1 ←→ Position 2
    ↓    ×   ×    ↓
Position 3 ←→ Position 4

All connections computed simultaneously
```

---

## Core Concept: Attention Mechanism

### What is Attention?

**Intuition:** When reading a sentence, we don't give equal weight to all words. We focus (attend) to relevant parts.

**Example:**
```
Query: "What did the cat do?"
Context: "The cat sat on the mat and slept peacefully"

Attention weights:
- "cat": 0.4 (high - directly mentioned)
- "sat": 0.3 (high - the action)
- "slept": 0.2 (medium - another action)
- "the", "on", "and": 0.1 (low - less relevant)
```

### Attention as Information Retrieval

Think of attention like a database query:

```
Database:
Key 1: "animal" → Value 1: "cat"
Key 2: "action" → Value 2: "sat"
Key 3: "location" → Value 3: "mat"

Query: "animal"
→ Compare query with all keys
→ Key 1 matches best
→ Return Value 1: "cat"
```

### Mathematical Formulation

```
Attention(Query, Keys, Values) = Weighted sum of Values

where weights are determined by similarity between Query and Keys
```

**Formula:**
```
Attention(Q, K, V) = softmax(score(Q, K)) × V

where:
- Q = Query (what we're looking for)
- K = Keys (what we compare against)
- V = Values (what we retrieve)
- score(Q, K) = how well Q matches each K
```

---

## Self-Attention in Detail

### Step-by-Step Explanation

**Given:** Input sequence of word embeddings

```
Input: "The cat sat"

Word embeddings (simplified to 4 dimensions):
x₁ = [0.2, 0.5, 0.1, 0.3]  # "The"
x₂ = [0.4, 0.1, 0.8, 0.2]  # "cat"
x₃ = [0.3, 0.7, 0.2, 0.6]  # "sat"
```

### Step 1: Create Q, K, V

Transform each word into Query, Key, and Value vectors:

```
Q = X × W^Q
K = X × W^K
V = X × W^V

where:
- X ∈ ℝ^(n×d): input matrix (n words, d dimensions)
- W^Q, W^K, W^V ∈ ℝ^(d×d_k): learned weight matrices
```

**Why three transformations?**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I offer?"
- **Value (V)**: "What do I actually contain?"

**Example with numbers:**
```
Let d=4, d_k=3

W^Q = [[0.2, 0.3, 0.1],
       [0.4, 0.2, 0.5],
       [0.1, 0.6, 0.3],
       [0.3, 0.1, 0.4]]

For x₁ = [0.2, 0.5, 0.1, 0.3]:
q₁ = x₁ × W^Q = [0.31, 0.33, 0.29]

Similarly compute k₁, v₁, q₂, k₂, v₂, etc.
```

### Step 2: Compute Attention Scores

Calculate how much each word should attend to every other word:

```
Score(qᵢ, kⱼ) = qᵢ · kⱼ  (dot product)

Create score matrix:
       k₁    k₂    k₃
q₁  [  ?     ?     ?  ]
q₂  [  ?     ?     ?  ]
q₃  [  ?     ?     ?  ]
```

**Example:**
```
q₁ = [0.31, 0.33, 0.29]
k₁ = [0.25, 0.35, 0.22]
k₂ = [0.42, 0.18, 0.31]
k₃ = [0.33, 0.28, 0.19]

Score(q₁, k₁) = 0.31×0.25 + 0.33×0.35 + 0.29×0.22 = 0.257
Score(q₁, k₂) = 0.31×0.42 + 0.33×0.18 + 0.29×0.31 = 0.280
Score(q₁, k₃) = 0.31×0.33 + 0.33×0.28 + 0.29×0.19 = 0.250

Score matrix for q₁:
[0.257, 0.280, 0.250]
```

### Step 3: Scale Scores

Divide by √d_k to prevent softmax saturation:

```
Scaled_Score = Score / √d_k

Why scaling?
- For high dimensions, dot products become very large
- Large values → softmax saturates → small gradients
- Scaling maintains variance
```

**Example:**
```
d_k = 3
√d_k = √3 ≈ 1.732

Scaled scores for q₁:
[0.257/1.732, 0.280/1.732, 0.250/1.732]
= [0.148, 0.162, 0.144]
```

**Mathematical justification:**
```
For random vectors q, k ∈ ℝ^d_k:
E[q·k] = 0
Var[q·k] = d_k

After scaling by 1/√d_k:
Var[q·k/√d_k] = 1

This keeps gradients healthy
```

### Step 4: Apply Softmax

Convert scores to probabilities (attention weights):

```
Attention_Weights = softmax(Scaled_Scores)

softmax([s₁, s₂, s₃]) = [exp(s₁), exp(s₂), exp(s₃)] / Σexp(sᵢ)
```

**Example:**
```
Scaled scores: [0.148, 0.162, 0.144]

exp(0.148) = 1.160
exp(0.162) = 1.176
exp(0.144) = 1.155
Sum = 3.491

Attention weights:
[1.160/3.491, 1.176/3.491, 1.155/3.491]
= [0.332, 0.337, 0.331]

Properties:
- All weights sum to 1.0
- All weights between 0 and 1
- Higher scores → higher weights
```

### Step 5: Compute Weighted Sum

Multiply attention weights by values and sum:

```
Output = Σ (Attention_Weight × Value)

For position i:
outputᵢ = α_{i,1}·v₁ + α_{i,2}·v₂ + α_{i,3}·v₃
```

**Example:**
```
Attention weights for q₁: [0.332, 0.337, 0.331]

Values:
v₁ = [0.3, 0.4, 0.2]
v₂ = [0.5, 0.2, 0.6]
v₃ = [0.4, 0.5, 0.3]

output₁ = 0.332×[0.3, 0.4, 0.2] 
        + 0.337×[0.5, 0.2, 0.6]
        + 0.331×[0.4, 0.5, 0.3]

        = [0.100, 0.133, 0.066]
        + [0.169, 0.067, 0.202]
        + [0.132, 0.166, 0.099]

        = [0.401, 0.366, 0.367]

This is the new representation for "The" that incorporates
context from "cat" and "sat"
```

### Complete Formula

```
Self-Attention(X) = softmax(QK^T / √d_k) V

where:
- X ∈ ℝ^(n×d): input embeddings
- Q = XW^Q, K = XW^K, V = XW^V
- QK^T ∈ ℝ^(n×n): attention score matrix
- softmax(QK^T / √d_k) ∈ ℝ^(n×n): attention weights
- Output ∈ ℝ^(n×d_k): context-aware representations
```

### Visual Summary

```
Input:
"The cat sat"

         Q          K          V
        ↓          ↓          ↓
    [q₁ q₂ q₃] [k₁ k₂ k₃] [v₁ v₂ v₃]
        ↓          ↓
    QK^T → Scores
        ↓
    Scale by √d_k
        ↓
    Softmax → Weights
        ↓
    Weights × V → Output

Output: Context-aware embeddings
```

---

## Multi-Head Attention

### Why Multiple Heads?

**Problem with Single Attention:**
```
One attention can focus on one aspect:
"The cat sat on the mat"
→ Might focus on subject-verb relationships

But we also want:
- Syntactic relationships
- Semantic relationships
- Positional relationships
- etc.
```

**Solution:** Use multiple attention heads in parallel!

### How It Works

```
Instead of one attention:
Attention(Q, K, V) → Output ∈ ℝ^(n×d)

Use h heads:
head₁ = Attention(Q₁, K₁, V₁)
head₂ = Attention(Q₂, K₂, V₂)
...
headₕ = Attention(Qₕ, Kₕ, Vₕ)

Concatenate and project:
Output = Concat(head₁, ..., headₕ) × W^O
```

### Mathematical Formulation

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

where:
headᵢ = Attention(QW_i^Q, KW_i^K, VW_i^V)

Parameters:
- W_i^Q, W_i^K, W_i^V ∈ ℝ^(d_model × d_k)
- W^O ∈ ℝ^(hd_k × d_model)
- d_k = d_model / h (typical choice)
```

### Detailed Example

**Setup:**
```
d_model = 512  (embedding dimension)
h = 8         (number of heads)
d_k = 512/8 = 64 (dimension per head)
```

**Step 1: Split into heads**
```
Original:
Q ∈ ℝ^(n×512)

Split for each head:
Q₁ = Q × W₁^Q ∈ ℝ^(n×64)
Q₂ = Q × W₂^Q ∈ ℝ^(n×64)
...
Q₈ = Q × W₈^Q ∈ ℝ^(n×64)

Each head gets its own projection matrices
```

**Step 2: Compute attention for each head**
```
head₁ = Attention(Q₁, K₁, V₁) ∈ ℝ^(n×64)
head₂ = Attention(Q₂, K₂, V₂) ∈ ℝ^(n×64)
...
head₈ = Attention(Q₈, K₈, V₈) ∈ ℝ^(n×64)

Each head learns to focus on different aspects
```

**Step 3: Concatenate**
```
Concat(head₁, ..., head₈) ∈ ℝ^(n×512)

[head₁ | head₂ | head₃ | head₄ | head₅ | head₆ | head₇ | head₈]
 n×64   n×64   n×64   n×64   n×64   n×64   n×64   n×64
→ n×512
```

**Step 4: Final projection**
```
Output = Concat × W^O ∈ ℝ^(n×512)

W^O ∈ ℝ^(512×512)

This mixes information from all heads
```

### What Do Different Heads Learn?

**Example from trained model:**

```
Head 1: Subject-Verb agreement
"The cat [sits] on the mat"
 High attention between "cat" and "sits"

Head 2: Determiner-Noun
"[The] cat sits on [the] mat"
 High attention between determiners and nouns

Head 3: Preposition-Object
"sits [on the mat]"
 High attention in prepositional phrases

Head 4: Long-range dependencies
"The [cat] ... eventually [slept]"
 Connects distant related words

... and so on
```

### Computational Complexity

```
Single-head attention:
- QK^T: O(n² × d)
- Softmax: O(n²)
- Weighted sum: O(n² × d)
Total: O(n² × d)

Multi-head (h heads):
- Per head: O(n² × d/h)
- h heads: O(n² × d)
- Projection: O(n × d²)
Total: O(n² × d + n × d²)

For small n: dominated by O(n × d²)
For large n: dominated by O(n² × d)
```

---

## Positional Encoding

### The Position Problem

**Issue:** Self-attention has no notion of order!

```
"The cat sat" has same attention as "sat cat The"

Without position info:
Self-Attention([The, cat, sat]) = Self-Attention([sat, cat, The])

This is BAD for language!
```

### Solution: Positional Encoding

Add position information to embeddings:

```
Final_Embedding = Word_Embedding + Positional_Encoding

PE ∈ ℝ^(max_len × d_model)
```

### Sinusoidal Positional Encoding

The paper uses sine and cosine functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index (0, 1, 2, ..., d_model/2-1)
- Even dimensions use sin
- Odd dimensions use cos
```

### Why This Formula?

**1. Unique encoding for each position**
```
Different positions → different values
PE(0) ≠ PE(1) ≠ PE(2) ...
```

**2. Relative position information**
```
PE(pos+k) can be expressed as linear function of PE(pos)

This helps model learn relative distances:
"2 words apart" has consistent pattern regardless of absolute position
```

**3. Extrapolation to longer sequences**
```
The formula works for any position
Can handle sequences longer than seen during training
```

### Detailed Example

```
d_model = 4 (simplified)
Let's compute PE for positions 0, 1, 2

For position 0:
i=0: PE(0,0) = sin(0 / 10000^(0/4)) = sin(0) = 0
i=0: PE(0,1) = cos(0 / 10000^(0/4)) = cos(0) = 1
i=1: PE(0,2) = sin(0 / 10000^(2/4)) = sin(0) = 0
i=1: PE(0,3) = cos(0 / 10000^(2/4)) = cos(0) = 1

PE(0) = [0, 1, 0, 1]

For position 1:
i=0: PE(1,0) = sin(1 / 10000^(0/4)) = sin(1) ≈ 0.841
i=0: PE(1,1) = cos(1 / 10000^(0/4)) = cos(1) ≈ 0.540
i=1: PE(1,2) = sin(1 / 10000^(0.5)) = sin(0.01) ≈ 0.010
i=1: PE(1,3) = cos(1 / 10000^(0.5)) = cos(0.01) ≈ 1.000

PE(1) = [0.841, 0.540, 0.010, 1.000]

For position 2:
PE(2,0) = sin(2 / 1) = sin(2) ≈ 0.909
PE(2,1) = cos(2 / 1) = cos(2) ≈ -0.416
PE(2,2) = sin(2 / 100) = sin(0.02) ≈ 0.020
PE(2,3) = cos(2 / 100) = cos(0.02) ≈ 0.9998

PE(2) = [0.909, -0.416, 0.020, 0.9998]
```

### Frequency Pattern

```
Different dimensions have different frequencies:

Dimension 0, 1: High frequency (changes rapidly with position)
  → Captures fine-grained position differences

Dimension 2, 3: Low frequency (changes slowly)
  → Captures coarse-grained position differences

This gives model multiple "wavelengths" to understand position
```

### Visualization

```
Position encoding as waves:

Dim 0 (fast): ∿∿∿∿∿∿∿∿∿∿∿∿∿
Dim 1 (fast): ∿∿∿∿∿∿∿∿∿∿∿∿∿
Dim 2 (slow): ∿∿∿∿∿
Dim 3 (slow): ∿∿∿∿∿
...

Each position gets a unique "fingerprint"
```

### Adding to Embeddings

```
Word: "cat" at position 5

Word embedding: [0.2, 0.5, 0.1, 0.3, ...]
Positional encoding: [0.1, 0.4, 0.8, 0.2, ...]

Final embedding: [0.3, 0.9, 0.9, 0.5, ...]
                  ↑   ↑   ↑   ↑
                  Element-wise addition
```

---

## Complete Transformer Architecture

### Overall Structure

```
Input Sequence
     ↓
Input Embedding + Positional Encoding
     ↓
[Encoder Stack] (N layers)
     ↓
Encoder Output
     ↓
[Decoder Stack] (N layers)
     ↓
Linear + Softmax
     ↓
Output Probabilities
```

### Encoder Architecture

**Single Encoder Layer:**
```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Normalize (Residual Connection + Layer Norm)
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

**Mathematical Flow:**

```
# Sub-layer 1: Multi-Head Attention
x₁ = x + MultiHeadAttention(x, x, x)
x₁ = LayerNorm(x₁)

# Sub-layer 2: Feed-Forward
x₂ = x₁ + FFN(x₁)
x₂ = LayerNorm(x₂)

Output: x₂
```

**Detailed Equations:**

```
1. Multi-Head Attention:
   attention_output = MultiHeadAttention(x, x, x)
   x = LayerNorm(x + attention_output)

2. Feed-Forward Network:
   ffn_output = FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   x = LayerNorm(x + ffn_output)

where:
- W₁ ∈ ℝ^(d_model × d_ff), typically d_ff = 4 × d_model
- W₂ ∈ ℝ^(d_ff × d_model)
```

### Decoder Architecture

**Single Decoder Layer:**
```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Normalize
  ↓
Multi-Head Cross-Attention (attends to encoder)
  ↓
Add & Normalize
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

**Mathematical Flow:**

```
# Sub-layer 1: Masked Self-Attention
x₁ = x + MaskedMultiHeadAttention(x, x, x)
x₁ = LayerNorm(x₁)

# Sub-layer 2: Cross-Attention
x₂ = x₁ + MultiHeadAttention(Q=x₁, K=encoder_output, V=encoder_output)
x₂ = LayerNorm(x₂)

# Sub-layer 3: Feed-Forward
x₃ = x₂ + FFN(x₂)
x₃ = LayerNorm(x₃)

Output: x₃
```

### Masked Self-Attention

**Purpose:** Prevent decoder from seeing future tokens

```
When generating position i, can only attend to positions ≤ i

Attention mask:
       k₁   k₂   k₃   k₄
q₁  [  0   -∞   -∞   -∞ ]
q₂  [  0    0   -∞   -∞ ]
q₃  [  0    0    0   -∞ ]
q₄  [  0    0    0    0 ]

-∞ values become 0 after softmax
```

**Implementation:**
```
scores = QK^T / √d_k

# Add mask (upper triangular -∞)
masked_scores = scores + mask

# Softmax zeros out -∞ positions
attention_weights = softmax(masked_scores)
```

**Example:**
```
Generating "The cat sat"

Position 1 (generating "cat"):
  Can see: ["The"]
  Cannot see: ["cat", "sat"]

Position 2 (generating "sat"):
  Can see: ["The", "cat"]
  Cannot see: ["sat"]

This ensures autoregressive property
```

### Cross-Attention

**Purpose:** Decoder attends to encoder output

```
Q: from decoder (what we're generating)
K, V: from encoder (source sequence representation)

This allows decoder to focus on relevant parts of input
```

**Example (Translation):**
```
Input (encoder): "Le chat dort"
Output (decoder): "The cat sleeps"

When generating "cat":
Q = decoder representation of current position
K, V = encoder representations of ["Le", "chat", "dort"]

Attention learns: "cat" should attend strongly to "chat"
```

### Residual Connections

**Formula:**
```
Output = LayerNorm(x + Sublayer(x))

where Sublayer ∈ {Attention, FFN}
```

**Why needed?**

```
Without residual:
Layer 1 → Layer 2 → Layer 3 → ... → Layer N
Gradients must flow through all transformations
→ Vanishing gradients for deep networks

With residual:
Each layer has direct path to input
→ Gradients can flow directly
→ Enables training of deep networks (N=6, 12, 24, ...)
```

### Layer Normalization

**Formula:**
```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β

where:
- μ = mean(x) = (1/d)Σxᵢ
- σ = std(x) = √((1/d)Σ(xᵢ - μ)²)
- γ, β = learned parameters
- ⊙ = element-wise multiplication
```

**Example:**
```
x = [2, 4, 6, 8]

μ = (2+4+6+8)/4 = 5
σ = √((9+1+1+9)/4) = √5 ≈ 2.236

Normalized: 
[(2-5)/2.236, (4-5)/2.236, (6-5)/2.236, (8-5)/2.236]
= [-1.34, -0.45, 0.45, 1.34]

If γ = [1, 1, 1, 1], β = [0, 0, 0, 0]:
Output = [-1.34, -0.45, 0.45, 1.34]
```

**Benefits:**
- Stabilizes training
- Reduces internal covariate shift
- Allows higher learning rates

### Feed-Forward Network

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

Applied identically to each position
```

**Example:**
```
d_model = 512
d_ff = 2048

Position 1: FFN([...512 dims...]) → [...512 dims...]
Position 2: FFN([...512 dims...]) → [...512 dims...]
...

Same FFN parameters for all positions
But different inputs → different outputs
```

### Complete Parameter Count

**For N=6 layers, d_model=512, h=8:**

```
Encoder layer:
- Multi-head attention: 4 × 512² = 1,048,576
- FFN: 2 × 512 × 2048 = 2,097,152
- Layer norms: 4 × 512 = 2,048
Total per layer: ~3.15M

Encoder stack (6 layers): ~18.9M

Decoder layer:
- Masked attention: ~1.05M
- Cross attention: ~1.05M
- FFN: ~2.1M
- Layer norms: ~2K
Total per layer: ~4.2M

Decoder stack (6 layers): ~25.2M

Embeddings: 2 × vocab_size × 512

Total (vocab=37K): ~100M parameters
```

---

## Training Process

### Training Objective

**Task:** Sequence-to-sequence learning (e.g., translation)

```
Input: Source sentence
Output: Target sentence

Maximize: P(Target | Source)
```

### Loss Function

**Cross-Entropy Loss:**

```
L = -Σₜ log P(yₜ | y₁, ..., yₜ₋₁, Source)

where:
- yₜ = target token at position t
- y₁, ..., yₜ₋₁ = previously generated tokens
- Source = encoder representation
```

**Detailed Computation:**

```
1. Encoder processes source:
   encoder_output = Encoder(source)

2. Decoder generates probabilities:
   logits = Decoder(target_prefix, encoder_output)
   probabilities = softmax(logits)

3. Compute loss:
   For each position t:
     loss_t = -log(P(yₜ | context))
   
   Total loss = average(loss_t for all t)
```

### Training Example

```
Source: "Je suis étudiant"
Target: "I am a student"

Step 1: Encode source
encoder_output = Encoder("Je suis étudiant")

Step 2: Decode with teacher forcing
Input to decoder: [START, "I", "am", "a"]
Expected output:  ["I", "am", "a", "student"]

Decoder processes:
Position 1: P("I" | START, encoder_output)
Position 2: P("am" | START, "I", encoder_output)
Position 3: P("a" | START, "I", "am", encoder_output)
Position 4: P("student" | START, "I", "am", "a", encoder_output)

Loss = -[log P₁ + log P₂ + log P₃ + log P₄] / 4
```

### Optimization

**Optimizer:** Adam with custom learning rate schedule

```
Learning rate schedule:
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

where:
- d_model = 512
- warmup = 4000 steps
```

**Schedule visualization:**
```
Steps:     0    2K   4K   6K   8K   10K
LR:        0    ↗    ↘    ↘    ↘    ↘
           0   0.7   1.0  0.8  0.7  0.6

Phase 1 (0-4K): Linear warmup
Phase 2 (4K+): Inverse square root decay
```

**Why this schedule?**
- Warmup: Prevents instability in early training
- Decay: Allows fine-tuning in later training

### Label Smoothing

**Problem:** One-hot labels can make model overconfident

```
Standard: [0, 0, 1, 0, 0]  (100% probability on correct token)
Smoothed:  [0.02, 0.02, 0.92, 0.02, 0.02]  (distribute some probability)
```

**Formula:**
```
y_smoothed = (1 - ε) × y_true + ε / K

where:
- ε = 0.1 (smoothing parameter)
- K = vocabulary size
```

**Benefits:**
- Prevents overfitting
- Improves generalization
- Better calibrated probabilities

### Dropout

Applied at multiple places:

```
1. After embeddings: dropout(embeddings, p=0.1)
2. After attention: dropout(attention_output, p=0.1)
3. After FFN: dropout(ffn_output, p=0.1)
4. In FFN: dropout(ReLU(x), p=0.1)
```

### Batch Processing

**Batching strategy:**

```
Group sentences by similar length:
Batch 1: [sent1(len=10), sent2(len=11), sent3(len=9)]
Batch 2: [sent4(len=25), sent5(len=27), sent6(len=24)]

Pad to max length in batch
Use attention masks to ignore padding
```

**Efficient batching:**
```
Target: ~25,000 tokens per batch

If sentences average 50 tokens:
batch_size ≈ 25,000 / 50 = 500 sentences

Dynamically adjust based on length
```

---

## Practical Examples

### Example 1: Simple Attention Calculation

**Given:** 3 words, 4 dimensions

```
Input:
x₁ = [1, 0, 1, 0]  # "The"
x₂ = [0, 2, 0, 2]  # "cat"
x₃ = [1, 1, 1, 1]  # "sat"

Weight matrices (simplified):
W^Q = W^K = W^V = I₄ₓ₄ (identity for simplicity)

So Q = K = V = X
```

**Step 1: Compute QK^T**

```
Q = [[1, 0, 1, 0],
     [0, 2, 0, 2],
     [1, 1, 1, 1]]

K^T = [[1, 0, 1],
       [0, 2, 1],
       [1, 0, 1],
       [0, 2, 1]]

QK^T = [[2, 0, 2],
        [0, 8, 4],
        [2, 4, 4]]
```

**Step 2: Scale**

```
d_k = 4, √d_k = 2

Scaled = QK^T / 2 = [[1.0, 0.0, 1.0],
                      [0.0, 4.0, 2.0],
                      [1.0, 2.0, 2.0]]
```

**Step 3: Softmax**

```
Row 1: [1.0, 0.0, 1.0]
exp: [2.718, 1.0, 2.718]
softmax: [0.42, 0.16, 0.42]

Row 2: [0.0, 4.0, 2.0]
exp: [1.0, 54.6, 7.39]
softmax: [0.02, 0.87, 0.11]

Row 3: [1.0, 2.0, 2.0]
exp: [2.718, 7.39, 7.39]
softmax: [0.16, 0.42, 0.42]

Attention weights:
[[0.42, 0.16, 0.42],
 [0.02, 0.87, 0.11],
 [0.16, 0.42, 0.42]]
```

**Step 4: Weighted sum**

```
V = [[1, 0, 1, 0],
     [0, 2, 0, 2],
     [1, 1, 1, 1]]

Output for position 1:
= 0.42×[1,0,1,0] + 0.16×[0,2,0,2] + 0.42×[1,1,1,1]
= [0.42,0,0.42,0] + [0,0.32,0,0.32] + [0.42,0.42,0.42,0.42]
= [0.84, 0.74, 0.84, 0.74]

Output for position 2:
= 0.02×[1,0,1,0] + 0.87×[0,2,0,2] + 0.11×[1,1,1,1]
= [0.13, 1.85, 0.13, 1.85]

Output for position 3:
= 0.16×[1,0,1,0] + 0.42×[0,2,0,2] + 0.42×[1,1,1,1]
= [0.58, 1.26, 0.58, 1.26]
```

**Interpretation:**

```
Position 2 ("cat"):
- Attends mostly to itself (0.87)
- Output [0.13, 1.85, 0.13, 1.85] is close to original [0,2,0,2]

Positions 1 & 3 ("The", "sat"):
- Attention more distributed
- Outputs blend information from multiple positions
```

### Example 2: Positional Encoding

```
Compute positional encoding for position 3, d_model = 6

i=0: PE(3, 0) = sin(3 / 10000^(0/6)) = sin(3) ≈ 0.141
i=0: PE(3, 1) = cos(3 / 10000^(0/6)) = cos(3) ≈ -0.990

i=1: PE(3, 2) = sin(3 / 10000^(2/6)) = sin(3/21.54) ≈ 0.139
i=1: PE(3, 3) = cos(3 / 10000^(2/6)) = cos(3/21.54) ≈ 0.990

i=2: PE(3, 4) = sin(3 / 10000^(4/6)) = sin(3/464.16) ≈ 0.0065
i=2: PE(3, 5) = cos(3 / 10000^(4/6)) = cos(3/464.16) ≈ 1.000

PE(3) = [0.141, -0.990, 0.139, 0.990, 0.0065, 1.000]

Notice how higher dimensions change more slowly (capturing coarse position)
```

### Example 3: Multi-Head Attention Split

```
d_model = 8, h = 2 heads

Input: x ∈ ℝ^(n×8)

Head 1:
Q₁ = x × W₁^Q ∈ ℝ^(n×4)  (first 4 dimensions)
K₁ = x × W₁^K ∈ ℝ^(n×4)
V₁ = x × W₁^V ∈ ℝ^(n×4)
head₁ = Attention(Q₁, K₁, V₁) ∈ ℝ^(n×4)

Head 2:
Q₂ = x × W₂^Q ∈ ℝ^(n×4)  (last 4 dimensions)
K₂ = x × W₂^K ∈ ℝ^(n×4)
V₂ = x × W₂^V ∈ ℝ^(n×4)
head₂ = Attention(Q₂, K₂, V₂) ∈ ℝ^(n×4)

Concatenate:
[head₁ | head₂] ∈ ℝ^(n×8)

Project:
output = [head₁ | head₂] × W^O ∈ ℝ^(n×8)
```

---

## Mathematical Deep Dive

### Why Attention Works: Information Theory View

**Mutual Information:**

```
Attention maximizes relevant information flow:

I(output; relevant_input) should be high
I(output; irrelevant_input) should be low

Softmax attention achieves this by:
- High scores → high attention → high information flow
- Low scores → low attention → filtered out
```

### Gradient Flow Analysis

**Why scaling by √d_k matters:**

```
Without scaling:
scores = QK^T ∈ [-∞, +∞]

For large d_k, variance grows:
Var[q·k] = d_k × Var[q] × Var[k]

Large variance → saturation in softmax:
softmax([100, 1, 2]) ≈ [1.0, 0, 0]  (saturated)

Saturated softmax → vanishing gradients:
∂softmax/∂input ≈ 0

Scaling keeps variance constant:
Var[q·k / √d_k] = Var[q] × Var[k]

Healthy gradients throughout training
```

### Computational Complexity Analysis

**Self-Attention:**
```
QK^T: O(n² × d)
Softmax: O(n²)
Weighted sum: O(n² × d)

Total: O(n² × d)

Memory: O(n²) for attention matrix
```

**Comparison with RNN:**
```
RNN: O(n × d²) time, O(1) memory per step
Transformer: O(n² × d) time, O(n²) memory

Trade-off:
- RNN: Sequential, lower memory, quadratic in d
- Transformer: Parallel, higher memory, linear in d

For typical values (n < 1000, d > 1000):
Transformer is faster when parallelized
```

### Why Multiple Layers?

**Depth enables hierarchy:**

```
Layer 1: Local patterns
- Bigrams, trigrams
- "New York", "United States"

Layer 2: Phrasal patterns
- Noun phrases, verb phrases
- "the big red car"

Layer 3: Sentence-level patterns
- Subject-verb-object
- Dependencies

Layer 4+: Discourse and reasoning
- Coreference
- Logical relationships
```

**Empirical findings:**
```
Shallow (1-2 layers): Poor performance
Medium (4-6 layers): Good performance
Deep (12+ layers): Best performance (with residuals)
```

### Attention Pattern Analysis

**Typical learned patterns:**

```
Layer 1:
- Attend to adjacent words (local structure)
- "The [cat]" → attends to "The"

Layer 2-3:
- Attend to syntactic heads
- "sat" → attends to "cat" (subject)

Layer 4-6:
- Long-range dependencies
- Pronouns → referents
- "it" → "the cat"
```

### Universal Approximation

**Theorem:** Transformers are Turing complete

```
With sufficient depth and width:
- Can compute any function
- Can simulate any algorithm

Attention provides:
- Dynamic routing of information
- Content-based addressing
- Soft differentiable operations
```

---

## Summary

### Key Innovations

1. **Self-Attention:** O(1) path length between any two positions
2. **Multi-Head:** Attend to multiple representation subspaces
3. **Positional Encoding:** Inject sequence order information
4. **Parallel Processing:** Entire sequence processed simultaneously

### Core Equations

```
1. Attention:
   Attention(Q, K, V) = softmax(QK^T / √d_k)V

2. Multi-Head:
   MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
   where headᵢ = Attention(QW_i^Q, KW_i^K, VW_i^V)

3. Positional Encoding:
   PE(pos,2i) = sin(pos/10000^(2i/d))
   PE(pos,2i+1) = cos(pos/10000^(2i/d))

4. Transformer Block:
   x = LayerNorm(x + MultiHeadAttention(x))
   x = LayerNorm(x + FFN(x))
```

### Complexity

```
Time: O(n² × d) per layer
Space: O(n² + n×d)

Bottleneck: n² attention matrix
```

### Impact

```
Revolutionized NLP:
- BERT (2018)
- GPT-2, GPT-3 (2019-2020)
- T5, BART (2020)
- ChatGPT (2022)
- GPT-4 (2023)

Extended to other domains:
- Vision Transformers (ViT)
- Audio processing
- Protein structure prediction (AlphaFold)
```

---

## References

1. **Original Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
2. **BERT:** "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
3. **GPT:** "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
4. **The Illustrated Transformer:** Jay Alammar's blog
5. **Formal Algorithms for Transformers:** Phuong & Hutter, 2022

---

*Last Updated: February 2026*
*Document Version: 1.0*
