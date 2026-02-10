 
# Self-Attention: Complete Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [The Intuition](#the-intuition)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Detailed Examples](#detailed-examples)
6. [Variants of Self-Attention](#variants-of-self-attention)
7. [Multi-Head Self-Attention](#multi-head-self-attention)
8. [Computational Complexity](#computational-complexity)
9. [Practical Applications](#practical-applications)
10. [Common Pitfalls](#common-pitfalls)

---

## Introduction

### What is Self-Attention?

**Self-Attention** is a mechanism that allows each element in a sequence to attend to (focus on) all other elements in the same sequence, including itself.

**Key Idea:**
> When processing a word in a sentence, self-attention lets the model decide which other words are important for understanding that word.

### Why Do We Need It?

**Problem with traditional models:**

```
RNN/LSTM: Process sequentially
"The cat sat on the mat" 
→ Process "The" → then "cat" → then "sat" → ...

Issues:
1. Sequential processing (slow)
2. Long-range dependencies are hard
3. Information bottleneck
```

**Self-Attention solution:**

```
Process ALL words simultaneously
Each word can directly look at ANY other word
No sequential bottleneck
Parallelizable
```

### Historical Context

```
Timeline:
2014: Attention mechanism (Bahdanau et al.)
      - Used WITH RNNs for translation
      
2017: Self-Attention (Vaswani et al.)
      - "Attention Is All You Need"
      - Replace RNNs entirely with self-attention
      
2018+: Transformers everywhere
      - BERT, GPT, T5, etc.
```

---

## The Intuition

### Analogy 1: Library Search

Imagine you're in a library looking for information about "cats":

```
Query (What you're looking for): "Information about cats"

Library Books (Keys and Values):
Book 1 - Key: "Animals", Value: "Encyclopedia of Animals"
Book 2 - Key: "Pets", Value: "Guide to Pet Care"
Book 3 - Key: "Physics", Value: "Quantum Mechanics"
Book 4 - Key: "Cats", Value: "Everything About Cats"

Self-Attention Process:
1. Compare your query with each book's key
   - "cats" vs "Animals" → moderate match (0.3)
   - "cats" vs "Pets" → good match (0.4)
   - "cats" vs "Physics" → no match (0.0)
   - "cats" vs "Cats" → perfect match (0.9)

2. Use matches to decide which books to read
   - Mostly read "Everything About Cats" (90%)
   - Also check "Guide to Pet Care" (40%)
   - Glance at "Encyclopedia of Animals" (30%)
   - Ignore "Quantum Mechanics" (0%)

3. Combine information from selected books
   - Final understanding = weighted mix of relevant books
```

### Analogy 2: Conversation Context

```
Sentence: "The animal didn't cross the street because it was too tired"

When understanding "it":
- Could refer to "animal" or "street"
- Self-attention learns to focus on "animal"

Self-Attention weights:
"it" attending to:
- "animal": 0.7 (high - correct referent)
- "street": 0.2 (low - unlikely referent)
- "tired": 0.5 (medium - related concept)
- other words: 0.1 (low)

Result: Model understands "it" = "animal"
```

### The Three Components: Q, K, V

**Think of it as a recommendation system:**

```
Query (Q): "What am I looking for?"
- Your current interest/focus
- Example: "I want to understand 'cat'"

Key (K): "What do I offer?"
- Index/summary of content
- Example: Each word's "topic" or "role"

Value (V): "What is my actual content?"
- The actual information to retrieve
- Example: The word's meaning/representation
```

**Concrete Example:**

```
Sentence: "The cat sat on the mat"

Word: "cat"
- Query: "I'm a noun, who modifies me?"
- Key: "I offer noun-related information"
- Value: [0.2, 0.8, 0.1, ...] (actual embedding)

Word: "The"
- Query: "I'm a determiner, what noun do I modify?"
- Key: "I offer determiner information"
- Value: [0.1, 0.3, 0.9, ...]
```

### Visual Representation

```
Input: ["The", "cat", "sat"]

Self-Attention Process:

      Q           K           V
      ↓           ↓           ↓
    [q₁]        [k₁]        [v₁]   ← "The"
    [q₂]        [k₂]        [v₂]   ← "cat"
    [q₃]        [k₃]        [v₃]   ← "sat"
      ↓           ↓           ↓
    Compare Q with all K
         ↓
    Attention Scores
         ↓
    Normalize (Softmax)
         ↓
    Attention Weights
         ↓
    Weighted Sum of V
         ↓
    Output: Context-aware representations
```

---

## Mathematical Foundation

### Core Formula

**Self-Attention in one equation:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
- Q ∈ ℝ^(n×d_k): Query matrix
- K ∈ ℝ^(n×d_k): Key matrix
- V ∈ ℝ^(n×d_v): Value matrix
- n: sequence length
- d_k: dimension of queries/keys
- d_v: dimension of values
```

### Step-by-Step Breakdown

#### Step 1: Create Q, K, V

```
Input: X ∈ ℝ^(n×d_model)

Linear transformations:
Q = X W^Q    where W^Q ∈ ℝ^(d_model×d_k)
K = X W^K    where W^K ∈ ℝ^(d_model×d_k)
V = X W^V    where W^V ∈ ℝ^(d_model×d_v)

Typically: d_k = d_v = d_model
```

**Why three separate transformations?**

```
Flexibility: Each can learn different aspects
- Q learns "what to look for"
- K learns "what to offer"
- V learns "what content to provide"

If we used X directly for all three:
- Limited expressiveness
- Cannot distinguish query vs content
```

#### Step 2: Compute Attention Scores

```
Scores = Q K^T ∈ ℝ^(n×n)

Element-wise:
Score(i,j) = qᵢ · kⱼ = Σₖ qᵢₖ × kⱼₖ

Interpretation:
- High score → query i strongly matches key j
- Low score → query i doesn't match key j
```

**Matrix form:**

```
        [k₁^T]
Q K^T = [q₁ q₂ q₃] [k₂^T]
                    [k₃^T]

      = [q₁·k₁  q₁·k₂  q₁·k₃]
        [q₂·k₁  q₂·k₂  q₂·k₃]
        [q₃·k₁  q₃·k₂  q₃·k₃]

Each row i: How much position i attends to each position
Each column j: How much attention position j receives
```

#### Step 3: Scale Scores

```
Scaled_Scores = Scores / √d_k

Why scaling?
- For large d_k, dot products grow large
- Large values → softmax saturates
- Saturated softmax → vanishing gradients

Mathematical justification:
If q, k ~ N(0, 1) (normalized):
  E[q·k] = 0
  Var[q·k] = d_k

After scaling:
  Var[q·k / √d_k] = 1

Keeps variance constant regardless of dimension
```

**Example:**

```
d_k = 64

Without scaling:
Scores = [32, 48, 16, 40]  (large values)
Softmax ≈ [0.0003, 0.9994, 0.0000, 0.0003]  (saturated)

With scaling (divide by 8):
Scores = [4, 6, 2, 5]  (moderate values)
Softmax ≈ [0.09, 0.66, 0.01, 0.24]  (healthy distribution)
```

#### Step 4: Apply Softmax

```
Attention_Weights = softmax(Scaled_Scores)

For row i:
αᵢⱼ = exp(scoreᵢⱼ) / Σₖ exp(scoreᵢₖ)

Properties:
- Σⱼ αᵢⱼ = 1  (weights sum to 1)
- 0 ≤ αᵢⱼ ≤ 1  (all weights non-negative)
- Differentiable (enables backpropagation)
```

**Softmax intuition:**

```
Input: [2, 4, 1, 3]

Step 1: Exponentiate
exp([2, 4, 1, 3]) = [7.39, 54.6, 2.72, 20.1]

Step 2: Normalize
Sum = 84.81
Output = [0.087, 0.644, 0.032, 0.237]

Effect:
- Largest value (4) gets highest weight (0.644)
- Smallest value (1) gets lowest weight (0.032)
- Differences are amplified but smooth
```

#### Step 5: Weighted Sum

```
Output = Attention_Weights × V ∈ ℝ^(n×d_v)

For position i:
outputᵢ = Σⱼ αᵢⱼ vⱼ

This is a weighted average of all value vectors
```

**Detailed calculation:**

```
For position i:

Values:
v₁ = [1, 0, 2]
v₂ = [0, 3, 1]
v₃ = [2, 1, 0]

Attention weights for position i:
α = [0.2, 0.5, 0.3]

Output:
outputᵢ = 0.2×[1,0,2] + 0.5×[0,3,1] + 0.3×[2,1,0]
        = [0.2,0,0.4] + [0,1.5,0.5] + [0.6,0.3,0]
        = [0.8, 1.8, 0.9]

Interpretation:
- Mostly influenced by v₂ (50% weight)
- Moderately by v₃ (30%)
- Slightly by v₁ (20%)
```

### Complete Mathematical Flow

```
Input: X ∈ ℝ^(n×d)

1. Linear transformations:
   Q = XW^Q ∈ ℝ^(n×d_k)
   K = XW^K ∈ ℝ^(n×d_k)
   V = XW^V ∈ ℝ^(n×d_v)

2. Compute scores:
   S = QK^T ∈ ℝ^(n×n)

3. Scale:
   S' = S / √d_k

4. Softmax:
   A = softmax(S') ∈ ℝ^(n×n)
   where Aᵢⱼ = exp(S'ᵢⱼ) / Σₖ exp(S'ᵢₖ)

5. Output:
   Output = AV ∈ ℝ^(n×d_v)

Complete formula:
Output = softmax(QK^T / √d_k) V
```

---

## Step-by-Step Walkthrough

### Example 1: Tiny Self-Attention (Manual Calculation)

**Setup:**

```
Sentence: "cat sat"
Sequence length: n = 2
Embedding dimension: d = 3
```

**Input Embeddings:**

```
X = [x₁]  = [1  0  1]  ← "cat"
    [x₂]    [0  1  1]  ← "sat"

Shape: (2, 3)
```

**Weight Matrices (simplified - using identity for clarity):**

```
W^Q = W^K = W^V = I₃ₓ₃ (identity matrix)

So: Q = K = V = X
```

#### Step 1: Q, K, V

```
Q = X × W^Q = [1  0  1]
              [0  1  1]

K = X × W^K = [1  0  1]
              [0  1  1]

V = X × W^V = [1  0  1]
              [0  1  1]

All are (2, 3) matrices
```

#### Step 2: Compute QK^T

```
Q = [1  0  1]    K^T = [1  0]
    [0  1  1]          [0  1]
                       [1  1]

QK^T = [1  0  1] × [1  0]
       [0  1  1]   [0  1]
                   [1  1]

     = [1×1+0×0+1×1  1×0+0×1+1×1]
       [0×1+1×0+1×1  0×0+1×1+1×1]

     = [2  1]
       [1  2]

Shape: (2, 2)
```

**Interpretation:**

```
QK^T = [2  1]
       [1  2]

Position 1 ("cat"):
- Attends to itself with score 2
- Attends to "sat" with score 1

Position 2 ("sat"):
- Attends to "cat" with score 1
- Attends to itself with score 2
```

#### Step 3: Scale

```
d_k = 3
√d_k = √3 ≈ 1.732

Scaled = QK^T / √d_k = [2/1.732  1/1.732]
                       [1/1.732  2/1.732]

       = [1.155  0.577]
         [0.577  1.155]
```

#### Step 4: Softmax

```
For row 1: [1.155, 0.577]

exp(1.155) = 3.173
exp(0.577) = 1.781
Sum = 4.954

Softmax:
[3.173/4.954, 1.781/4.954] = [0.640, 0.360]

For row 2: [0.577, 1.155]
Softmax: [0.360, 0.640]

Attention matrix:
A = [0.640  0.360]
    [0.360  0.640]
```

**Interpretation:**

```
Position 1 ("cat"):
- 64% attention to itself
- 36% attention to "sat"

Position 2 ("sat"):
- 36% attention to "cat"
- 64% attention to itself

Both positions attend more to themselves
but also incorporate context from the other word
```

#### Step 5: Multiply by V

```
A = [0.640  0.360]    V = [1  0  1]
    [0.360  0.640]        [0  1  1]

Output = A × V = [0.640  0.360] × [1  0  1]
                 [0.360  0.640]   [0  1  1]

For position 1:
= 0.640×[1,0,1] + 0.360×[0,1,1]
= [0.640, 0, 0.640] + [0, 0.360, 0.360]
= [0.640, 0.360, 1.000]

For position 2:
= 0.360×[1,0,1] + 0.640×[0,1,1]
= [0.360, 0, 0.360] + [0, 0.640, 0.640]
= [0.360, 0.640, 1.000]

Final output:
Output = [0.640  0.360  1.000]
         [0.360  0.640  1.000]
```

**Analysis:**

```
Original "cat": [1, 0, 1]
After attention: [0.640, 0.360, 1.000]
→ Influenced by "sat" (gained value in dimension 2)

Original "sat": [0, 1, 1]
After attention: [0.360, 0.640, 1.000]
→ Influenced by "cat" (gained value in dimension 1)

Third dimension (1.000) unchanged because both had 1
```

### Example 2: Three-Word Sentence

**Input:**

```
Sentence: "The cat sat"

Embeddings (simplified):
x₁ = [1, 0, 0, 0]  ← "The"
x₂ = [0, 1, 0, 0]  ← "cat"
x₃ = [0, 0, 1, 0]  ← "sat"

X = [1  0  0  0]
    [0  1  0  0]
    [0  0  1  0]

Shape: (3, 4)
```

**Weight matrices:**

```
W^Q = [1  0  0  1]
      [0  1  0  1]
      [0  0  1  1]
      [1  1  1  0]

W^K = [0  1  1  0]
      [1  0  1  0]
      [1  1  0  0]
      [0  0  0  1]

W^V = [1  1  0  0]
      [0  1  1  0]
      [0  0  1  1]
      [1  0  0  1]

All are (4, 4) matrices
```

#### Compute Q, K, V

```
Q = X × W^Q

For "The" (row 1 of X = [1,0,0,0]):
q₁ = [1,0,0,0] × W^Q = first row of W^Q = [1, 0, 0, 1]

For "cat":
q₂ = [0,1,0,0] × W^Q = second row of W^Q = [0, 1, 0, 1]

For "sat":
q₃ = [0,0,1,0] × W^Q = third row of W^Q = [0, 0, 1, 1]

Q = [1  0  0  1]
    [0  1  0  1]
    [0  0  1  1]
```

Similarly:

```
K = X × W^K = [0  1  1  0]
              [1  0  1  0]
              [1  1  0  0]

V = X × W^V = [1  1  0  0]
              [0  1  1  0]
              [0  0  1  1]
```

#### Compute Attention Scores

```
QK^T = [1  0  0  1] × [0  1  1]^T
       [0  1  0  1]   [1  0  1]
       [0  0  1  1]   [1  1  0]
                      [0  0  0]

QK^T = [0  1  1]
       [1  0  1]
       [1  1  0]

Interpretation:
Row 1 ("The"): scores [0, 1, 1] → attends to "cat" and "sat" equally
Row 2 ("cat"): scores [1, 0, 1] → attends to "The" and "sat" equally
Row 3 ("sat"): scores [1, 1, 0] → attends to "The" and "cat" equally
```

#### Apply Softmax

```
Row 1: [0, 1, 1]
exp: [1.0, 2.718, 2.718]
sum: 6.436
softmax: [0.155, 0.422, 0.422]

Row 2: [1, 0, 1]
softmax: [0.422, 0.155, 0.422]

Row 3: [1, 1, 0]
softmax: [0.422, 0.422, 0.155]

Attention matrix:
A = [0.155  0.422  0.422]
    [0.422  0.155  0.422]
    [0.422  0.422  0.155]
```

**Patterns observed:**

```
Diagonal (self-attention): ~0.155 (lower)
Off-diagonal (cross-attention): ~0.422 (higher)

Each word attends more to OTHER words than to itself
This is just due to our specific weight matrices
```

#### Compute Output

```
Output = A × V

     = [0.155  0.422  0.422] × [1  1  0  0]
       [0.422  0.155  0.422]   [0  1  1  0]
       [0.422  0.422  0.155]   [0  0  1  1]

Position 1 output:
= 0.155×[1,1,0,0] + 0.422×[0,1,1,0] + 0.422×[0,0,1,1]
= [0.155, 0.155, 0, 0] + [0, 0.422, 0.422, 0] + [0, 0, 0.422, 0.422]
= [0.155, 0.577, 0.844, 0.422]

Similarly for positions 2 and 3:

Output = [0.155  0.577  0.844  0.422]
         [0.422  0.577  0.577  0.422]
         [0.422  0.844  0.577  0.155]
```

---

## Detailed Examples

### Example 3: Pronoun Resolution

**Sentence:** "The cat ate the fish because it was hungry"

**Question:** What does "it" refer to?

```
Positions:
1: "The"
2: "cat"
3: "ate"
4: "the"
5: "fish"
6: "because"
7: "it"
8: "was"
9: "hungry"
```

**Self-attention for position 7 ("it"):**

```
Query: q₇ = representation of "it"

Compute similarity with all keys:
Score(q₇, k₁) = 0.2  ("The")
Score(q₇, k₂) = 0.9  ("cat")    ← High!
Score(q₇, k₃) = 0.3  ("ate")
Score(q₇, k₄) = 0.1  ("the")
Score(q₇, k₅) = 0.4  ("fish")
Score(q₇, k₆) = 0.2  ("because")
Score(q₇, k₇) = 0.3  ("it")
Score(q₇, k₈) = 0.1  ("was")
Score(q₇, k₉) = 0.7  ("hungry")  ← Related

After softmax:
α₇,₁ = 0.03  ("The")
α₇,₂ = 0.45  ("cat")    ← Highest attention
α₇,₃ = 0.04  ("ate")
α₇,₄ = 0.02  ("the")
α₇,₅ = 0.08  ("fish")
α₇,₆ = 0.03  ("because")
α₇,₇ = 0.04  ("it")
α₇,₈ = 0.02  ("was")
α₇,₉ = 0.29  ("hungry")

Output for "it":
output₇ = Σ α₇,ᵢ × vᵢ
        ≈ 0.45×v₂ + 0.29×v₉ + small contributions from others
        ≈ mostly "cat" representation + some "hungry"

Result: "it" now has strong information about "cat"
→ Model learns "it" refers to "cat"
```

### Example 4: Dependency Parsing

**Sentence:** "The big red car"

**Attention pattern learned:**

```
Position 2 ("big"):
Attends to:
- "car" (0.6) ← What it modifies
- "big" (0.2) ← Self
- "red" (0.1) ← Other adjective
- "The" (0.1)

Position 3 ("red"):
Attends to:
- "car" (0.7) ← What it modifies
- "red" (0.2) ← Self
- "big" (0.1)

Position 4 ("car"):
Attends to:
- "car" (0.4) ← Self
- "big" (0.3) ← Modifier
- "red" (0.2) ← Modifier
- "The" (0.1)

Pattern: Adjectives attend to their noun
         Noun attends to its modifiers
```

### Example 5: Long-Range Dependencies

**Sentence:** "The keys, which were on the table near the door, are missing"

**Traditional RNN problem:**

```
"The keys" ← ... many words ... → "are"

RNN must remember "keys" through entire clause
Information degrades over distance
```

**Self-attention solution:**

```
Position of "are":
Query: q_are = "I'm a verb, find my subject"

Attention scores:
"The": 0.05
"keys": 0.85  ← Direct connection!
"which": 0.02
"were": 0.03
"on": 0.01
...
"are": 0.02

Direct path from "are" to "keys"
No degradation over distance
```

---

## Variants of Self-Attention

### 1. Masked Self-Attention (Causal)

**Purpose:** Prevent attending to future positions (for autoregressive models)

```
Standard attention:
Position 2 can attend to [1, 2, 3, 4, ...]

Masked attention:
Position 2 can only attend to [1, 2]
```

**Implementation:**

```
1. Compute QK^T normally

2. Apply mask:
   Mask = [[ 0  -∞  -∞  -∞]
           [ 0   0  -∞  -∞]
           [ 0   0   0  -∞]
           [ 0   0   0   0]]

3. Masked scores = QK^T + Mask

4. Softmax converts -∞ to 0

Result:
A = [[1.0  0    0    0  ]
     [0.3  0.7  0    0  ]
     [0.2  0.3  0.5  0  ]
     [0.1  0.2  0.3  0.4]]

Each position only attends to past
```

**Example:**

```
Generating: "The cat"

Position 1 ("The"):
Can attend to: ["The"]
Cannot see: ["cat"] (not generated yet)

Position 2 ("cat"):
Can attend to: ["The", "cat"]
Cannot see future tokens

This ensures autoregressive property
```

### 2. Local/Windowed Attention

**Purpose:** Reduce computation for very long sequences

```
Standard: Attend to all n positions → O(n²)
Local: Attend to window of w positions → O(n×w)

Example with window=3:
Position 5 attends to: [3, 4, 5, 6, 7]
Position 10 attends to: [8, 9, 10, 11, 12]
```

**Attention matrix (window=2):**

```
      1  2  3  4  5
  1 [[✓  ✓  ✗  ✗  ✗]
  2  [✓  ✓  ✓  ✗  ✗]
  3  [✗  ✓  ✓  ✓  ✗]
  4  [✗  ✗  ✓  ✓  ✓]
  5  [✗  ✗  ✗  ✓  ✓]]

✓ = can attend
✗ = cannot attend
```

### 3. Global + Local Attention

**Purpose:** Combine global context with local detail

```
Strategy:
- Some positions attend globally (to all positions)
- Most positions attend locally (to nearby positions)

Example:
Position 1: Global (special token like [CLS])
Positions 2-100: Local (window of 5)
Position 101: Global (another special token)

Benefit:
- Most computation is local: O(n×w)
- Global tokens propagate information
```

### 4. Cross-Attention

**Purpose:** Attend from one sequence to another

```
Encoder-Decoder:
Decoder attends to encoder output

Query: From decoder
Key, Value: From encoder

Example (translation):
Source: "Le chat"
Target: "The cat"

When generating "cat":
Q = decoder state for "cat"
K, V = encoder states for ["Le", "chat"]

Attention learns: "cat" ← "chat"
```

**Mathematics:**

```
Self-Attention:
Q, K, V all from same sequence X

Cross-Attention:
Q from sequence Y (decoder)
K, V from sequence X (encoder)

Attention(Q_Y, K_X, V_X) = softmax(Q_Y K_X^T / √d_k) V_X
```

### 5. Sparse Attention Patterns

**Purpose:** Structured sparsity for efficiency

**Strided Attention:**

```
Attend to every k-th position

Position 10 attends to: [0, 3, 6, 9, 10, 12, 15, ...]
(stride = 3)

Matrix:
     0  1  2  3  4  5  6  7  8  9
  0 [✓  ✗  ✗  ✓  ✗  ✗  ✓  ✗  ✗  ✓]
  3 [✓  ✗  ✗  ✓  ✗  ✗  ✓  ✗  ✗  ✓]
  6 [✓  ✗  ✗  ✓  ✗  ✗  ✓  ✗  ✗  ✓]
  9 [✓  ✗  ✗  ✓  ✗  ✗  ✓  ✗  ✗  ✓]
```

**Fixed Patterns (Longformer):**

```
Combine:
1. Local window (all positions)
2. Global attention (few positions)
3. Dilated (strided) attention

Result: O(n) complexity instead of O(n²)
```

---

## Multi-Head Self-Attention

### Why Multiple Heads?

**Problem with single head:**

```
One attention learns one type of relationship

Example: Might learn subject-verb
But miss determiner-noun, adjective-noun, etc.
```

**Solution: Multiple heads in parallel**

```
Head 1: Learns syntactic dependencies
Head 2: Learns semantic similarities
Head 3: Learns positional patterns
Head 4: Learns coreference
...
Head h: Learns different aspect

Combine all heads → Rich representation
```

### Mathematical Formulation

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

where:
headᵢ = Attention(QW_i^Q, KW_i^K, VW_i^V)

Parameters:
- h = number of heads
- d_model = model dimension
- d_k = d_model / h (dimension per head)
- W_i^Q, W_i^K, W_i^V ∈ ℝ^(d_model×d_k)
- W^O ∈ ℝ^(d_model×d_model)
```

### Detailed Example

**Configuration:**

```
d_model = 8
h = 2 heads
d_k = d_model / h = 4
```

**Input:**

```
X = [x₁]  where each xᵢ ∈ ℝ⁸
    [x₂]
    [x₃]

Shape: (3, 8)
```

#### Head 1

```
W₁^Q ∈ ℝ^(8×4) - projects to 4 dimensions

Q₁ = X W₁^Q ∈ ℝ^(3×4)
K₁ = X W₁^K ∈ ℝ^(3×4)
V₁ = X W₁^V ∈ ℝ^(3×4)

head₁ = Attention(Q₁, K₁, V₁) ∈ ℝ^(3×4)
```

#### Head 2

```
W₂^Q ∈ ℝ^(8×4) - different weights

Q₂ = X W₂^Q ∈ ℝ^(3×4)
K₂ = X W₂^K ∈ ℝ^(3×4)
V₂ = X W₂^V ∈ ℝ^(3×4)

head₂ = Attention(Q₂, K₂, V₂) ∈ ℝ^(3×4)
```

#### Concatenate

```
Concat(head₁, head₂) = [head₁ | head₂]

Shape: (3, 8)
- First 4 dimensions from head₁
- Last 4 dimensions from head₂
```

#### Final Projection

```
Output = Concat(head₁, head₂) W^O

where W^O ∈ ℝ^(8×8)

Output ∈ ℝ^(3×8)
```

### What Each Head Learns

**Empirical observations from trained models:**

```
Head 1: Short-range dependencies
[The] → [cat]
[big] → [car]

Attention pattern:
      The  cat  sat
The  [0.1  0.8  0.1]  ← Attends to next word
cat  [0.2  0.2  0.6]
sat  [0.1  0.3  0.6]

Head 2: Long-range dependencies
[The, ...]  → [are]  (subject-verb)

Attention pattern:
      The  cat  ...  are
The  [0.7  0.2  ...  0.1]
cat  [0.1  0.6  ...  0.3]
are  [0.6  0.2  ...  0.2]  ← Attends back to "cat"

Head 3: Positional patterns
Attends to similar positions

Head 4: Semantic similarity
Words with similar meanings attend to each other
```

### Computational Details

```
Parameters per head:
- W^Q: d_model × d_k = 8 × 4 = 32
- W^K: d_model × d_k = 8 × 4 = 32
- W^V: d_model × d_k = 8 × 4 = 32
Total per head: 96 parameters

For h heads:
96 × h parameters

Output projection:
W^O: d_model × d_model = 8 × 8 = 64

Total: 96×h + 64 = 96×2 + 64 = 256 parameters
```

---

## Computational Complexity

### Time Complexity

**Self-Attention:**

```
Step 1: QK^T
- Q ∈ ℝ^(n×d), K^T ∈ ℝ^(d×n)
- Complexity: O(n² × d)

Step 2: Softmax
- Applied to (n×n) matrix
- Complexity: O(n²)

Step 3: Multiply by V
- (n×n) × (n×d)
- Complexity: O(n² × d)

Total: O(n²d)
```

**Comparison with RNN:**

```
RNN: O(n × d²)
- Process n steps
- Each step: matrix multiply d×d

Self-Attention: O(n² × d)

For typical values:
- n < 1000
- d > 1000
- n² × d < n × d²

Self-attention can be faster when parallelized
```

### Space Complexity

```
Attention matrix: O(n²)
- Store attention weights for all pairs
- Bottleneck for long sequences

Activations: O(n × d)
Q, K, V: O(n × d) each

Total: O(n² + n×d)

For long sequences (n > 10,000):
O(n²) dominates → memory problem
```

### Optimization Techniques

#### 1. Gradient Checkpointing

```
Trade computation for memory:
- Don't store all intermediate activations
- Recompute during backward pass

Memory: O(√n) instead of O(n)
Compute: ~1.5× slower
```

#### 2. Flash Attention

```
Optimize attention computation:
- Fuse operations
- Use fast SRAM instead of slow HBM
- Block-wise computation

Speed: 2-4× faster
Memory: Same or less
Quality: Exact (not approximate)
```

#### 3. Approximations

```
Linear Attention:
- Approximate softmax(QK^T)V
- Use kernel methods
- Complexity: O(n × d²) instead of O(n²×d)

When beneficial:
- n >> d (very long sequences)
- Quality trade-off acceptable
```

---

## Practical Applications

### 1. Machine Translation

```
Input: "Le chat mange"
Output: "The cat eats"

Self-attention learns:
- "Le" ← determiner pattern
- "chat" ← noun, subject
- "mange" ← verb, agrees with subject

Cross-attention learns:
- "The" ← "Le"
- "cat" ← "chat"
- "eats" ← "mange"
```

### 2. Question Answering

```
Context: "Paris is the capital of France. It has 2 million residents."
Question: "What is the capital of France?"

Self-attention in context:
- "It" attends to "Paris"
- "capital" attends to "Paris" and "France"

Attention between question and context:
- "capital" in question → "capital" in context
- "France" in question → "France" in context
→ Extract "Paris"
```

### 3. Text Summarization

```
Document: Long article about climate change

Self-attention identifies:
- Main topics (high attention from many positions)
- Supporting details (moderate attention)
- Redundant information (similar attention patterns)

Generate summary:
- Focus on high-attention sentences
- Maintain coherence through attention patterns
```

### 4. Sentiment Analysis

```
Sentence: "The movie was not bad, actually quite good"

Self-attention learns:
- "not" attends to "bad" → negation
- "actually" attends to "not bad" and "good" → contrast
- "quite" attends to "good" → intensifier

Final representation captures:
Positive sentiment (despite "bad" word present)
```

### 5. Code Understanding

```
Code: "for i in range(len(array)): result.append(array[i])"

Self-attention:
- "i" attends to "range", "len", "array[i]"
- "array[i]" attends to "array", "i", "append"
- "result.append" attends to loop context

Understanding:
Loop iterates over array indices
Each element added to result
```

---

## Common Pitfalls

### 1. Attention is Not Explanation

```
❌ Wrong: "High attention = model uses this information"
✓ Right: "High attention = one signal among many"

Attention weights show where model looks
But not necessarily what it learns or how it decides

Multiple heads may contradict each other
Final output also depends on feed-forward layers
```

### 2. Computational Cost

```
❌ Problem: Using self-attention on very long sequences

Example:
n = 100,000 tokens
Attention matrix: 100,000 × 100,000 = 10 billion entries
Memory: 40GB (at 4 bytes per entry)

✓ Solutions:
- Sparse attention patterns
- Local windows
- Linear attention approximations
- Hierarchical models
```

### 3. Lack of Inductive Bias

```
❌ Problem: Self-attention has no built-in positional awareness

Without positional encoding:
"cat dog" = "dog cat"
Order doesn't matter → bad for language

✓ Solution:
Add positional encoding
sin/cos functions or learned embeddings
```

### 4. Training Instability

```
❌ Problem: Large attention scores → exploding/vanishing gradients

Without scaling:
d_k = 512
Dot products can be ±100
Softmax saturates

✓ Solution:
Scale by √d_k
Stabilizes gradient flow
```

### 5. Overfitting on Small Data

```
❌ Problem: Many parameters, small dataset

Multi-head attention has many parameters:
4 × d_model² per head × h heads

With limited data:
Model memorizes instead of generalizing

✓ Solutions:
- Dropout (0.1-0.3)
- Pre-training on large corpus
- Regularization
- Data augmentation
```

---

## Advanced Topics

### Attention Visualization

```
Tools for understanding learned attention:
- BertViz
- Attention maps
- Head importance analysis

Example visualization:
Position → Position attention matrix
Darker = stronger attention

     The  cat  sat  on  the  mat
The  [██  ░░  ░░  ░░  ░░  ░░]
cat  [██  ██  ░░  ░░  ░░  ░░]
sat  [░░  ██  ██  ██  ░░  ░░]
on   [░░  ░░  ░░  ██  ██  ██]
the  [░░  ░░  ░░  ░░  ██  ░░]
mat  [░░  ░░  ░░  ░░  ██  ██]
```

### Interpretability

```
Questions:
1. What linguistic phenomena does each head capture?
2. Are heads redundant?
3. Can we prune heads without losing performance?

Findings:
- Different heads specialize
- Some heads more important than others
- Can often remove 20-30% of heads
```

---

## Summary

### Key Concepts

```
1. Self-Attention Mechanism:
   - Each position attends to all positions
   - Attention(Q, K, V) = softmax(QK^T/√d_k)V
   - Enables parallel processing

2. Components:
   - Query: What am I looking for?
   - Key: What do I offer?
   - Value: What is my content?

3. Multi-Head:
   - Multiple attention patterns
   - Different aspects in parallel
   - Concat and project

4. Complexity:
   - Time: O(n²d)
   - Space: O(n²)
   - Bottleneck for long sequences
```

### When to Use Self-Attention

```
✓ Good for:
- Variable-length sequences
- Long-range dependencies
- Parallel processing
- Rich contextual representations

✗ Consider alternatives for:
- Very long sequences (>10K tokens)
- Extremely limited compute
- When position/order is paramount
- Sequential processing constraints
```

### Comparison with Other Mechanisms

```
RNN:
+ Sequential, lower memory
- Slow, limited dependencies

CNN:
+ Fast, local patterns
- Fixed receptive field

Self-Attention:
+ Parallel, long-range
- High memory, O(n²)

Combination often best:
Attention for global + CNN for local
```

---
