# Causal Attention: Complete Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem: Why Causal?](#the-problem-why-causal)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Detailed Examples](#detailed-examples)
6. [Implementation Variants](#implementation-variants)
7. [Comparison with Bidirectional Attention](#comparison-with-bidirectional-attention)
8. [Training vs Inference](#training-vs-inference)
9. [Practical Applications](#practical-applications)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

### What is Causal Attention?

**Causal Attention** (also called **Masked Attention** or **Autoregressive Attention**) is a variant of self-attention where each position can only attend to **previous positions** (including itself), not future positions.

**Key Constraint:**
> Position i can only look at positions 1, 2, ..., i
> Position i CANNOT look at positions i+1, i+2, ..., n

### Visual Representation

```
Standard Self-Attention (Bidirectional):
Every position can see every position

     1  2  3  4  5
  1 [✓  ✓  ✓  ✓  ✓]
  2 [✓  ✓  ✓  ✓  ✓]
  3 [✓  ✓  ✓  ✓  ✓]
  4 [✓  ✓  ✓  ✓  ✓]
  5 [✓  ✓  ✓  ✓  ✓]

Causal Attention (Autoregressive):
Each position can only see current and previous

     1  2  3  4  5
  1 [✓  ✗  ✗  ✗  ✗]  ← Position 1 sees only itself
  2 [✓  ✓  ✗  ✗  ✗]  ← Position 2 sees 1,2
  3 [✓  ✓  ✓  ✗  ✗]  ← Position 3 sees 1,2,3
  4 [✓  ✓  ✓  ✓  ✗]  ← Position 4 sees 1,2,3,4
  5 [✓  ✓  ✓  ✓  ✓]  ← Position 5 sees all

Pattern: Lower triangular matrix
```

### Why "Causal"?

**Causal** means respecting the flow of time/causality:
- Past can influence present
- Present CANNOT influence past
- Future CANNOT influence present

**Example:**
```
When generating: "The cat sat on"

At position 4 ("on"):
- Can see: "The", "cat", "sat", "on" ✓
- Cannot see: future words (not yet generated) ✗

This maintains causal order: past → present → future
```

---

## The Problem: Why Causal?

### Problem 1: Information Leakage

**Without causal masking:**

```
Task: Predict next word

Input: "The cat sat on the"
Target: "mat"

During training, if we use bidirectional attention:
When predicting "the" → model can see "mat" (the answer!)
→ Model cheats by looking at future
→ Won't work at inference time (future unknown)
```

**With causal masking:**

```
When predicting "the":
- Can see: "The cat sat on"
- Cannot see: "mat"
→ Must learn from past context only
→ Matches inference scenario
```

### Problem 2: Autoregressive Generation

**Goal:** Generate text one token at a time

```
Step 1: Generate token 1 given nothing
Step 2: Generate token 2 given token 1
Step 3: Generate token 3 given tokens 1,2
...
Step t: Generate token t given tokens 1,2,...,t-1
```

**Challenge:** Each step sees different amount of context

**Solution:** Causal attention ensures each position only uses available context

### Problem 3: Training Efficiency

**Without parallelization:**

```
Sequential generation:
Step 1: Predict word 1 → forward pass
Step 2: Predict word 2 → forward pass
Step 3: Predict word 3 → forward pass
...
Total: n forward passes for sequence length n
```

**With causal attention:**

```
Parallel training:
Single forward pass computes all predictions simultaneously
- Position 1 predicts word 2 (sees word 1)
- Position 2 predicts word 3 (sees words 1,2)
- Position 3 predicts word 4 (sees words 1,2,3)
...

Total: 1 forward pass for entire sequence
Speedup: n× faster training
```

### Use Cases

**Causal Attention is for:**
- ✓ Language modeling (GPT, GPT-2, GPT-3, GPT-4)
- ✓ Text generation
- ✓ Autoregressive models
- ✓ Dialogue systems
- ✓ Code generation

**Bidirectional Attention is for:**
- ✓ Text classification (BERT)
- ✓ Question answering (BERT)
- ✓ Named entity recognition
- ✓ Sentence embeddings
- ✓ Understanding tasks (not generation)

---

## Mathematical Foundation

### Standard Self-Attention (Review)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
- Q ∈ ℝ^(n×d_k): Queries
- K ∈ ℝ^(n×d_k): Keys  
- V ∈ ℝ^(n×d_v): Values
- n: sequence length
- d_k: key/query dimension
```

### Causal Attention Formula

```
CausalAttention(Q, K, V) = softmax((QK^T + M) / √d_k) V

where M is the causal mask:
M ∈ ℝ^(n×n)

M_ij = { 0     if i ≥ j  (can attend)
       { -∞   if i < j  (cannot attend)
```

### The Causal Mask in Detail

**Mask Matrix Structure:**

```
For n = 5:

M = [[ 0  -∞  -∞  -∞  -∞]
     [ 0   0  -∞  -∞  -∞]
     [ 0   0   0  -∞  -∞]
     [ 0   0   0   0  -∞]
     [ 0   0   0   0   0]]

Upper triangular part: -∞ (masked)
Lower triangular part (including diagonal): 0 (allowed)
```

**Why -∞?**

```
Softmax converts -∞ to 0:

Example:
scores = [2, 4, -∞, 1]

exp(scores) = [7.39, 54.6, 0, 2.72]

softmax = [7.39, 54.6, 0, 2.72] / 64.71
        = [0.114, 0.844, 0.000, 0.042]

-∞ → 0 after softmax (no attention to those positions)
```

### Complete Mathematical Flow

**Step-by-step:**

```
1. Compute Q, K, V:
   Q = XW^Q ∈ ℝ^(n×d_k)
   K = XW^K ∈ ℝ^(n×d_k)
   V = XW^V ∈ ℝ^(n×d_v)

2. Compute attention scores:
   S = QK^T ∈ ℝ^(n×n)

3. Apply causal mask:
   S_masked = S + M
   
   Example:
   S = [[5  3  2  1]      M = [[ 0  -∞  -∞  -∞]
        [4  6  3  2]            [ 0   0  -∞  -∞]
        [3  5  7  4]            [ 0   0   0  -∞]
        [2  4  6  8]]           [ 0   0   0   0]]
   
   S_masked = [[5  -∞  -∞  -∞]
               [4   6  -∞  -∞]
               [3   5   7  -∞]
               [2   4   6   8]]

4. Scale:
   S_scaled = S_masked / √d_k

5. Softmax (row-wise):
   A = softmax(S_scaled)
   
   Example for row 1: [5, -∞, -∞, -∞]
   → softmax([5]) = [1.0]
   
   Row 2: [4, 6, -∞, -∞]
   → softmax([4, 6]) = [0.12, 0.88]
   
   Row 3: [3, 5, 7, -∞]
   → softmax([3, 5, 7]) = [0.09, 0.24, 0.67]

6. Apply to values:
   Output = A V ∈ ℝ^(n×d_v)
```

---

## Step-by-Step Walkthrough

### Example 1: Tiny Causal Attention (3 tokens)

**Setup:**

```
Sequence: "cat sat mat"
n = 3
d = 4 (simplified dimension)
```

**Input embeddings:**

```
X = [x₁]   [1  0  1  0]  ← "cat"
    [x₂] = [0  1  0  1]  ← "sat"
    [x₃]   [1  1  0  0]  ← "mat"

Shape: (3, 4)
```

**Weight matrices (using identity for simplicity):**

```
W^Q = W^K = W^V = I₄ₓ₄

Therefore: Q = K = V = X
```

#### Step 1: Compute QK^T

```
Q = [1  0  1  0]    K^T = [1  0  1]
    [0  1  0  1]          [0  1  1]
    [1  1  0  0]          [1  0  0]
                          [0  1  0]

QK^T = [1  0  1  0]   [1  0  1]
       [0  1  0  1] × [0  1  1]
       [1  1  0  0]   [1  0  0]
                      [0  1  0]

     = [1×1+0×0+1×1+0×0  1×0+0×1+1×0+0×1  1×1+0×1+1×0+0×0]
       [0×1+1×0+0×1+1×0  0×0+1×1+0×0+1×1  0×1+1×1+0×0+1×0]
       [1×1+1×0+0×1+0×0  1×0+1×1+0×0+0×1  1×1+1×1+0×0+0×0]

     = [2  0  1]
       [0  2  1]
       [1  1  2]
```

**Interpretation (before masking):**

```
Position 1 ("cat"):
- Score with "cat" (itself): 2
- Score with "sat": 0
- Score with "mat": 1

Position 2 ("sat"):
- Score with "cat": 0
- Score with "sat" (itself): 2
- Score with "mat": 1

Position 3 ("mat"):
- Score with "cat": 1
- Score with "sat": 1
- Score with "mat" (itself): 2
```

#### Step 2: Apply Causal Mask

```
Mask M = [[ 0  -∞  -∞]
          [ 0   0  -∞]
          [ 0   0   0]]

Masked scores = QK^T + M

              = [2  0  1]   [[ 0  -∞  -∞]
                [0  2  1] + [ 0   0  -∞]
                [1  1  2]   [ 0   0   0]]

              = [[2  -∞  -∞]
                 [0   2  -∞]
                 [1   1   2]]
```

**After masking:**

```
Position 1: [2, -∞, -∞] → Can only see "cat"
Position 2: [0, 2, -∞]  → Can see "cat", "sat"
Position 3: [1, 1, 2]   → Can see all three
```

#### Step 3: Scale

```
d_k = 4
√d_k = 2

Scaled = [[2/2  -∞   -∞ ]
          [0/2  2/2  -∞ ]
          [1/2  1/2  2/2]]

       = [[1.0  -∞   -∞ ]
          [0.0  1.0  -∞ ]
          [0.5  0.5  1.0]]
```

#### Step 4: Softmax

```
Row 1: [1.0, -∞, -∞]
Only one valid value
softmax([1.0]) = [1.0]
Full row: [1.0, 0.0, 0.0]

Row 2: [0.0, 1.0, -∞]
exp([0.0, 1.0]) = [1.0, 2.718]
sum = 3.718
softmax = [1.0/3.718, 2.718/3.718]
        = [0.269, 0.731]
Full row: [0.269, 0.731, 0.0]

Row 3: [0.5, 0.5, 1.0]
exp([0.5, 0.5, 1.0]) = [1.649, 1.649, 2.718]
sum = 6.016
softmax = [1.649/6.016, 1.649/6.016, 2.718/6.016]
        = [0.274, 0.274, 0.452]

Attention matrix A:
A = [[1.000  0.000  0.000]
     [0.269  0.731  0.000]
     [0.274  0.274  0.452]]
```

**Interpretation:**

```
Position 1 ("cat"):
- 100% attention to itself (only option)

Position 2 ("sat"):
- 26.9% attention to "cat"
- 73.1% attention to itself
- 0% to "mat" (masked)

Position 3 ("mat"):
- 27.4% to "cat"
- 27.4% to "sat"
- 45.2% to itself
```

#### Step 5: Apply to Values

```
V = [1  0  1  0]
    [0  1  0  1]
    [1  1  0  0]

Output = A × V

Position 1 output:
= 1.0×[1,0,1,0] + 0.0×[0,1,0,1] + 0.0×[1,1,0,0]
= [1, 0, 1, 0]
(Unchanged - only sees itself)

Position 2 output:
= 0.269×[1,0,1,0] + 0.731×[0,1,0,1] + 0.0×[1,1,0,0]
= [0.269, 0, 0.269, 0] + [0, 0.731, 0, 0.731]
= [0.269, 0.731, 0.269, 0.731]

Position 3 output:
= 0.274×[1,0,1,0] + 0.274×[0,1,0,1] + 0.452×[1,1,0,0]
= [0.274, 0, 0.274, 0] + [0, 0.274, 0, 0.274] + [0.452, 0.452, 0, 0]
= [0.726, 0.726, 0.274, 0.274]

Final Output:
Output = [[1.000, 0.000, 1.000, 0.000]
          [0.269, 0.731, 0.269, 0.731]
          [0.726, 0.726, 0.274, 0.274]]
```

**Analysis:**

```
"cat" (position 1):
- Only saw itself → output = input
- No context from other words

"sat" (position 2):
- Saw "cat" and itself
- Output blends both representations
- Gained information from previous word

"mat" (position 3):
- Saw all previous words
- Output is weighted combination
- Rich contextual representation
```

---

## Detailed Examples

### Example 2: Language Modeling

**Task:** Predict next word

```
Input sequence: "The cat sat on"
Target: "the"

Training with causal attention:

Position 0: Input: <START>     → Predict: "The"
           Can see: <START>
           
Position 1: Input: "The"       → Predict: "cat"
           Can see: <START>, "The"
           
Position 2: Input: "cat"       → Predict: "sat"
           Can see: <START>, "The", "cat"
           
Position 3: Input: "sat"       → Predict: "on"
           Can see: <START>, "The", "cat", "sat"
           
Position 4: Input: "on"        → Predict: "the"
           Can see: <START>, "The", "cat", "sat", "on"
```

**Attention Pattern:**

```
Causal mask for 5 positions:

         <S>  The  cat  sat  on
<START>  [ 1    0    0    0   0 ]
The      [ 1    1    0    0   0 ]
cat      [ 1    1    1    0   0 ]
sat      [ 1    1    1    1   0 ]
on       [ 1    1    1    1   1 ]

After softmax (example learned weights):

         <S>  The  cat  sat  on
<START>  [1.0  0    0    0   0  ]
The      [0.2  0.8  0    0   0  ]
cat      [0.1  0.3  0.6  0   0  ]
sat      [0.1  0.2  0.3  0.4 0  ]
on       [0.1  0.1  0.2  0.3 0.3]
```

### Example 3: Generation Process

**Generating:** "The cat sat"

```
Step 1: Generate first word
Input: <START>
Attention: Only on <START>
Output: "The"

Step 2: Generate second word
Input: <START>, "The"
Attention pattern:
  Position 0 (<START>): [1.0, 0.0]
  Position 1 ("The"):   [0.3, 0.7]
Output: "cat"

Step 3: Generate third word
Input: <START>, "The", "cat"
Attention pattern:
  Position 0: [1.0, 0.0, 0.0]
  Position 1: [0.3, 0.7, 0.0]
  Position 2: [0.2, 0.3, 0.5]
Output: "sat"

Each step only uses previously generated tokens
```

### Example 4: Comparison at Each Position

**Sentence:** "I love machine learning"

**Bidirectional Attention (BERT-style):**

```
Position "love":
Can attend to: ["I", "love", "machine", "learning"]
Attention: [0.2, 0.3, 0.3, 0.2]
Uses full sentence context

Position "machine":
Can attend to: ["I", "love", "machine", "learning"]
Attention: [0.1, 0.2, 0.5, 0.2]
Uses full sentence context
```

**Causal Attention (GPT-style):**

```
Position "love":
Can attend to: ["I", "love"]
Attention: [0.4, 0.6, 0.0, 0.0]
Only past context

Position "machine":
Can attend to: ["I", "love", "machine"]
Attention: [0.2, 0.3, 0.5, 0.0]
Only past context
```

### Example 5: Training vs Inference Consistency

**Training (with causal masking):**

```
Batch: ["The cat sat on the mat"]

Forward pass computes all positions at once:

Position 1 predicts "cat"    using ["The"]
Position 2 predicts "sat"    using ["The", "cat"]
Position 3 predicts "on"     using ["The", "cat", "sat"]
Position 4 predicts "the"    using ["The", "cat", "sat", "on"]
Position 5 predicts "mat"    using ["The", "cat", "sat", "on", "the"]

All predictions in single forward pass (parallel)
Causal mask ensures each sees only past
```

**Inference (sequential generation):**

```
Step 1: Input: ["The"]
        Predict: "cat"

Step 2: Input: ["The", "cat"]
        Predict: "sat"

Step 3: Input: ["The", "cat", "sat"]
        Predict: "on"

Each step is separate forward pass (sequential)
Naturally only sees past (tokens not yet generated)
```

**Key Point:** Causal masking makes training match inference!

---

## Implementation Variants

### Variant 1: Additive Mask

```python
# Most common implementation
scores = QK^T / √d_k
scores = scores + mask  # mask has -∞ for future positions

# Example
scores = [[5.0, 3.0, 2.0],
          [4.0, 6.0, 3.0],
          [3.0, 5.0, 7.0]]

mask = [[  0,  -∞,  -∞],
        [  0,   0,  -∞],
        [  0,   0,   0]]

masked_scores = [[5.0, -∞,  -∞],
                 [4.0, 6.0, -∞],
                 [3.0, 5.0, 7.0]]
```

### Variant 2: Multiplicative Mask

```python
# Alternative implementation
mask = [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]]

attention_weights = softmax(scores) * mask
# Then renormalize
attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
```

### Variant 3: In-Place Masking

```python
# Set future positions to very negative value
scores[:, i, j] = -1e9 for j > i

# Or use masked_fill
scores = scores.masked_fill(mask == 0, -1e9)
```

### Variant 4: Attention Bias

```python
# Some implementations use attention bias
# Instead of additive mask on scores

attention_bias = create_causal_bias(seq_len)
# attention_bias is learned or fixed

scores = scores + attention_bias
```

---

## Comparison with Bidirectional Attention

### Attention Patterns

**Bidirectional (BERT):**

```
Sentence: "The cat sat"

         The  cat  sat
The     [0.3  0.4  0.3]  ← Sees all
cat     [0.2  0.5  0.3]  ← Sees all
sat     [0.3  0.3  0.4]  ← Sees all

Every position fully connected
Rich contextual understanding
BUT: Cannot generate autoregressively
```

**Causal (GPT):**

```
Sentence: "The cat sat"

         The  cat  sat
The     [1.0  0.0  0.0]  ← Sees only self
cat     [0.4  0.6  0.0]  ← Sees The, cat
sat     [0.2  0.3  0.5]  ← Sees all past

Lower triangular pattern
Each position sees only past
Can generate new tokens
```

### Information Flow

**Bidirectional:**

```
"The" ←→ "cat" ←→ "sat"
  ↑       ↑       ↑
  └───────┼───────┘
          └───────┘

All positions influence each other
Symmetric information flow
```

**Causal:**

```
"The" → "cat" → "sat"

Information flows forward only
Asymmetric
Past influences present
Present cannot influence past
```

### Use Case Comparison

| Task | Bidirectional | Causal |
|------|--------------|--------|
| Text Generation | ✗ | ✓ |
| Classification | ✓ | △ |
| Question Answering | ✓ | △ |
| Language Modeling | ✗ | ✓ |
| Sentiment Analysis | ✓ | △ |
| Translation (Encoder) | ✓ | ✗ |
| Translation (Decoder) | ✗ | ✓ |
| Summarization | ✓ | △ |
| Dialog Generation | ✗ | ✓ |

△ = Can work but not optimal

### Performance Characteristics

```
Training Speed:
Bidirectional: Fast (full parallelization)
Causal: Fast (parallel with masking)

Inference Speed:
Bidirectional: Fast (single pass)
Causal: Slow (sequential generation)

Context:
Bidirectional: Full context (past + future)
Causal: Partial context (past only)

Generation:
Bidirectional: Cannot generate
Causal: Native generation support
```

---

## Training vs Inference

### Training Mode

**Parallel Processing:**

```python
# Input: Full sequence
input_ids = ["The", "cat", "sat", "on", "the"]  # token IDs

# Forward pass
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# Compute loss for all positions at once
targets = ["cat", "sat", "on", "the", "mat"]
loss = cross_entropy(logits[:-1], targets)

# All predictions computed in parallel
# Causal mask ensures no future information
```

**Teacher Forcing:**

```
Position 1: Input "The"     → Predict "cat"   (truth: "cat")
Position 2: Input "cat"     → Predict "sat"   (truth: "sat")
Position 3: Input "sat"     → Predict "on"    (truth: "on")
...

Always use ground truth as input
Even if previous prediction was wrong
Fast and stable training
```

### Inference Mode

**Sequential Generation:**

```python
# Start with prompt
generated = ["The"]

for step in range(max_length):
    # Forward pass with current sequence
    logits = model(generated)  # [1, len(generated), vocab_size]
    
    # Get prediction for last position
    next_token_logits = logits[0, -1, :]
    
    # Sample next token
    next_token = sample(next_token_logits)
    
    # Append to sequence
    generated.append(next_token)
    
    if next_token == "<END>":
        break

# Sequential: n forward passes for length n
```

**No Teacher Forcing:**

```
Step 1: Input "The"           → Predict "cat"
Step 2: Input "The cat"       → Predict "sat"
Step 3: Input "The cat sat"   → Predict "on"

Each step uses model's own predictions
Errors can accumulate
Slower (sequential)
```

### KV Caching Optimization

**Problem:**

```
Step 1: Process ["The"]                    → 1 token
Step 2: Process ["The", "cat"]             → 2 tokens (redundant)
Step 3: Process ["The", "cat", "sat"]      → 3 tokens (redundant)

Recomputing keys and values for past tokens is wasteful
```

**Solution: KV Cache:**

```python
# Step 1
K_cache = compute_keys(["The"])
V_cache = compute_values(["The"])

# Step 2
K_new = compute_keys(["cat"])
V_new = compute_values(["cat"])
K_cache = concat([K_cache, K_new])  # Cache past keys
V_cache = concat([V_cache, V_new])  # Cache past values

# Step 3
K_new = compute_keys(["sat"])
V_new = compute_values(["sat"])
K_cache = concat([K_cache, K_new])
V_cache = concat([V_cache, V_new])

# Only compute Q for new token
# Use cached K, V for attention
```

**Speedup:**

```
Without cache: O(n²) for n tokens
With cache: O(n) for n tokens

~10-20× faster generation
```

---

## Practical Applications

### Application 1: GPT-style Language Models

```
Model: GPT-3

Architecture:
- 96 transformer layers
- Each layer has causal self-attention
- Generates text left-to-right

Example:
Prompt: "Once upon a time"
Generate: "there was a brave knight who lived in a castle..."

Each word generated only sees previous words
```

### Application 2: Code Generation (Copilot)

```
Input: "def fibonacci(n):"

Generation with causal attention:
def fibonacci(n):
    if n <= 1:          ← sees only "def fibonacci(n):"
        return n        ← sees previous lines
    return fibonacci(n-1) + fibonacci(n-2)  ← sees all above

Each line generated considering only previous lines
Maintains code consistency
```

### Application 3: Dialogue Systems

```
Conversation:
User: "What's the weather like?"
Bot: "It's sunny today."      ← sees only user message
User: "Should I bring an umbrella?"
Bot: "No need, it's clear."   ← sees all previous turns

Each response generated seeing only past conversation
Maintains context
Cannot "look ahead" to future turns
```

### Application 4: Story Completion

```
Prompt: "The detective walked into the dark room"

Causal generation:
"and noticed a shadow in the corner."     ← sees prompt
"His hand moved slowly to his gun."       ← sees prompt + prev sentence  
"Then he heard a sound behind him."       ← sees all previous

Story unfolds naturally
Each sentence builds on previous
No "future knowledge"
```

### Application 5: Translation (Decoder)

```
Source: "Le chat dort"  (French)
Encoder: Bidirectional attention on source

Decoder: Causal attention on target
<START> → "The"    (sees <START>)
"The" → "cat"      (sees <START>, "The")
"cat" → "sleeps"   (sees <START>, "The", "cat")

Target generated autoregressively with causal attention
Source attended with cross-attention (bidirectional OK)
```

---

## Advanced Topics

### 1. Relative Position in Causal Attention

**Standard positional encoding:**

```
Position 1: PE(1)
Position 2: PE(2)
Position 3: PE(3)

Absolute positions encoded
```

**Relative position in causal context:**

```
When position i attends to position j:
Relative distance: j - i

For causal: j ≤ i
So relative distance: j - i ≤ 0 (always non-positive)

Encode relative distances: -1, -2, -3, ...
```

**Implementation:**

```python
# For position i attending to position j
relative_position = j - i  # ≤ 0 for causal

# Use relative position embedding
relative_bias = relative_position_embedding[relative_position]

# Add to attention scores
scores[i, j] = scores[i, j] + relative_bias
```

### 2. Sliding Window Causal Attention

**Problem:** Very long sequences

```
Standard causal: Position 1000 can attend to all 1000 previous positions
Memory: O(n²) grows quadratically
```

**Solution: Windowed causal attention:**

```
Attend only to last w positions (e.g., w=256)

Position 1000 attends to: [745, 746, ..., 1000]
Not: [1, 2, 3, ..., 1000]

Memory: O(n×w) - linear in sequence length
```

**Mask pattern:**

```
Window = 3

      1  2  3  4  5  6
  1  [✓  ✗  ✗  ✗  ✗  ✗]
  2  [✓  ✓  ✗  ✗  ✗  ✗]
  3  [✓  ✓  ✓  ✗  ✗  ✗]
  4  [✗  ✓  ✓  ✓  ✗  ✗]  ← Window of 3
  5  [✗  ✗  ✓  ✓  ✓  ✗]  ← Slides
  6  [✗  ✗  ✗  ✓  ✓  ✓]

Band diagonal pattern
```

### 3. Prefix Causal Attention

**Use case:** Encoder-decoder with causal decoder

```
Prefix (input): "Translate to French: Hello"
Completion: "Bonjour"

Attention pattern:
- Prefix tokens: Can see all prefix (bidirectional within prefix)
- Completion tokens: Causal (see prefix + past completion)

      Prefix   Completion
      T  r  :  H  |  B  o  n
  T  [✓  ✓  ✓  ✓  |  ✗  ✗  ✗]
  r  [✓  ✓  ✓  ✓  |  ✗  ✗  ✗]
  :  [✓  ✓  ✓  ✓  |  ✗  ✗  ✗]
  H  [✓  ✓  ✓  ✓  |  ✗  ✗  ✗]
  ---|------------|----------
  B  [✓  ✓  ✓  ✓  |  ✓  ✗  ✗]  ← Causal in completion
  o  [✓  ✓  ✓  ✓  |  ✓  ✓  ✗]
  n  [✓  ✓  ✓  ✓  |  ✓  ✓  ✓]
```

### 4. Efficient Causal Masking

**Naive implementation:**

```python
# Create mask for every forward pass
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
```

**Optimized:**

```python
# Pre-compute and cache for max length
class CausalMask:
    def __init__(self, max_seq_len):
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('mask', mask)
    
    def get_mask(self, seq_len):
        return self.mask[:seq_len, :seq_len]

# Reuse cached mask
```

### 5. Causal with Padding

**Challenge:** Variable-length sequences with padding

```
Sequence 1: "The cat sat" + <PAD>
Sequence 2: "I love ML"

Need to mask:
1. Future positions (causal)
2. Padding positions
```

**Combined mask:**

```python
# Causal mask
causal_mask = torch.triu(torch.ones(n, n), diagonal=1) == 0

# Padding mask
padding_mask = (input_ids != PAD_TOKEN).unsqueeze(1)

# Combine
full_mask = causal_mask & padding_mask

# Apply
scores = scores.masked_fill(~full_mask, -1e9)
```

---

## Common Patterns and Best Practices

### Pattern 1: Training Loop

```python
for batch in dataloader:
    input_ids = batch['input_ids']      # [batch, seq_len]
    target_ids = batch['target_ids']    # [batch, seq_len]
    
    # Forward with causal attention (automatic via model)
    logits = model(input_ids)           # [batch, seq_len, vocab]
    
    # Shift for next-token prediction
    # Input:  "The cat sat"
    # Target: "cat sat <END>"
    shift_logits = logits[:, :-1]       # Remove last prediction
    shift_targets = target_ids[:, 1:]   # Remove first token
    
    # Loss
    loss = cross_entropy(shift_logits.reshape(-1, vocab_size),
                        shift_targets.reshape(-1))
    
    loss.backward()
    optimizer.step()
```

### Pattern 2: Generation with Sampling

```python
def generate(model, prompt, max_length, temperature=1.0):
    tokens = tokenize(prompt)
    
    for _ in range(max_length):
        # Forward pass
        logits = model(tokens)              # [1, len, vocab]
        next_logits = logits[0, -1] / temperature
        
        # Sample
        probs = softmax(next_logits)
        next_token = torch.multinomial(probs, 1)
        
        tokens = torch.cat([tokens, next_token])
        
        if next_token == END_TOKEN:
            break
    
    return tokens
```

### Pattern 3: Beam Search

```python
def beam_search(model, prompt, beam_size=5):
    # Initialize beams
    beams = [(tokenize(prompt), 0.0)]  # (tokens, score)
    
    for _ in range(max_length):
        candidates = []
        
        for tokens, score in beams:
            if tokens[-1] == END_TOKEN:
                candidates.append((tokens, score))
                continue
            
            # Get next token probabilities
            logits = model(tokens)
            probs = softmax(logits[0, -1])
            
            # Top-k candidates
            top_probs, top_indices = torch.topk(probs, beam_size)
            
            for prob, idx in zip(top_probs, top_indices):
                new_tokens = torch.cat([tokens, idx.unsqueeze(0)])
                new_score = score + torch.log(prob)
                candidates.append((new_tokens, new_score))
        
        # Keep top beam_size beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return beams[0][0]  # Best sequence
```

---

## Summary

### Key Concepts

```
1. Causal Constraint:
   - Position i can only attend to j where j ≤ i
   - Implements autoregressive property
   - Future cannot influence past

2. Mask Implementation:
   - Add -∞ to upper triangular
   - Softmax converts -∞ to 0
   - Lower triangular: allowed attention

3. Training vs Inference:
   - Training: Parallel with teacher forcing
   - Inference: Sequential generation
   - Causal mask ensures consistency

4. Applications:
   - Language modeling (GPT)
   - Text generation
   - Dialogue systems
   - Code generation
```

### Mathematical Core

```
CausalAttention(Q, K, V) = softmax((QK^T + M) / √d_k) V

where M_ij = { 0    if i ≥ j
             { -∞   if i < j

Result: Lower triangular attention matrix
```

### Comparison Summary

| Aspect | Bidirectional | Causal |
|--------|--------------|--------|
| Pattern | Full matrix | Lower triangular |
| Context | Past + Future | Past only |
| Generation | No | Yes |
| Training | Parallel | Parallel |
| Inference | Single pass | Sequential |
| Models | BERT, RoBERTa | GPT, GPT-2/3/4 |

### When to Use Causal Attention

```
✓ Use when:
- Generating sequences (text, code, music)
- Language modeling
- Autoregressive tasks
- Sequential decision making
- Future is unknown at inference

✗ Don't use when:
- Classification tasks
- Full context available
- Bidirectional understanding needed
- Not generating new content
```

---
