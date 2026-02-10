# Large Language Models (LLMs): Complete End-to-End Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Foundational Concepts](#foundational-concepts)
3. [Tokenization](#tokenization)
4. [Embeddings](#embeddings)
5. [Transformer Architecture](#transformer-architecture)
6. [Training Process](#training-process)
7. [Inference](#inference)
8. [Fine-tuning Methods](#fine-tuning-methods)
9. [Optimization Techniques](#optimization-techniques)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Advanced Topics](#advanced-topics)
12. [Practical Implementation](#practical-implementation)

---

## Introduction

### What is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained on vast amounts of text data to understand and generate human-like text. LLMs learn statistical patterns, relationships, and structures in language to perform tasks like:

- Text generation
- Translation
- Question answering
- Summarization
- Code generation
- Reasoning

### Evolution Timeline

```
2017: Transformer Architecture (Vaswani et al.)
2018: BERT (340M params), GPT-1 (117M params)
2019: GPT-2 (1.5B params), T5 (11B params)
2020: GPT-3 (175B params)
2021: GPT-3.5, Codex
2022: ChatGPT, LLaMA (7B-65B)
2023: GPT-4, Claude 2, LLaMA 2, Mistral
2024: Claude 3, GPT-4o, Gemini, LLaMA 3
2025: Claude 4, Gemini 2.0, DeepSeek-V3
```

### Key Characteristics

| Property | Range |
|----------|-------|
| Parameters | 1B - 1.8T |
| Training Data | 100B - 15T tokens |
| Context Window | 2K - 1M+ tokens |
| Training Cost | $1M - $100M+ |
| Training Time | Weeks - Months |

---

## Foundational Concepts

### 1. Neural Network Basics

#### Single Neuron
```
y = f(w·x + b)

where:
- x = input vector
- w = weight vector
- b = bias term
- f = activation function
```

#### Multi-Layer Network
```
Layer 1: h₁ = f₁(W₁·x + b₁)
Layer 2: h₂ = f₂(W₂·h₁ + b₂)
...
Output: y = fₙ(Wₙ·hₙ₋₁ + bₙ)
```

### 2. Activation Functions

#### Common Activations

**ReLU (Rectified Linear Unit):**
```
ReLU(x) = max(0, x)

Derivative: 
∂ReLU/∂x = 1 if x > 0, else 0
```

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x · Φ(x)
        = x · 0.5[1 + erf(x/√2)]

where Φ(x) is the Gaussian CDF
```

**Softmax (for output layer):**
```
Softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

Properties:
- Σᵢ Softmax(z)ᵢ = 1
- 0 ≤ Softmax(z)ᵢ ≤ 1
```

### 3. Probability and Information Theory

#### Cross-Entropy Loss
```
H(p, q) = -Σᵢ p(xᵢ) log q(xᵢ)

For language modeling:
L = -Σₜ log P(xₜ | x₁, ..., xₜ₋₁)
```

#### Perplexity
```
PPL = exp(H(p, q))
    = exp(-1/N Σᵢ log P(xᵢ))

Lower perplexity = better model
```

#### KL Divergence
```
DₖL(P || Q) = Σᵢ P(xᵢ) log(P(xᵢ)/Q(xᵢ))

Measures difference between distributions
```

---

## Tokenization

### What is Tokenization?

Tokenization converts raw text into discrete units (tokens) that the model can process.

```
Text: "Hello, world!"
Tokens: ["Hello", ",", " world", "!"]
IDs: [15496, 11, 995, 0]
```

### Tokenization Methods

#### 1. Word-Level Tokenization
```
Text: "The cat sat"
Tokens: ["The", "cat", "sat"]

Vocabulary size: ~50,000 - 100,000
Problem: Can't handle unknown words
```

#### 2. Character-Level Tokenization
```
Text: "cat"
Tokens: ["c", "a", "t"]

Vocabulary size: ~256 (ASCII)
Problem: Very long sequences
```

#### 3. Subword Tokenization

**Byte Pair Encoding (BPE):**

Algorithm:
```
1. Start with character vocabulary
2. Find most frequent adjacent pair
3. Merge pair into new token
4. Repeat until vocabulary size reached

Example:
Initial: ['l', 'o', 'w', 'e', 'r']
Merge 'e'+'r' → 'er': ['l', 'o', 'w', 'er']
Merge 'l'+'o' → 'lo': ['lo', 'w', 'er']
...
```

**WordPiece (BERT):**
```
Score(pair) = freq(pair) / (freq(first) × freq(second))

Maximize likelihood of training data
```

**SentencePiece (LLaMA, GPT):**
```
- Treats input as raw byte sequence
- Language agnostic
- Includes spaces as tokens
```

### Tokenization Mathematics

#### Vocabulary Size Trade-off
```
Sequence_Length = f(Vocab_Size)

Small vocab: Longer sequences, more computation
Large vocab: Shorter sequences, larger embedding matrix

Optimal vocab size: 30,000 - 100,000
```

#### Compression Ratio
```
Compression = Num_Characters / Num_Tokens

English: ~4.5 characters/token
Code: ~3.5 characters/token
```

### Example: GPT Tokenization

```python
Text: "I love learning about AI!"

# BPE tokenization
Tokens: ["I", " love", " learn", "ing", " about", " AI", "!"]
IDs: [40, 1842, 4745, 278, 546, 15592, 0]

# Vocabulary lookup
vocab = {
    "I": 40,
    " love": 1842,
    " learn": 4745,
    "ing": 278,
    ...
}
```

---

## Embeddings

### What are Embeddings?

Embeddings convert discrete tokens into continuous vector representations.

```
Token ID → Embedding Vector

"cat" (ID: 2053) → [0.23, -0.15, 0.87, ..., 0.42] ∈ ℝᵈ
```

### Token Embeddings

#### Embedding Matrix
```
E ∈ ℝ^(V×d)

where:
- V = vocabulary size (e.g., 50,000)
- d = embedding dimension (e.g., 768, 4096)

Embedding lookup:
xₑ = E[token_id]
```

#### Embedding Dimension

Common sizes:
- Small models: d = 768 (BERT-base)
- Medium models: d = 2048
- Large models: d = 4096, 8192, 12288

**Memory requirement:**
```
Embedding_Memory = V × d × bytes_per_param

Example (GPT-3):
50,257 × 12,288 × 2 bytes = ~1.2 GB
```

### Positional Embeddings

LLMs need to know token positions in the sequence.

#### Absolute Positional Encoding (Original Transformer)

Sinusoidal encoding:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
- pos = position in sequence
- i = dimension index
- d = embedding dimension
```

**Properties:**
- Deterministic (no learned parameters)
- PE(pos+k) can be expressed as linear function of PE(pos)
- Extrapolates to longer sequences

#### Learned Positional Embeddings (GPT)
```
P ∈ ℝ^(max_seq×d)

Position embedding:
xₚ = P[position]

Final embedding:
x = xₑ + xₚ
```

#### Rotary Position Embedding (RoPE)

Used in modern LLMs (LLaMA, GPT-NeoX):
```
RoPE rotates query and key vectors:

q̃ₘ = Rₘ qₘ
k̃ₙ = Rₙ kₙ

where Rₘ is rotation matrix for position m:

Rₘ = [
  [cos(mθ₀), -sin(mθ₀), 0, 0, ...]
  [sin(mθ₀), cos(mθ₀), 0, 0, ...]
  [0, 0, cos(mθ₁), -sin(mθ₁), ...]
  [0, 0, sin(mθ₁), cos(mθ₁), ...]
  ...
]

θᵢ = 10000^(-2i/d)
```

**Advantages:**
- Relative position information
- Better extrapolation to longer contexts
- Efficient computation

### Combining Embeddings

Final input representation:
```
X = Token_Embedding + Position_Embedding

X ∈ ℝ^(seq_len×d)
```

---

## Transformer Architecture

### Overview

The Transformer is the foundational architecture for modern LLMs.

```
Input Text
    ↓
Tokenization
    ↓
Embeddings (Token + Position)
    ↓
[Transformer Blocks] × N layers
    ↓
Output Logits
    ↓
Softmax → Probabilities
    ↓
Next Token Prediction
```

### Core Components

#### 1. Self-Attention Mechanism

**Purpose:** Allow each token to attend to all other tokens in the sequence.

**Single-Head Attention:**
```
Q = XWᵠ    (Query)
K = XWᴷ    (Key)
V = XWⱽ    (Value)

Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

where:
- X ∈ ℝ^(seq×d_model)
- Wᵠ, Wᴷ, Wⱽ ∈ ℝ^(d_model×dₖ)
- dₖ = dimension of key/query vectors
```

**Step-by-step calculation:**

```
1. Compute similarity scores:
   S = QKᵀ ∈ ℝ^(seq×seq)
   Sᵢⱼ = qᵢ · kⱼ (dot product)

2. Scale by √dₖ:
   S' = S / √dₖ
   (Prevents softmax saturation)

3. Apply softmax:
   A = softmax(S')
   Aᵢⱼ = exp(S'ᵢⱼ) / Σₖ exp(S'ᵢₖ)

4. Weighted sum of values:
   Output = AV
   Outputᵢ = Σⱼ Aᵢⱼvⱼ
```

**Why scaling by √dₖ?**
```
For random vectors q, k ∈ ℝᵈ:
E[q·k] = 0
Var[q·k] = d

Scaling by 1/√d normalizes variance to 1
```

#### 2. Multi-Head Attention

**Purpose:** Attend to different representation subspaces simultaneously.

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ

where:
headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)

Parameters:
- h = number of heads (e.g., 8, 16, 32)
- Wᵢᵠ, Wᵢᴷ, Wᵢⱽ ∈ ℝ^(d_model×dₖ)
- dₖ = d_model / h
- Wᴼ ∈ ℝ^(d_model×d_model)
```

**Example with numbers:**
```
d_model = 768
h = 12 heads
dₖ = 768/12 = 64

Each head works in 64-dimensional space
Concatenate 12 heads → 768 dimensions
Project with Wᴼ → 768 dimensions
```

**Parameter count:**
```
Params_MHA = 4 × d_model × d_model
           = 4 × d² (for Wᵠ, Wᴷ, Wⱽ, Wᴼ)

For d=768: 4 × 768² = 2.36M parameters
```

#### 3. Masked Self-Attention (Causal Attention)

For autoregressive models (GPT), prevent attending to future tokens:

```
Mask matrix M ∈ ℝ^(seq×seq):
M = [
  [0, -∞, -∞, -∞]
  [0,  0, -∞, -∞]
  [0,  0,  0, -∞]
  [0,  0,  0,  0]
]

Masked Attention:
A = softmax((QKᵀ/√dₖ) + M)V
```

Visual representation:
```
Token:  [The] [cat] [sat] [down]
  [The]   ✓    ✗     ✗     ✗
  [cat]   ✓    ✓     ✗     ✗
  [sat]   ✓    ✓     ✓     ✗
  [down]  ✓    ✓     ✓     ✓
```

#### 4. Feed-Forward Network (FFN)

Applied independently to each position:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

or with GELU:
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

where:
- W₁ ∈ ℝ^(d_model×d_ff)
- W₂ ∈ ℝ^(d_ff×d_model)
- d_ff = 4 × d_model (typically)
```

**Modern variant - SwiGLU:**
```
SwiGLU(x) = (Swish(xW₁) ⊙ (xW₂))W₃

where:
- Swish(x) = x · sigmoid(x)
- ⊙ = element-wise multiplication
```

**Parameter count:**
```
Params_FFN = d_model × d_ff + d_ff × d_model
           = 2 × d_model × d_ff
           = 2 × d × 4d = 8d²

For d=768: 8 × 768² = 4.72M parameters
```

#### 5. Layer Normalization

Normalize activations within each layer:

**LayerNorm:**
```
LN(x) = γ ⊙ (x - μ)/σ + β

where:
- μ = 1/d Σᵢ xᵢ (mean)
- σ = √(1/d Σᵢ (xᵢ - μ)²) (std dev)
- γ, β = learned scale and shift parameters
```

**RMSNorm (modern LLMs):**
```
RMSNorm(x) = x / RMS(x) · γ

where RMS(x) = √(1/d Σᵢ xᵢ²)

Advantages:
- 10-15% faster
- Simpler computation
- Better numerical stability
```

**Placement:**
```
Pre-LN (GPT-2, modern models):
x = x + Attention(LN(x))
x = x + FFN(LN(x))

Post-LN (original Transformer):
x = LN(x + Attention(x))
x = LN(x + FFN(x))
```

#### 6. Residual Connections

Skip connections that help gradient flow:

```
x_out = x_in + Layer(x_in)

Gradient flow:
∂L/∂x_in = ∂L/∂x_out × (1 + ∂Layer/∂x_in)

The "+1" ensures gradient can flow directly
```

**Benefits:**
- Prevents vanishing gradients
- Enables training of deep networks (100+ layers)
- Allows learning identity function

### Complete Transformer Block

```
Input: x ∈ ℝ^(seq×d)

1. Multi-head self-attention:
   x₁ = x + MultiHeadAttention(LayerNorm(x))

2. Feed-forward network:
   x₂ = x₁ + FFN(LayerNorm(x₁))

Output: x₂
```

### Full Architecture

#### Decoder-Only (GPT-style)

```
Architecture:
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
[Transformer Block] × N layers
    ↓
Layer Norm
    ↓
Linear(d_model → vocab_size)
    ↓
Softmax → Next token probabilities
```

**Layer count examples:**
- GPT-2 Small: 12 layers
- GPT-2 Large: 36 layers
- GPT-3: 96 layers
- LLaMA 70B: 80 layers
- GPT-4: ~120 layers (estimated)

#### Encoder-Only (BERT-style)

```
Used for: Classification, encoding, embeddings

Architecture:
Input Tokens
    ↓
Token + Position + Segment Embeddings
    ↓
[Transformer Block with bi-directional attention] × N
    ↓
Pooling / Classification head
```

#### Encoder-Decoder (T5-style)

```
Used for: Translation, summarization

Encoder: Process input
Decoder: Generate output with cross-attention
```

### Parameter Calculations

**Total parameters for decoder-only model:**

```
Params = Embedding_Params + Layer_Params × N + Output_Params

where:
Embedding_Params = V × d + max_seq × d
Layer_Params = 12 × d²  (4d² for MHA + 8d² for FFN)
Output_Params = d × V (often tied with embedding)

Example (GPT-2 Small):
d = 768, V = 50,257, N = 12

Embedding: 50,257 × 768 ≈ 38.6M
Layers: 12 × 12 × 768² ≈ 85.1M
Output: tied with embedding
Total: ≈ 124M parameters
```

---

## Training Process

### Training Objective

#### Next Token Prediction

The fundamental task for autoregressive LLMs:

```
Given sequence: x₁, x₂, ..., xₜ
Predict: xₜ₊₁

Probability:
P(xₜ₊₁ | x₁, ..., xₜ) = softmax(f(x₁, ..., xₜ))
```

#### Loss Function

**Cross-Entropy Loss:**
```
L = -1/T Σₜ log P(xₜ | x₁, ..., xₜ₋₁)

For single example:
L = -log P(x₂|x₁) - log P(x₃|x₁,x₂) - ... - log P(xₜ|x₁,...,xₜ₋₁)

For batch of B examples:
L = -1/(B×T) Σᵦ Σₜ log P(xₜ,ᵦ | x₁:ₜ₋₁,ᵦ)
```

**Detailed computation:**
```
1. Forward pass:
   logits = Model(x₁, ..., xₜ₋₁) ∈ ℝ^(T×V)

2. Apply softmax:
   P(·|context) = softmax(logits)

3. Compute loss:
   L = -Σₜ log P(xₜ|x₁:ₜ₋₁)

4. Backward pass:
   Compute ∂L/∂θ for all parameters θ

5. Update parameters:
   θ ← θ - α∇θ
```

### Training Data

#### Dataset Composition

Example (GPT-3 style):
```
Total tokens: ~500 billion

Sources:
- Common Crawl: 60% (410B tokens)
- WebText2: 19% (95B tokens)
- Books1: 8% (40B tokens)
- Books2: 8% (40B tokens)
- Wikipedia: 5% (25B tokens)
```

#### Data Processing Pipeline

```
1. Data Collection
   ↓
2. Filtering
   - Remove duplicates (MinHash, exact match)
   - Quality filtering (perplexity, classifiers)
   - Content filtering (toxicity, privacy)
   ↓
3. Deduplication
   - Document-level: exact/near duplicates
   - Sequence-level: repeated n-grams
   ↓
4. Tokenization
   - Apply BPE/SentencePiece
   - Create token sequences
   ↓
5. Batching
   - Group sequences by length
   - Create training batches
```

#### Quality Filtering

**Perplexity-based filtering:**
```
Keep document if:
PPL(doc, reference_model) < threshold

Lower perplexity → more similar to high-quality text
```

**Classifier-based filtering:**
```
P(high_quality | document) > threshold

Train classifier on curated high-quality examples
```

### Optimization

#### Gradient Descent

**Basic update rule:**
```
θₜ₊₁ = θₜ - α∇L(θₜ)

where:
- θ = parameters
- α = learning rate
- ∇L = gradient of loss
```

#### Stochastic Gradient Descent (SGD)

```
For each mini-batch:
  g = ∇L(θ; batch)
  θ ← θ - αg
```

#### Adam Optimizer

Most common optimizer for LLMs:

```
Initialize:
m₀ = 0  (first moment)
v₀ = 0  (second moment)

At step t:
gₜ = ∇L(θₜ)                      (gradient)
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ           (momentum)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²          (adaptive learning rate)

Bias correction:
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

Update:
θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)

Typical hyperparameters:
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸
```

#### AdamW (Weight Decay)

```
θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε) - αλθₜ

where λ = weight decay coefficient (0.01 - 0.1)

Decouples weight decay from gradient-based update
```

### Learning Rate Scheduling

#### Warmup + Cosine Decay

```
Phase 1 - Warmup (0 to warmup_steps):
lr(t) = lr_max × (t / warmup_steps)

Phase 2 - Cosine Decay:
lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(π·t'/T'))

where:
- t' = t - warmup_steps
- T' = total_steps - warmup_steps
- lr_min = 0.1 × lr_max (typically)
```

**Typical values:**
```
lr_max = 6e-4 (small models) to 3e-4 (large models)
warmup_steps = 2,000 - 10,000
total_steps = 100,000 - 1,000,000+
```

#### Inverse Square Root

```
lr(t) = lr_max × min(t⁻⁰·⁵, t × warmup⁻¹·⁵)
```

### Gradient Clipping

Prevent exploding gradients:

```
Global norm clipping:
g' = g × min(1, max_norm / ||g||)

where:
- g = gradient vector
- max_norm = 1.0 (typical)
- ||g|| = √(Σᵢ gᵢ²)
```

### Mixed Precision Training

#### FP16 Training

```
Forward pass: FP16
Loss computation: FP32
Backward pass: FP16 → FP32
Parameter update: FP32

Benefits:
- 2× faster training
- 2× less memory
- Requires loss scaling to prevent underflow
```

**Loss scaling:**
```
1. Scale loss by factor S (e.g., 2¹⁶)
2. Backward pass with scaled loss
3. Unscale gradients: g' = g / S
4. Update parameters
```

#### BF16 (Brain Float 16)

```
Format: 1 sign + 8 exponent + 7 mantissa bits
- Same range as FP32
- Lower precision than FP16
- No loss scaling needed

Preferred for modern hardware (A100, H100)
```

### Batch Size

#### Effective Batch Size

```
Effective_Batch = Batch_Per_GPU × Num_GPUs × Gradient_Accumulation

Example:
4 tokens × 8 GPUs × 16 accumulation = 512 effective batch
```

#### Gradient Accumulation

```
for i in range(accumulation_steps):
    loss = forward_backward(batch[i])
    loss.backward()  # accumulate gradients

# Update after accumulation
optimizer.step()
optimizer.zero_grad()
```

**Memory trade-off:**
```
Small batch: Less memory, more updates, noisier gradients
Large batch: More memory, fewer updates, stabler gradients

Optimal batch size: 256 - 4096 sequences (varies by model)
```

### Training Dynamics

#### Training Steps Calculation

```
Total_Tokens = Training_Data_Size
Tokens_Per_Step = Batch_Size × Sequence_Length
Steps = Total_Tokens / Tokens_Per_Step

Example:
300B tokens / (512 batch × 2048 seq) = 286,458 steps
```

#### Training Time Estimation

```
Time = Steps × Seconds_Per_Step

Seconds_Per_Step = (Forward + Backward + Update) / Throughput

Example (GPT-3):
~500,000 steps × 3-5 seconds ≈ 20-30 days on 1000+ GPUs
```

#### Compute Budget

**FLOPs for training:**
```
FLOPs ≈ 6 × Parameters × Tokens

For GPT-3 (175B params, 300B tokens):
FLOPs ≈ 6 × 175B × 300B
     ≈ 3.14 × 10²³ FLOPs
```

**GPU requirements:**
```
GPU_Hours = Total_FLOPs / (GPU_FLOPs_per_second × 3600)

For A100 (312 TFLOPS):
GPU_Hours ≈ 3.14×10²³ / (312×10¹² × 3600)
         ≈ 280,000 GPU-hours
```

**Cost estimation:**
```
Cost = GPU_Hours × Cost_Per_Hour

At $2/hour per A100:
Cost ≈ 280,000 × $2 = $560,000
```

---

## Inference

### Generation Process

#### Autoregressive Generation

```
1. Start with prompt: [x₁, x₂, ..., xₙ]

2. Generate tokens one by one:
   For t = n+1 to max_length:
       logits = Model(x₁, ..., xₜ₋₁)
       xₜ = Sample(logits)
       Append xₜ to sequence
       
3. Stop when:
   - End-of-sequence token generated
   - Maximum length reached
   - Custom stopping condition
```

### Sampling Strategies

#### Greedy Decoding

```
xₜ = argmax P(·|x₁:ₜ₋₁)

Always pick most probable token

Pros: Deterministic, simple
Cons: Repetitive, boring output
```

#### Sampling (Temperature)

```
P'(xᵢ) = exp(logits_i / T) / Σⱼ exp(logits_j / T)

xₜ ~ P'(·|x₁:ₜ₋₁)

where T = temperature:
- T → 0: approaches greedy (peaked distribution)
- T = 1: standard sampling
- T > 1: more random (flat distribution)
```

**Effect of temperature:**
```
Original: [0.7, 0.2, 0.1]

T = 0.5: [0.89, 0.08, 0.03]  (more peaked)
T = 1.0: [0.7, 0.2, 0.1]     (unchanged)
T = 2.0: [0.55, 0.28, 0.17]  (flatter)
```

#### Top-k Sampling

```
1. Sort tokens by probability
2. Keep only top-k tokens
3. Renormalize and sample

k = 50 (typical value)

Example:
All tokens: [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, ...]
Top-3: [0.3, 0.25, 0.2] → [0.4, 0.33, 0.27]
```

#### Top-p (Nucleus) Sampling

```
1. Sort tokens by probability (descending)
2. Find minimum set with cumulative probability ≥ p
3. Renormalize and sample from this set

p = 0.9 (typical)

Example:
Sorted probs: [0.5, 0.2, 0.15, 0.08, 0.04, 0.02, 0.01]
Cumulative:   [0.5, 0.7, 0.85, 0.93, ...]
Select: [0.5, 0.2, 0.15, 0.08] (sum = 0.93 ≥ 0.9)
```

#### Best Practices

```
Creative writing:
- temperature = 0.8-1.0
- top_p = 0.9
- top_k = 50

Factual tasks:
- temperature = 0.3-0.5
- top_p = 0.95
- top_k = 10

Code generation:
- temperature = 0.2
- top_p = 0.95
```

### Beam Search

Search for high-probability sequences:

```
1. Keep top-B candidates (beams)
2. For each beam, expand with top-B tokens
3. Keep overall top-B sequences
4. Repeat until all beams end

B = beam width (4-10 typical)
```

**Scoring:**
```
Score = log P(x₁:ₜ) / t^α

where α = length penalty (0.6-0.8)
Prevents bias toward shorter sequences
```

### KV Cache Optimization

#### Problem

Each token requires full context attention:
```
Token 1: Attend to [1]
Token 2: Attend to [1, 2]
Token 3: Attend to [1, 2, 3]
...

Redundant computation of keys and values for previous tokens
```

#### Solution: KV Caching

```
Cache computed keys and values:

K_cached = [k₁, k₂, ..., kₜ₋₁]
V_cached = [v₁, v₂, ..., vₜ₋₁]

For new token t:
1. Compute only qₜ, kₜ, vₜ
2. K = [K_cached, kₜ]
3. V = [V_cached, vₜ]
4. Attention(qₜ, K, V)
```

**Memory requirement:**
```
KV_Cache = 2 × Layers × Heads × Head_Dim × Seq_Len × Batch × Bytes

Example (LLaMA 7B, seq=2048):
2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes ≈ 1 GB per sample
```

**Speed improvement:**
```
Without cache: O(n²) for n tokens
With cache: O(n) for n tokens

~10-20× faster generation
```

### Batched Inference

#### Static Batching

```
Process multiple sequences together:

Batch = [seq₁, seq₂, ..., seqᵦ]

Challenge: Different sequence lengths
Solution: Padding to max length
```

#### Dynamic Batching

```
Continuous batching:
1. Add new requests as they arrive
2. Remove completed sequences
3. Maintain full GPU utilization

Tools: vLLM, TensorRT-LLM, Text Generation Inference
```

### Quantization for Inference

#### Post-Training Quantization

**8-bit (INT8):**
```
Quantization:
x_q = round(x / scale) + zero_point

Dequantization:
x ≈ (x_q - zero_point) × scale

Memory: 4× reduction vs FP32
Speed: 2-3× faster
Quality: <1% accuracy loss
```

**4-bit (INT4/NF4):**
```
More aggressive compression:
Memory: 8× reduction vs FP32
Speed: 3-4× faster
Quality: 1-3% accuracy loss

Used in GPTQ, AWQ, NF4 methods
```

#### GPTQ (Gradient-based PTQ)

```
Minimize quantization error layer-by-layer:

min ||WX - W_qX||²

Using Hessian information for optimal rounding
```

### Inference Metrics

#### Throughput

```
Tokens_Per_Second = Total_Tokens / Time

Batch throughput: 100-1000+ tokens/second
Single sequence: 10-50 tokens/second
```

#### Latency

```
Time_to_First_Token (TTFT) = Prefill_Time
Time_per_Output_Token (TPOT) = Decode_Time

Total_Time = TTFT + (Num_Tokens - 1) × TPOT
```

#### Memory Usage

```
Total_Memory = Model_Weights + KV_Cache + Activations

Example (LLaMA 7B, FP16):
Weights: 13 GB
KV_Cache (2K ctx): 1 GB
Activations: 2 GB
Total: ~16 GB (fits on single GPU)
```

---

## Fine-tuning Methods

### Full Fine-tuning

Update all model parameters:

```
For each example (x, y):
    loss = -log P(y|x; θ)
    θ ← θ - α∇loss

Pros: Best performance
Cons: Requires full model copy, expensive
```

### Supervised Fine-Tuning (SFT)

```
Dataset: {(prompt₁, completion₁), ..., (promptₙ, completionₙ)}

Loss:
L_SFT = -Σᵢ log P(completionᵢ | promptᵢ)

Typical data size: 10K - 100K examples
Training time: Hours to days
```

### Instruction Tuning

```
Format:
Instruction: "Translate to French: Hello"
Output: "Bonjour"

Loss includes both instruction and output:
L = -log P(output | instruction)

Enables zero-shot task following
```

### Parameter-Efficient Fine-Tuning (PEFT)

#### 1. LoRA (Low-Rank Adaptation)

**Concept:** Add trainable low-rank matrices to frozen weights

```
Original: y = Wx
LoRA: y = Wx + BAx

where:
- W ∈ ℝ^(d×k): frozen pretrained weights
- B ∈ ℝ^(d×r): trainable
- A ∈ ℝ^(r×k): trainable
- r << min(d, k): rank (4-64 typical)
```

**Parameter reduction:**
```
Original: d × k parameters
LoRA: d×r + r×k = r(d+k) parameters

Reduction ratio = r(d+k) / (d×k)

Example (d=k=4096, r=8):
Original: 16.8M
LoRA: 65.5K
Ratio: 0.4% of original
```

**Training:**
```
1. Freeze W
2. Initialize A ~ N(0, σ²), B = 0
3. Train only A and B
4. Merge at inference: W' = W + BA
```

**Multiple LoRA adapters:**
```
Switch tasks by swapping (A, B) pairs:
- Base model: 7B parameters (shared)
- Adapter 1: 8M parameters (task 1)
- Adapter 2: 8M parameters (task 2)
- ...
```

#### 2. Prefix Tuning

```
Add trainable prefix vectors to each layer:

Original: Attention(Q, K, V)
Prefix: Attention(Q, [P_k; K], [P_v; V])

where:
- P_k, P_v: trainable prefix for keys/values
- Prefix length: 10-100 tokens

Parameters: prefix_len × layers × d_model
```

#### 3. Adapter Layers

```
Insert small bottleneck layers:

x_out = x + Adapter(x)

Adapter(x) = W_up(ReLU(W_down(x)))

where:
- W_down ∈ ℝ^(d×r): project down
- W_up ∈ ℝ^(r×d): project up
- r = bottleneck dimension (64-256)

Parameters per layer: 2×d×r
```

#### 4. QLoRA (Quantized LoRA)

```
1. Quantize base model to 4-bit (NF4)
2. Add LoRA adapters in higher precision
3. Train adapters while keeping base frozen

Memory savings:
Base: 4-bit (8× reduction)
LoRA: FP16 (small, <1% of total)

Can fine-tune 65B model on single 48GB GPU
```

### Reinforcement Learning from Human Feedback (RLHF)

#### Stage 1: Reward Model Training

```
Collect preferences:
{(prompt, response_win, response_lose)}

Reward model:
r_θ(x, y) → scalar reward

Loss (Bradley-Terry):
L_RM = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

where σ = sigmoid function
```

#### Stage 2: Policy Optimization

**PPO (Proximal Policy Optimization):**

```
Objective:
L_PPO = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A) - β·KL(π_θ||π_ref)]

where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s): probability ratio
- A: advantage (reward - baseline)
- ε = 0.2: clip parameter
- β = 0.01: KL penalty coefficient
- π_ref: reference (original) policy
```

**Advantage computation:**
```
A(s, a) = Q(s, a) - V(s)
        = r + γV(s') - V(s)

where:
- Q(s, a): action-value function
- V(s): state-value function
- γ = 0.99: discount factor
```

#### Stage 3: Iterative Refinement

```
For iteration = 1 to N:
    1. Generate responses with current policy
    2. Collect human preferences
    3. Update reward model
    4. Update policy with PPO
    
Typical iterations: 3-5
```

### Direct Preference Optimization (DPO)

Simpler alternative to RLHF:

```
Skip reward model, optimize policy directly:

L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) 
                  - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]

Advantages:
- No reward model needed
- More stable training
- Simpler implementation
```

---

## Optimization Techniques

### Model Compression

#### 1. Pruning

**Magnitude-based pruning:**
```
Remove weights below threshold:

W_pruned = W ⊙ M

where M_ij = 1 if |W_ij| > t, else 0

Sparsity = Number_of_zeros / Total_weights
```

**Structured pruning:**
```
Remove entire:
- Attention heads
- FFN neurons
- Layers

Easier to accelerate on hardware
```

#### 2. Knowledge Distillation

```
Student learns from teacher:

L_distill = α·L_CE(y_true) + (1-α)·L_KL(P_teacher, P_student)

where:
- L_CE: standard cross-entropy
- L_KL: KL divergence between distributions
- α: balance parameter (0.5 typical)

Temperature scaling:
P_soft = softmax(logits / T)

Higher T → softer distributions → better distillation
```

**Size reduction examples:**
```
GPT-3 (175B) → DistilGPT (1.5B)
BERT-base (110M) → DistilBERT (66M)

Typically: 2-10× smaller, 97-99% performance
```

#### 3. Mixture of Experts (MoE)

```
Instead of dense FFN:
FFN(x) = Σᵢ G(x)ᵢ · Expertᵢ(x)

where:
- G(x): gating function (which experts to use)
- Top-k selection: use only k experts per token

Parameters: N × expert_size
Active per token: k × expert_size

Scaling: 10× parameters, 2× compute
```

### Training Optimizations

#### 1. Gradient Checkpointing

```
Trade computation for memory:

Forward pass:
- Save only subset of activations (checkpoints)
- Discard intermediate activations

Backward pass:
- Recompute activations from checkpoints

Memory reduction: 3-5×
Compute increase: 20-30%
```

#### 2. Flash Attention

```
Optimize attention computation:

Standard: O(n²) memory for attention matrix
Flash: Compute attention in blocks, keep in SRAM

Memory: O(n) instead of O(n²)
Speed: 2-4× faster
Exact (not approximate)
```

#### 3. ZeRO (Zero Redundancy Optimizer)

```
Distribute across GPUs:

Stage 1: Partition optimizer states
Stage 2: + Partition gradients
Stage 3: + Partition parameters

Memory per GPU = Total_Memory / Num_GPUs

Enable training 100B+ models on consumer GPUs
```

### Parallel Training

#### 1. Data Parallelism

```
Each GPU has full model:
- Different data batches
- Synchronize gradients

Effective batch = Single_batch × Num_GPUs

Scaling: Linear up to ~8-16 GPUs
```

#### 2. Model Parallelism (Tensor Parallelism)

```
Split layers across GPUs:

Layer weights: W = [W₁ | W₂ | ... | Wₙ]
Each GPU computes: yᵢ = Wᵢx

Combine outputs: y = [y₁; y₂; ...; yₙ]

Used for very large layers
```

#### 3. Pipeline Parallelism

```
Split model into stages:

GPU 1: Layers 1-20
GPU 2: Layers 21-40
GPU 3: Layers 41-60
GPU 4: Layers 61-80

Micro-batching to keep all GPUs busy
```

#### 4. 3D Parallelism

```
Combine all three:
- Data parallel: Across nodes
- Tensor parallel: Within node
- Pipeline parallel: Across layers

Example (GPT-3):
- 1024 GPUs total
- 8-way tensor parallel
- 8-way pipeline parallel
- 16-way data parallel
```

---

## Evaluation Metrics

### Perplexity

```
PPL = exp(-1/N Σᵢ log P(xᵢ|x₁:ᵢ₋₁))

Lower = better language modeling

Typical values:
- Good model: PPL < 20
- SOTA: PPL 5-15
```

### Benchmark Performance

#### 1. GLUE/SuperGLUE

Classification, QA, inference tasks:
```
Metrics: Accuracy, F1, Matthews correlation

Example scores (0-100):
- Human performance: ~90
- SOTA models: 85-95
```

#### 2. MMLU (Massive Multitask Language Understanding)

```
57 subjects across STEM, humanities, social sciences

Accuracy across subjects

GPT-4: ~86%
Claude 3 Opus: ~87%
Random: ~25%
```

#### 3. HumanEval (Code)

```
Metric: Pass@k

Pass@k = 1 - C(n-c, k) / C(n, k)

where:
- n: total samples generated
- c: correct samples
- k: attempts allowed

GPT-4: Pass@1 ≈ 67%
Claude 3.5 Sonnet: Pass@1 ≈ 92%
```

#### 4. GSM8K (Math Reasoning)

```
Grade school math problems

Metric: Exact match accuracy

o1-preview: ~90%
GPT-4: ~92%
```

### Human Evaluation

#### 1. Preference Ranking

```
Show outputs from models A and B
Human rates: A > B, B > A, or Tie

Win rate = Wins / (Wins + Losses)

Significant if: |Win_rate - 50%| > margin_of_error
```

#### 2. Likert Scale

```
Rate quality 1-5:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Aggregate: Mean, median, distribution
```

#### 3. Task Success Rate

```
Success rate = Completed_successfully / Total_attempts

For specific tasks:
- Factual accuracy
- Instruction following
- Code correctness
```

### Safety Metrics

#### 1. Toxicity

```
Toxicity_score = P(toxic | text)

Using classifiers like Perspective API

Target: <1% toxic responses
```

#### 2. Bias

```
Measure stereotyping, fairness across groups

Metrics:
- Demographic parity
- Equalized odds
- Individual fairness
```

#### 3. Hallucination Rate

```
Hallucination_rate = False_statements / Total_statements

Measure via:
- Fact-checking
- Consistency checks
- Attribution to sources
```

---

## Advanced Topics

### Context Length Extension

#### 1. Positional Interpolation

```
Scale positions to fit longer sequences:

pos' = pos × (original_length / new_length)

Apply RoPE with scaled positions

Can extend 2× with minimal fine-tuning
```

#### 2. YaRN (Yet another RoPE extensioN)

```
θ'ᵢ = θᵢ × s(i) × scale^λᵢ

where:
- s(i): interpolation function
- scale: extension factor
- λᵢ: dimension-specific weight

Extends 16×-32× context efficiently
```

#### 3. Sparse Attention

```
Instead of full O(n²) attention:

Local attention: Attend to nearby tokens
Global attention: Attend to special tokens
Dilated attention: Attend to every k-th token

Reduces to O(n√n) or O(n log n)
```

### Multimodal LLMs

#### Vision-Language Models

```
Architecture:
Image → Vision Encoder → Projector → LLM

Vision Encoder: CLIP, ViT
Projector: Linear layer or small MLP
LLM: Standard language model

Training:
1. Freeze vision encoder
2. Train projector on image-text pairs
3. Optionally fine-tune full model
```

**Cross-attention:**
```
Visual tokens as keys/values:
K_v = Vision_Encoder(image)
V_v = Vision_Encoder(image)

Text queries attend to visual tokens:
Attention(Q_text, K_v, V_v)
```

### Retrieval-Augmented Generation (RAG)

```
1. Retrieve relevant documents:
   docs = Retrieve(query, database)

2. Concatenate with query:
   context = [docs; query]

3. Generate with context:
   answer = LLM(context)
```

**Retrieval scoring:**
```
Similarity(query, doc) = cosine(Embed(query), Embed(doc))

Top-k documents by similarity
```

**Benefits:**
- Access external knowledge
- Reduce hallucinations
- Update without retraining

### Chain-of-Thought Reasoning

#### Zero-shot CoT

```
Prompt: "Question: [Q]. Let's think step by step."

Forces model to reason explicitly
```

#### Few-shot CoT

```
Examples with reasoning steps:

Q: Roger has 5 balls. He buys 2 cans of 3 balls. How many balls?
A: Roger started with 5 balls. 2 cans × 3 balls/can = 6 balls. 
   5 + 6 = 11 balls.

Q: [New question]
A: [Step-by-step reasoning]
```

### Tool Use and Function Calling

```
1. Model detects need for tool
2. Generates function call:
   {
     "name": "search",
     "arguments": {"query": "current weather"}
   }

3. Execute function → get result
4. Feed result back to model
5. Model generates final response

Enables:
- Web search
- Calculator
- Database queries
- API calls
```

---

## Practical Implementation

### Example: Training a Small LLM

#### 1. Model Configuration

```python
config = {
    'vocab_size': 50257,
    'd_model': 768,
    'n_layers': 12,
    'n_heads': 12,
    'head_dim': 64,  # d_model / n_heads
    'd_ff': 3072,    # 4 * d_model
    'max_seq_len': 1024,
    'dropout': 0.1
}

Total parameters: ~117M (GPT-2 Small size)
```

#### 2. Training Configuration

```python
training_config = {
    'batch_size': 32,
    'gradient_accumulation': 4,  # effective batch = 128
    'learning_rate': 6e-4,
    'warmup_steps': 2000,
    'max_steps': 100000,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'eval_interval': 1000,
}
```

#### 3. Memory Estimation

```
Parameters: 117M × 2 bytes (FP16) = 234 MB
Optimizer states (Adam): 117M × 8 bytes = 936 MB
Gradients: 117M × 2 bytes = 234 MB
Activations (batch=32, seq=1024): ~4 GB

Total: ~5.5 GB → fits on single consumer GPU
```

#### 4. Training Time

```
Tokens per step: 32 batch × 1024 seq = 32,768
Total steps: 100,000
Total tokens: 3.28 billion

At 5 seconds/step:
100,000 steps × 5 sec = 500,000 sec ≈ 6 days
```

### Example: Inference Optimization

```python
# Optimization checklist
optimizations = {
    '1. KV caching': '10-20× faster',
    '2. Batch processing': '2-5× throughput',
    '3. Quantization (INT8)': '2× faster, 4× less memory',
    '4. Flash Attention': '2-3× faster attention',
    '5. Compilation (torch.compile)': '1.5-2× faster',
}

# Example configuration
inference_config = {
    'batch_size': 8,
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,
    'use_cache': True,  # KV caching
}
```

### Cost Analysis

#### Training Costs

```
GPU rental: $2/hour per A100

Small model (1B params):
- GPUs: 8
- Time: 1 week
- Cost: 8 × 24 × 7 × $2 = $2,688

Medium model (7B params):
- GPUs: 64
- Time: 2 weeks
- Cost: 64 × 24 × 14 × $2 = $43,008

Large model (70B params):
- GPUs: 256
- Time: 1 month
- Cost: 256 × 24 × 30 × $2 = $368,640
```

#### Inference Costs

```
Cost per 1M tokens (GPT-4 pricing):
Input: $10
Output: $30

1000 users × 100 queries/day × 1000 tokens = 100M tokens/day
Daily cost: $1,000 - $3,000
```

---

## Summary

### Key Takeaways

1. **Architecture**: Transformers with self-attention are the foundation
2. **Training**: Next-token prediction on massive text corpora
3. **Scaling**: Larger models + more data = better performance
4. **Efficiency**: MoE, quantization, PEFT enable practical deployment
5. **Alignment**: RLHF/DPO make models helpful and safe

### Mathematical Core

```
Embedding: x ∈ ℝᵈ
Attention: softmax(QKᵀ/√d)V
FFN: GELU(xW₁)W₂
Loss: -Σ log P(xₜ|x₁:ₜ₋₁)
Optimization: Adam with lr scheduling
```

### Critical Numbers

| Aspect | Typical Range |
|--------|---------------|
| Parameters | 1B - 1T |
| Training tokens | 100B - 15T |
| Context length | 2K - 1M |
| Training time | Weeks - Months |
| Cost | $10K - $100M+ |
| Inference speed | 10-100 tokens/sec |

### Future Directions

- Longer context windows (1M+)
- Multimodal capabilities
- Better reasoning and planning
- More efficient architectures
- Improved alignment
- Reduced hallucinations

---

## Glossary

**Autoregressive**: Generating one token at a time, conditioned on previous tokens

**Attention**: Mechanism to weigh importance of different input positions

**Embedding**: Dense vector representation of discrete tokens

**Fine-tuning**: Adapting pretrained model to specific tasks

**Hallucination**: Generating plausible but incorrect information

**Perplexity**: Measure of how well model predicts text (lower = better)

**Quantization**: Reducing numerical precision to save memory/compute

**Temperature**: Controls randomness in sampling (lower = more focused)

**Token**: Basic unit of text (word, subword, or character)

**Transformer**: Neural architecture using self-attention

---

## References

### Foundational Papers

1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
3. "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
4. "Training language models to follow instructions with human feedback" (InstructGPT, Ouyang et al., 2022)
5. "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)

### Key Concepts

1. Scaling Laws (Kaplan et al., 2020)
2. Chain-of-Thought Prompting (Wei et al., 2022)
3. LoRA (Hu et al., 2021)
4. Flash Attention (Dao et al., 2022)
5. RoPE (Su et al., 2021)

---

*Last Updated: February 2026*
*Document Version: 1.0*
*Author: Comprehensive LLM Guide*
