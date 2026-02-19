# Multi-Token Prediction: Complete Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem with Standard Language Modeling](#the-problem-with-standard-language-modeling)
3. [Multi-Token Prediction Concept](#multi-token-prediction-concept)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Architecture Design](#architecture-design)
6. [Training Methodology](#training-methodology)
7. [DeepSeek's Implementation](#deepseeks-implementation)
8. [Meta's Approach](#metas-approach)
9. [Benefits and Trade-offs](#benefits-and-trade-offs)
10. [Detailed Examples](#detailed-examples)
11. [Inference Strategies](#inference-strategies)
12. [Advanced Topics](#advanced-topics)

---

## Introduction

### What is Multi-Token Prediction?

**Multi-Token Prediction (MTP)** is a training technique where the model simultaneously predicts multiple future tokens instead of just the next token.

**Core Idea:**
> Train the model to predict not just x_{t+1}, but also x_{t+2}, x_{t+3}, ..., x_{t+n} at each step.

### Historical Context

```
Timeline:
2017: Standard Autoregressive LM (GPT-1)
      - Predict next token only
      - One prediction head

2023: Multi-Query Decoding (Speculative Decoding)
      - Generate multiple tokens at once
      - Inference-only technique

2024: Multi-Token Prediction Training
      - Meta AI: "Better & Faster LLMs via Multi-Token Prediction"
      - DeepSeek-V3: Integrated into training
      - Prediction heads for multiple future tokens
      - Training innovation

Key Innovation: Multiple prediction heads trained simultaneously
```

### Why Multi-Token Prediction?

```
Standard Training Problem:
  At each position, predict only 1 token
  → Model learns short-term dependencies well
  → Longer-term planning is implicit
  → Sample inefficiency

Multi-Token Prediction Solution:
  At each position, predict n tokens ahead
  → Explicit long-range prediction
  → Better planning capability
  → More training signal per sample
  → Improved sample efficiency
```

---

## The Problem with Standard Language Modeling

### Standard Next-Token Prediction

```
Standard Autoregressive Training:

Input sequence: [x₁, x₂, x₃, ..., xₜ]

Training:
  Position 1: Predict x₂ given [x₁]
  Position 2: Predict x₃ given [x₁, x₂]
  Position 3: Predict x₄ given [x₁, x₂, x₃]
  ...
  Position t: Predict x_{t+1} given [x₁, ..., xₜ]

Loss:
  L = -Σₜ log P(xₜ | x₁, ..., x_{t-1})

Issues:
  1. Only one prediction per position
  2. No explicit long-range planning
  3. Limited training signal
  4. Myopic optimization
```

### Myopic Prediction Problem

```
Example: Writing a story

Standard Training:
  "Once upon a" → predict "time"
  
  Model learns local patterns well:
  ✓ Grammar
  ✓ Common phrases
  ✓ Next word probability
  
  But struggles with:
  ✗ Story arc planning
  ✗ Character consistency
  ✗ Plot development
  ✗ Long-range coherence

Why?
  Only trained to predict immediate next token
  No explicit signal for future planning
```

### Sample Efficiency Issues

```
Training data usage:

Standard method:
  Each token used once as prediction target
  Sequence length n → n training signals
  
Example sequence: "The cat sat on the mat"
  Signals:
  - Position 1: predict "cat"
  - Position 2: predict "sat"
  - Position 3: predict "on"
  - Total: 6 signals from 7-token sequence

Inefficient!
  Model sees each token context only once
  No explicit future prediction practice
```

---

## Multi-Token Prediction Concept

### Basic Concept

```
Multi-Token Prediction (n=3):

Input sequence: [x₁, x₂, x₃, ..., xₜ]

Training at position t:
  Predict x_{t+1} given [x₁, ..., xₜ]  ← 1 step ahead
  Predict x_{t+2} given [x₁, ..., xₜ]  ← 2 steps ahead
  Predict x_{t+3} given [x₁, ..., xₜ]  ← 3 steps ahead

Loss:
  L = L₁ + α·L₂ + α²·L₃
  
  where:
    L₁ = -log P(x_{t+1} | x₁, ..., xₜ)
    L₂ = -log P(x_{t+2} | x₁, ..., xₜ)
    L₃ = -log P(x_{t+3} | x₁, ..., xₜ)
    α = decay factor (typically 0.3)

Key insight:
  Multiple predictions from same context
  → More training signal per position
  → Explicit future modeling
```

### Visual Representation

```
Standard Prediction:
┌─────────────────────────────────────┐
│  Context: [x₁, x₂, x₃]             │
│     ↓                               │
│  Transformer                        │
│     ↓                               │
│  Hidden state h₃                    │
│     ↓                               │
│  Single prediction head             │
│     ↓                               │
│  Predict: x₄                        │
└─────────────────────────────────────┘

Multi-Token Prediction (n=3):
┌─────────────────────────────────────┐
│  Context: [x₁, x₂, x₃]             │
│     ↓                               │
│  Transformer (shared)               │
│     ↓                               │
│  Hidden state h₃                    │
│     ↓                               │
│     ├──────────┬──────────┐         │
│     ↓          ↓          ↓         │
│  Head 1     Head 2     Head 3       │
│     ↓          ↓          ↓         │
│  Predict:  Predict:  Predict:       │
│    x₄         x₅         x₆         │
│  (next)    (+2 ahead)  (+3 ahead)   │
└─────────────────────────────────────┘
```

### Key Components

```
1. Multiple Prediction Heads:
   - Separate head for each future position
   - Each head specialized for its prediction distance
   - Share same transformer backbone

2. Intermediate States (optional):
   - Use ground-truth tokens to update hidden states
   - Creates "what-if" scenarios
   - Improves multi-step predictions

3. Loss Weighting:
   - Decay factor for distant predictions
   - Balances near-term vs long-term learning
   - Prevents overwhelming from multiple losses
```

---

## Mathematical Foundation

### Standard Autoregressive Model

```
Probability factorization:

P(x₁, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., x_{i-1})

Training objective:
  L = -Σᵢ log P(xᵢ | x₁, ..., x_{i-1})

Model:
  P(xᵢ | x₁, ..., x_{i-1}) = softmax(W h_{i-1})
  
  where h_{i-1} = Transformer(x₁, ..., x_{i-1})
```

### Multi-Token Prediction Formulation

**Version 1: Independent Predictions**

```
At each position t, predict n future tokens independently:

P(x_{t+1} | x_{1:t})
P(x_{t+2} | x_{1:t})  ← Independent of x_{t+1}
P(x_{t+3} | x_{1:t})  ← Independent of x_{t+1}, x_{t+2}

Loss:
  L_t = -log P(x_{t+1} | x_{1:t})
      - α·log P(x_{t+2} | x_{1:t})
      - α²·log P(x_{t+3} | x_{1:t})

Total loss:
  L = Σₜ L_t

Problem: Assumes independence (not realistic)
```

**Version 2: Chained Predictions (Better)**

```
Use ground-truth intermediate tokens:

P(x_{t+1} | x_{1:t})
P(x_{t+2} | x_{1:t}, x_{t+1})  ← Conditioned on ground truth x_{t+1}
P(x_{t+3} | x_{1:t}, x_{t+1}, x_{t+2})  ← Conditioned on ground truth

Implementation:
  h_{t} = Transformer(x_{1:t})
  
  # First prediction
  P(x_{t+1} | x_{1:t}) = softmax(W₁ h_t)
  L₁ = -log P(x_{t+1} | x_{1:t})
  
  # Second prediction (update hidden state with ground truth)
  h_{t+1} = Update(h_t, x_{t+1})  ← Use actual x_{t+1}
  P(x_{t+2} | x_{1:t+1}) = softmax(W₂ h_{t+1})
  L₂ = -log P(x_{t+2} | x_{1:t+1})
  
  # Third prediction
  h_{t+2} = Update(h_{t+1}, x_{t+2})  ← Use actual x_{t+2}
  P(x_{t+3} | x_{1:t+2}) = softmax(W₃ h_{t+2})
  L₃ = -log P(x_{t+3} | x_{1:t+2})

Combined loss:
  L_t = L₁ + α·L₂ + α²·L₃

Advantage: Proper conditional dependencies
```

**Version 3: Shared Predictions (Memory Efficient)**

```
Share embedding layer, use different output heads:

Architecture:
  Shared Transformer: x_{1:t} → h_t
  
  Head 1: h_t → P(x_{t+1})
  Head 2: h_t → P(x_{t+2})  ← Share h_t, different projection
  Head 3: h_t → P(x_{t+3})

Loss:
  L = Σₜ [log P(x_{t+1} | h_t) 
        + α·log P(x_{t+2} | h_t)
        + α²·log P(x_{t+3} | h_t)]

Trade-off:
  + Memory efficient (single forward pass)
  + Fast training
  - Less accurate for distant predictions
  - No intermediate ground-truth conditioning
```

### Loss Weight Decay

```
Why use decay factor α < 1?

Without decay (α = 1):
  L = L₁ + L₂ + L₃
  Total loss dominated by sum
  Equal weight to all predictions

With decay (α = 0.3):
  L = L₁ + 0.3·L₂ + 0.09·L₃
  
  Relative importance:
  - Next token: 100%
  - 2 ahead: 30%
  - 3 ahead: 9%
  
  Reasoning:
  1. Near-term is more important for immediate quality
  2. Far predictions have more uncertainty
  3. Prevents overwhelming from multiple losses
  4. Maintains training stability

Typical values: α ∈ [0.3, 0.5]
```

---

## Architecture Design

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│         Multi-Token Prediction Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: x₁, x₂, ..., xₜ                                     │
│    ↓                                                         │
│  ┌────────────────────────────┐                             │
│  │  Token Embeddings          │                             │
│  │  E(x₁), E(x₂), ..., E(xₜ)  │                             │
│  └──────────┬─────────────────┘                             │
│             ↓                                                │
│  ┌────────────────────────────┐                             │
│  │  Positional Encoding       │                             │
│  │  PE₁, PE₂, ..., PEₜ        │                             │
│  └──────────┬─────────────────┘                             │
│             ↓                                                │
│  ┌────────────────────────────┐                             │
│  │  Transformer Layers (L)     │                             │
│  │  (Shared backbone)          │                             │
│  └──────────┬─────────────────┘                             │
│             ↓                                                │
│         h₁, h₂, ..., hₜ                                     │
│             │                                                │
│             ↓ (at position t)                                │
│            hₜ                                                │
│             │                                                │
│   ┌─────────┼─────────┬──────────┐                         │
│   │         │         │          │                          │
│   ↓         ↓         ↓          ↓                          │
│ ┌────┐   ┌────┐   ┌────┐    ┌────┐                        │
│ │ W₁ │   │ W₂ │   │ W₃ │    │ W₄ │  ← Prediction heads    │
│ └─┬──┘   └─┬──┘   └─┬──┘    └─┬──┘                        │
│   │        │        │         │                             │
│   ↓        ↓        ↓         ↓                             │
│ P(x_{t+1}) P(x_{t+2}) P(x_{t+3}) P(x_{t+4})                │
│   │        │        │         │                             │
│   ↓        ↓        ↓         ↓                             │
│  L₁       L₂       L₃        L₄                             │
│   │        │        │         │                             │
│   └────────┴────────┴─────────┘                             │
│             ↓                                                │
│   L = L₁ + α·L₂ + α²·L₃ + α³·L₄                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Key:
- Single forward pass through transformer
- Multiple prediction heads (1 per future position)
- Weighted combination of losses
```

### Prediction Head Design

```
Option 1: Simple Linear Projection
┌─────────────────────────┐
│ Hidden state: hₜ        │
│     ↓                   │
│ Linear: W_i hₜ + b_i    │
│     ↓                   │
│ Logits: ∈ ℝ^vocab_size  │
│     ↓                   │
│ Softmax                 │
│     ↓                   │
│ P(x_{t+i})              │
└─────────────────────────┘

Parameters per head: vocab_size × d_model

Option 2: Small MLP Head
┌─────────────────────────┐
│ Hidden state: hₜ        │
│     ↓                   │
│ Linear: d_model → d_mlp │
│     ↓                   │
│ ReLU / GELU             │
│     ↓                   │
│ Linear: d_mlp → vocab   │
│     ↓                   │
│ Softmax                 │
└─────────────────────────┘

More parameters, potentially better predictions

Option 3: Shared Embedding (Memory Efficient)
┌─────────────────────────┐
│ Hidden state: hₜ        │
│     ↓                   │
│ Linear: d_model → d_emb │
│     ↓                   │
│ Dot product with        │
│ embedding matrix        │
│     ↓                   │
│ Logits                  │
└─────────────────────────┘

Ties weights with embedding layer
```

### Intermediate State Updates

```
For chained predictions (more accurate):

Position t:
┌─────────────────────────────────┐
│ h_t = Transformer(x_{1:t})      │
│   ↓                             │
│ Predict x_{t+1}                 │
│   ↓                             │
│ Update: h_{t+1} = f(h_t, x_{t+1})│  ← Use ground truth
│   ↓                             │
│ Predict x_{t+2}                 │
│   ↓                             │
│ Update: h_{t+2} = f(h_{t+1}, x_{t+2})│
│   ↓                             │
│ Predict x_{t+3}                 │
└─────────────────────────────────┘

Update function f():

Option 1: Lightweight Transformer
  h_{t+1} = h_t + MiniTransformer(E(x_{t+1}))

Option 2: Residual Connection
  h_{t+1} = h_t + W·E(x_{t+1})

Option 3: GRU-style Update
  h_{t+1} = GRU(h_t, E(x_{t+1}))

Trade-off:
  + More accurate predictions
  - More computation
  - More memory
```

---

## Training Methodology

### Training Algorithm

```python
def train_multi_token_prediction(model, data, n_ahead=3, alpha=0.3):
    """
    Train with multi-token prediction.
    
    Args:
        model: Language model with n prediction heads
        data: Training sequences
        n_ahead: Number of tokens to predict ahead
        alpha: Decay factor for loss weighting
    """
    for batch in data:
        # batch: [batch_size, seq_len]
        
        # Forward pass through transformer
        hidden_states = model.transformer(batch)
        # hidden_states: [batch_size, seq_len, d_model]
        
        total_loss = 0
        
        # For each position
        for t in range(seq_len - n_ahead):
            h_t = hidden_states[:, t, :]
            
            # Predict multiple future tokens
            for i in range(1, n_ahead + 1):
                # Get target token
                target = batch[:, t + i]
                
                # Predict using head i
                logits = model.prediction_heads[i-1](h_t)
                
                # Compute loss
                loss_i = cross_entropy(logits, target)
                
                # Add weighted loss
                weight = alpha ** (i - 1)
                total_loss += weight * loss_i
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
```

### Curriculum Learning

```
Progressive training strategy:

Stage 1 (Early Training):
  n_ahead = 1
  Standard next-token prediction
  Build basic language understanding

Stage 2 (Mid Training):
  n_ahead = 2
  Introduce 1-step-ahead prediction
  α = 0.5 (balanced)

Stage 3 (Late Training):
  n_ahead = 3-4
  Full multi-token prediction
  α = 0.3 (emphasize near-term)

Rationale:
  - Easier to harder
  - Stable initial training
  - Gradual complexity increase
```

### Hyperparameter Tuning

```
Key hyperparameters:

1. Number of future predictions (n_ahead):
   Values: 2-8
   Typical: 3-4
   Trade-off: More predictions = more signal but slower training

2. Decay factor (α):
   Values: 0.2-0.5
   Typical: 0.3
   Lower = emphasize near-term, higher = balanced

3. Loss weighting scheme:
   - Exponential: α^i (most common)
   - Linear: (n_ahead - i + 1) / sum
   - Custom: task-specific weights

4. Prediction head architecture:
   - Simple: Linear projection
   - Complex: 2-layer MLP
   - Shared: Tied with embeddings

5. Update mechanism:
   - None: Independent predictions (fast)
   - Simple: Linear update (balanced)
   - Full: Mini-transformer (accurate, slow)
```

---

## DeepSeek's Implementation

### DeepSeek-V3 Configuration

```
DeepSeek-V3 Multi-Token Prediction:
├── Number of predictions: 3
├── Decay factor: α = 0.3
├── Prediction heads: 3 separate linear layers
├── Update mechanism: Simple residual update
└── Training: Full sequence, all positions

Architecture:
  Input: x_{1:t}
    ↓
  Transformer (671B params, 37B active)
    ↓
  Hidden state: h_t
    ├───────────┬───────────┐
    ↓           ↓           ↓
  Head 1      Head 2      Head 3
    ↓           ↓           ↓
  x_{t+1}     x_{t+2}     x_{t+3}
    ↓           ↓           ↓
   L₁        0.3·L₂      0.09·L₃

Total loss: L = L₁ + 0.3·L₂ + 0.09·L₃
```

### DeepSeek Implementation Details

```python
class DeepSeekMultiTokenPredictor:
    """
    DeepSeek-V3 multi-token prediction.
    """
    def __init__(
        self,
        d_model: int = 7168,
        vocab_size: int = 128256,
        n_ahead: int = 3,
        alpha: float = 0.3
    ):
        # Main transformer (shared)
        self.transformer = DeepSeekV3Transformer(...)
        
        # Prediction heads (one per future position)
        self.prediction_heads = [
            nn.Linear(d_model, vocab_size)
            for _ in range(n_ahead)
        ]
        
        self.n_ahead = n_ahead
        self.alpha = alpha
    
    def forward(self, input_ids, labels):
        # Transformer forward
        hidden_states = self.transformer(input_ids)
        
        # Multi-token prediction losses
        losses = []
        
        for i in range(self.n_ahead):
            # Get predictions
            logits = self.prediction_heads[i](hidden_states)
            
            # Shift for i-step-ahead prediction
            shifted_labels = labels[:, i+1:]
            shifted_logits = logits[:, :-i-1, :]
            
            # Compute loss
            loss_i = F.cross_entropy(
                shifted_logits.reshape(-1, self.vocab_size),
                shifted_labels.reshape(-1),
                ignore_index=-100
            )
            
            # Weight by decay factor
            weighted_loss = (self.alpha ** i) * loss_i
            losses.append(weighted_loss)
        
        # Total loss
        total_loss = sum(losses)
        
        return total_loss
```

### Training Recipe

```
DeepSeek-V3 Training with MTP:

Data: 14.8T tokens

Configuration:
├── Batch size: 4.6M tokens per batch
├── Sequence length: 4096 tokens
├── Learning rate: Peak 4.2e-4, min 4.2e-5
├── Warmup: 2000 steps
├── Decay: Cosine
└── Multi-token: 3 predictions, α=0.3

Training time: ~2 months on thousands of GPUs

Benefits observed:
├── 15% faster convergence
├── Better long-context understanding
├── Improved code generation
└── More coherent outputs
```

---

## Meta's Approach

### Meta's Multi-Token Prediction Paper

```
"Better & Faster Large Language Models via Multi-Token Prediction"
(Meta AI Research, 2024)

Key contributions:
1. Systematic study of multi-token prediction
2. Comparison with standard training
3. Analysis of benefits across model sizes
4. Novel inference strategies

Configuration:
├── Models: 300M to 13B parameters
├── Predictions: 4 tokens ahead
├── Decay: α = 0.5
└── Architecture: Separate prediction heads
```

### Meta's Architecture

```
Meta MTP Architecture:

┌─────────────────────────────────────┐
│  Shared Transformer Backbone        │
│     (Standard decoder layers)        │
└──────────────┬──────────────────────┘
               ↓
        Hidden states
               │
     ┌─────────┼──────────┬──────────┐
     ↓         ↓          ↓          ↓
  Head 0    Head 1    Head 2    Head 3
  (next)   (+1)       (+2)       (+3)
     │         │          │          │
     ↓         ↓          ↓          ↓
  Logits    Logits    Logits    Logits
     │         │          │          │
     ↓         ↓          ↓          ↓
  Loss 0    Loss 1    Loss 2    Loss 3
   ×1.0      ×0.5      ×0.25     ×0.125

Combined: L = L₀ + 0.5·L₁ + 0.25·L₂ + 0.125·L₃
```

### Meta's Key Findings

```
Experimental Results:

1. Sample Efficiency:
   MTP (4 ahead) vs Standard:
   - 13B model: 15% fewer tokens to same perplexity
   - 3B model: 12% fewer tokens
   - 300M model: 8% fewer tokens

2. Downstream Tasks:
   Task                Standard    MTP-4    Improvement
   ────────────────────────────────────────────────────
   Code generation     35.2%       41.8%    +18.8%
   Math reasoning      42.1%       48.7%    +15.7%
   Long context QA     71.3%       76.9%    +7.9%
   
3. Model Size Scaling:
   Larger models benefit more from MTP
   13B: +18% vs 300M: +8%

4. Inference Speed:
   With speculative decoding: 2-3× faster
```

---

## Benefits and Trade-offs

### Advantages

```
1. Sample Efficiency:
   - More training signal per sample
   - Faster convergence (10-20%)
   - Better data utilization

2. Long-Range Planning:
   - Explicit future modeling
   - Better coherence
   - Improved reasoning

3. Code Generation:
   - Significant improvements (15-20%)
   - Better multi-line planning
   - Fewer logical errors

4. Few-Shot Learning:
   - Better in-context learning
   - More robust to prompts
   - Better instruction following

5. Speculative Decoding:
   - Natural fit for fast inference
   - Can use auxiliary heads
   - 2-3× speedup possible
```

### Disadvantages

```
1. Training Cost:
   - n× prediction heads
   - More computation per step
   - More memory for gradients
   Increase: ~30-50% training time

2. Implementation Complexity:
   - Multiple heads to manage
   - More hyperparameters
   - Harder to debug

3. Inference (without speculation):
   - Extra heads unused
   - Wasted parameters
   - Unless using speculative decoding

4. Hyperparameter Sensitivity:
   - α choice matters
   - n_ahead needs tuning
   - Head architecture decisions

5. Diminishing Returns:
   - Benefits plateau after 3-4 predictions
   - More predictions ≠ proportional gain
```

### When to Use Multi-Token Prediction

```
✓ Use MTP when:
  - Training large models (>1B params)
  - Code generation is important
  - Long-form generation needed
  - Sample efficiency critical
  - Have compute budget for training

✗ Skip MTP when:
  - Small models (<500M params)
  - Simple classification tasks
  - Very limited compute
  - Short-form generation only
  - Proven baseline needed first
```

---

## Detailed Examples

### Example 1: Code Generation

```
Without MTP (Standard Training):

Input: "def fibonacci(n):"
Model predicts next token: "
"
Then: "if"
Then: "n"
...

Problem:
  No explicit planning for function structure
  Myopic token-by-token generation

With MTP (n=3):

Input: "def fibonacci(n):"

Predictions:
  +1: "
"
  +2: "if"
  +3: "n"

Model learns:
  - Function needs indentation
  - Likely starts with conditional
  - Variable 'n' will be used

Result:
  Better function structure
  Fewer syntax errors
  More coherent logic flow
```

### Example 2: Story Writing

```
Without MTP:

"Once upon a time" → "there"
"there" → "was"
"was" → "a"

Focuses on immediate coherence
May lose plot thread

With MTP (n=3):

"Once upon a time" → ["there", "was", "a"]

Model simultaneously considers:
  - Immediate: "there"
  - Short-term: "was" 
  - Medium-term: "a"

Better planning for:
  - Character introduction
  - Setting establishment
  - Plot setup
```

### Example 3: Mathematical Reasoning

```
Problem: "What is 15 × 23?"

Without MTP:
"Let's" → "solve" → "this" → "step" → ...
May get stuck or make errors

With MTP:
Plans ahead:
  +1: "Let's"
  +2: "break"
  +3: "this"
  +4: "down"

Learns multi-step structure:
  Step 1: Setup
  Step 2: Decomposition
  Step 3: Calculation
  Step 4: Answer

Result: More reliable reasoning chains
```

---

## Inference Strategies

### Standard Inference (Wasteful)

```
Problem:
  Trained n prediction heads
  At inference: only use head 1
  Heads 2-n wasted

Solution options:
  1. Just accept waste (simple)
  2. Speculative decoding (fast)
  3. Ensemble predictions (quality)
```

### Speculative Decoding with MTP

```
Idea: Use auxiliary heads to speculate future tokens

Algorithm:
  1. Generate n candidates using all heads
  2. Verify in parallel
  3. Accept correct prefix
  4. Restart from rejection point

Example (n=3):

Step 1: Generate candidates
  Head 1 → token A (confidence: 0.9)
  Head 2 → token B (confidence: 0.7)
  Head 3 → token C (confidence: 0.5)
  
  Candidates: [A, B, C]

Step 2: Verify with full model
  Context + [A] → Actual next: A ✓
  Context + [A, B] → Actual next: B ✓
  Context + [A, B, C] → Actual next: D ✗

Step 3: Accept [A, B], restart from [A, B]

Speedup:
  Without: 3 sequential steps
  With: 2 tokens in ~1.5 steps
  Speedup: 2× (typical: 1.5-3×)
```

### Ensemble Predictions

```
Use multiple heads for better quality:

Voting:
  For each position:
    Vote_1 = argmax(Head_1)
    Vote_2 = argmax(Head_2)
    Vote_3 = argmax(Head_3)
  
  Final = majority_vote([Vote_1, Vote_2, Vote_3])

Averaging:
  Probs_combined = (Probs_1 + 0.5·Probs_2 + 0.25·Probs_3) / 1.75
  Final = argmax(Probs_combined)

Benefit:
  More robust predictions
  Reduced variance
  Better calibration
```

---

## Advanced Topics

### Multi-Token Prediction with RL

```
Combine MTP with reinforcement learning:

Reward Signal:
  R = reward for n-token sequence
  (e.g., code execution success)

Training:
  1. Generate n tokens using MTP heads
  2. Execute code, get reward
  3. Backprop reward to all n heads

Benefit:
  Multi-step planning with direct reward
  Better for complex tasks
```

### Adaptive n-ahead

```
Vary prediction depth based on context:

Easy contexts (common patterns):
  n_ahead = 2
  Less overhead needed

Hard contexts (complex reasoning):
  n_ahead = 4
  More planning beneficial

Dynamic adjustment:
  n_ahead = f(perplexity, task_type)
```

### Hierarchical Multi-Token Prediction

```
Predict at multiple granularities:

Level 1: Next token
Level 2: Next phrase (3-5 tokens)
Level 3: Next sentence (20-30 tokens)

Architecture:
  L1 heads: Predict tokens
  L2 heads: Predict phrase embeddings
  L3 heads: Predict sentence embeddings

Benefit:
  Multi-scale planning
  Better long-form generation
```

---

## Summary

### Key Concepts

```
1. Core Idea:
   - Predict multiple future tokens simultaneously
   - More training signal per position
   - Explicit long-range planning

2. Architecture:
   - Shared transformer backbone
   - Multiple prediction heads (1 per future position)
   - Weighted loss combination

3. Training:
   - Standard loss + weighted auxiliary losses
   - Decay factor α (typically 0.3)
   - 3-4 predictions typical

4. Benefits:
   - 10-20% faster convergence
   - Better code generation (+15-20%)
   - Improved reasoning
   - Enables speculative decoding
```

### Mathematical Summary

```
Standard: L = -log P(x_{t+1} | x_{1:t})

Multi-Token: L = Σᵢ αⁱ⁻¹ · (-log P(x_{t+i} | x_{1:t}))

where:
  i ∈ {1, 2, ..., n}
  α ∈ [0.3, 0.5] (decay factor)
  n ∈ [3, 4] (number of predictions)

Typical configuration:
  L = L₁ + 0.3·L₂ + 0.09·L₃
```

### Implementation Checklist

```
□ Multiple prediction heads (3-4)
□ Decay factor selection (α = 0.3)
□ Loss weighting implementation
□ Optional: Intermediate state updates
□ Optional: Curriculum learning
□ Testing: Speculative decoding
□ Monitoring: Per-head metrics
```

### Future Directions

```
1. Better head architectures
2. Dynamic n-ahead selection
3. Multi-scale prediction
4. Integration with RL
5. Improved speculative decoding
6. Task-specific weighting
```

