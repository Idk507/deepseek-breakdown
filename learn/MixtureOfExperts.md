# Mixture of Experts (MoE): Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Routing Mechanisms](#routing-mechanisms)
5. [Training Challenges and Solutions](#training-challenges-and-solutions)
6. [Architecture Variants](#architecture-variants)
7. [Sparse vs Dense MoE](#sparse-vs-dense-moe)
8. [Load Balancing](#load-balancing)
9. [Modern Implementations](#modern-implementations)
10. [Practical Considerations](#practical-considerations)

---

## Introduction

### What is Mixture of Experts?

**Mixture of Experts (MoE)** is a machine learning technique that uses multiple specialized sub-models (experts) and a gating mechanism (router) to determine which experts should process each input.

**Core Idea**: Instead of one large model processing everything, use many smaller specialized models, each becoming an expert at different aspects of the data.

### History and Motivation

**Origins**: Introduced by Jacobs et al. (1991) for supervised learning

**Modern Revival**: 
- Shazeer et al. (2017) - "Outrageously Large Neural Networks" (MoE in Transformers)
- Switch Transformers (2021) - Simplified sparse MoE
- GLaM (2021) - 1.2T parameter model
- GPT-4 (rumored to use MoE)

**Why MoE?**

1. **Conditional Computation**: Only activate relevant experts per input
2. **Model Capacity**: Increase parameters without increasing compute
3. **Specialization**: Different experts learn different patterns
4. **Scalability**: Efficiently scale to trillions of parameters

### Key Insight

**Traditional Model**:
- Parameters: $N$
- Computation: $O(N)$ per input

**MoE Model**:
- Total Parameters: $N \times E$ (E experts)
- Computation: $O(N \times k)$ per input (activate only $k$ experts)
- If $k \ll E$: Much more efficient!

---

## Core Concepts

### Basic Architecture

```
Input → Router (Gating Network) → Select Top-k Experts → Combine Outputs → Output
                                        ↓
                                   Expert 1
                                   Expert 2
                                   Expert 3
                                     ...
                                   Expert E
```

### Components

**1. Experts**: Specialized sub-networks (typically FFN layers)

**2. Router/Gating Network**: Decides which experts to activate

**3. Combination Function**: Merges outputs from selected experts

---

## Mathematical Formulation

### Basic MoE Layer

Given input $\mathbf{x} \in \mathbb{R}^d$:

**Step 1: Router Computes Expert Weights**

$$\mathbf{g}(\mathbf{x}) = \text{Softmax}(W_g \mathbf{x})$$

where:
- $W_g \in \mathbb{R}^{E \times d}$ is the router weight matrix
- $\mathbf{g}(\mathbf{x}) \in \mathbb{R}^E$ is the gating distribution
- $E$ is the number of experts

**Step 2: Each Expert Processes Input**

For expert $i$:

$$\mathbf{y}_i = E_i(\mathbf{x})$$

where $E_i$ is the $i$-th expert network (typically a feedforward network).

**Step 3: Weighted Combination**

$$\mathbf{y} = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot \mathbf{y}_i = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

### Detailed Mathematical Flow

**Router Logits**:

$$\mathbf{h} = W_g \mathbf{x} + \mathbf{b}_g$$

where $\mathbf{h} \in \mathbb{R}^E$ are the router logits.

**Gating Weights (Softmax)**:

$$g_i(\mathbf{x}) = \frac{e^{h_i}}{\sum_{j=1}^{E} e^{h_j}}$$

**Expert Output**:

Each expert is typically a 2-layer FFN:

$$E_i(\mathbf{x}) = W_{i,2} \cdot \sigma(W_{i,1} \mathbf{x} + \mathbf{b}_{i,1}) + \mathbf{b}_{i,2}$$

where:
- $W_{i,1} \in \mathbb{R}^{d_{ff} \times d}$ (first layer)
- $W_{i,2} \in \mathbb{R}^{d \times d_{ff}}$ (second layer)
- $\sigma$ is activation function (ReLU, GELU, etc.)

**Final Output**:

$$\mathbf{y} = \sum_{i=1}^{E} \frac{e^{h_i}}{\sum_{j=1}^{E} e^{h_j}} \cdot E_i(\mathbf{x})$$

---

## Routing Mechanisms

### 1. Soft Routing (Dense MoE)

**All experts are always used** with weighted combination.

$$\mathbf{y} = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

**Advantages**:
- ✅ Smooth gradients
- ✅ All experts contribute
- ✅ No routing instability

**Disadvantages**:
- ❌ Computationally expensive (all experts active)
- ❌ No conditional computation benefit

### 2. Hard Routing (Sparse MoE)

**Only top-k experts are used**.

**Top-k Selection**:

$$\mathcal{T}_k = \text{TopK}(\mathbf{h}, k)$$

where $\mathcal{T}_k$ contains indices of top-k experts.

**Sparse Gating**:

$$g_i(\mathbf{x}) = \begin{cases}
\frac{e^{h_i}}{\sum_{j \in \mathcal{T}_k} e^{h_j}} & \text{if } i \in \mathcal{T}_k \\
0 & \text{otherwise}
\end{cases}$$

**Output**:

$$\mathbf{y} = \sum_{i \in \mathcal{T}_k} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

**Advantages**:
- ✅ Conditional computation (efficient)
- ✅ Scales to many experts
- ✅ Specialization

**Disadvantages**:
- ❌ Gradient issues (discontinuous)
- ❌ Load imbalance
- ❌ Training instability

### 3. Top-1 Routing (Switch Transformer)

**Extreme case**: Only the single best expert is used ($k=1$).

$$i^* = \arg\max_i h_i$$

$$\mathbf{y} = E_{i^*}(\mathbf{x})$$

**Simplified**: No weighted combination needed!

**Advantages**:
- ✅ Simplest sparse routing
- ✅ Maximum efficiency
- ✅ Easy to implement

**Disadvantages**:
- ❌ No gradient to non-selected experts
- ❌ Severe load imbalance issues
- ❌ Requires careful load balancing

### 4. Noisy Top-k Routing

Add noise to encourage exploration:

$$\mathbf{h} = W_g \mathbf{x} + \text{StandardNormal}() \cdot \text{Softplus}(W_{\text{noise}} \mathbf{x})$$

**Purpose**: 
- Prevents routing collapse (all inputs to same expert)
- Encourages exploration during training
- Smooths the routing distribution

**Noise mechanism**:

$$h_i = (W_g \mathbf{x})_i + \epsilon_i \cdot \text{softplus}((W_{\text{noise}} \mathbf{x})_i)$$

where $\epsilon_i \sim \mathcal{N}(0, 1)$.

Then select top-k based on noisy logits.

---

## Training Challenges and Solutions

### Challenge 1: Load Imbalance

**Problem**: Some experts receive most inputs, others get few/none.

**Example**:
```
Expert 1: 80% of tokens
Expert 2: 15% of tokens
Expert 3: 4% of tokens
Expert 4: 1% of tokens  ← Underutilized!
```

**Consequences**:
- Wasted capacity
- Some experts undertrained
- Routing collapse

**Solution 1: Load Balancing Loss**

Add auxiliary loss to encourage balanced routing:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

where:
- $f_i = \frac{1}{N} \sum_{x \in \text{batch}} \mathbb{1}[i \in \mathcal{T}_k(x)]$ (fraction of tokens routed to expert $i$)
- $P_i = \frac{1}{N} \sum_{x \in \text{batch}} g_i(x)$ (average router probability for expert $i$)
- $\alpha$ is balancing coefficient (typically 0.01)

**Intuition**: Penalize experts that receive high routing probability AND high token fraction.

**Ideal**: $f_i = P_i = 1/E$ for all experts.

**Solution 2: Capacity Factor**

Limit how many tokens each expert can process:

$$\text{Capacity}_i = \frac{k \cdot N}{E} \cdot \text{capacity\_factor}$$

where:
- $N$ is batch size
- $k$ is top-k value
- capacity_factor > 1 allows some overflow (typically 1.25)

Tokens exceeding capacity are handled by residual connection or dropped.

**Solution 3: Expert Choice Routing**

**Instead of tokens choosing experts, experts choose tokens!**

Each expert selects top-k tokens it wants to process based on affinity scores.

$$\text{For expert } i: \text{Select top-k tokens from } \{h_{i,1}, h_{i,2}, ..., h_{i,N}\}$$

This naturally balances load (each expert processes same number of tokens).

### Challenge 2: Gradient Issues

**Problem**: In sparse routing, non-selected experts receive no gradient.

**Solutions**:

**1. Straight-Through Estimator (STE)**

Forward pass: Use discrete top-k selection
Backward pass: Treat as if continuous (pass gradients through)

**2. Gumbel-Softmax**

Approximate discrete sampling with continuous relaxation:

$$g_i = \frac{\exp((h_i + \epsilon_i) / \tau)}{\sum_j \exp((h_j + \epsilon_j) / \tau)}$$

where:
- $\epsilon_i = -\log(-\log(u_i))$, $u_i \sim \text{Uniform}(0,1)$ (Gumbel noise)
- $\tau$ is temperature (lower = more discrete)

**3. Auxiliary Losses**

Explicitly train router to produce useful distributions even for non-selected experts.

### Challenge 3: Routing Collapse

**Problem**: Router learns to send all inputs to a few experts.

**Why it happens**:
- Positive feedback loop
- Experts that process more tokens get better gradients
- Better experts get selected more → even better → selected even more

**Solutions**:

**1. Importance Loss**

Encourage router to use all experts:

$$\mathcal{L}_{\text{importance}} = \text{CV}(\{f_1, f_2, ..., f_E\})^2$$

where CV is coefficient of variation.

**2. Expert Dropout**

Randomly disable some experts during training to force router to learn diverse strategies.

**3. Random Routing**

With small probability, route to random expert instead of top-k.

---

## Architecture Variants

### 1. Standard MoE (Shazeer et al., 2017)

**Architecture**: Replace FFN in Transformer with MoE layer

```
Input
  ↓
Multi-Head Attention
  ↓
Layer Norm
  ↓
MoE Layer (instead of FFN)
  ├─ Expert 1 (FFN)
  ├─ Expert 2 (FFN)
  ├─ Expert 3 (FFN)
  └─ ...
  ↓
Layer Norm
  ↓
Output
```

**Details**:
- Every other FFN replaced with MoE
- Top-2 or top-4 routing
- Capacity factor = 1.25

### 2. Switch Transformer (Fedus et al., 2021)

**Key Innovation**: Top-1 routing (simplest possible)

**Equation**:

$$\mathbf{y} = E_{\arg\max_i (W_g \mathbf{x})_i}(\mathbf{x})$$

**Features**:
- Every FFN replaced with MoE (not every other)
- Expert capacity with residual connections
- Simplified training
- Scales to 1.6T parameters

**Load Balancing**:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \sum_{i=1}^{E} f_i \cdot P_i$$

### 3. Expert Choice Routing (Zhou et al., 2022)

**Key Innovation**: Experts choose tokens, not tokens choose experts

**Algorithm**:

For each expert $i$:
1. Compute affinity scores for all tokens: $s_{i,j} = (W_g \mathbf{x}_j)_i$
2. Select top-k tokens: $\mathcal{T}_i = \text{TopK}(\{s_{i,1}, ..., s_{i,N}\}, k)$
3. Process selected tokens: $\mathbf{y}_j = E_i(\mathbf{x}_j)$ for $j \in \mathcal{T}_i$

**Advantages**:
- Perfect load balance (each expert processes exactly k tokens)
- No capacity factor needed
- No load balancing loss needed

**Disadvantage**:
- Token may be processed by multiple experts (need to combine)

### 4. Soft MoE (Puigcerver et al., 2023)

**Key Innovation**: Experts operate on weighted combinations of ALL tokens

**For expert $i$**:

$$\mathbf{z}_i = \sum_{j=1}^{N} \phi_{i,j} \mathbf{x}_j$$

where $\phi_{i,j}$ is mixing weight from token $j$ to expert $i$.

**Output**:

$$\mathbf{y}_j = \sum_{i=1}^{E} \psi_{j,i} E_i(\mathbf{z}_i)$$

where $\psi_{j,i}$ is weight from expert $i$ to output token $j$.

**Advantages**:
- Fully differentiable
- No discrete routing
- Automatic load balancing

**Disadvantage**:
- Computationally more expensive

### 5. GLaM (Generalist Language Model, 2021)

**Scale**: 1.2 trillion parameters (64 experts × 18.75B each)

**Architecture**:
- Decoder-only Transformer
- MoE every other layer
- Top-2 routing
- 64 experts per MoE layer

**Training**:
- 280B tokens
- Uses only 96B parameters per forward pass
- 3x more efficient than GPT-3 at same quality

### 6. Sparse Upcycling

**Key Idea**: Convert pre-trained dense model to sparse MoE

**Algorithm**:
1. Take pre-trained dense model
2. Replace FFN layers with MoE
3. Initialize each expert with copy of original FFN
4. Continue training (much cheaper than training from scratch)

**Benefits**:
- Reuse expensive pre-training
- Faster convergence
- Better final performance

---

## Sparse vs Dense MoE

### Dense MoE (Soft Routing)

**Computation**:

$$\mathbf{y} = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

All $E$ experts are computed.

**Pros**:
- Smooth optimization
- All experts trained
- Stable training

**Cons**:
- $E \times$ more computation
- No efficiency gain

**Use case**: Small number of experts (2-4)

### Sparse MoE (Hard Routing)

**Computation**:

$$\mathbf{y} = \sum_{i \in \mathcal{T}_k} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

Only $k$ experts are computed.

**Pros**:
- Conditional computation
- Scales to many experts (64-256+)
- Efficient inference

**Cons**:
- Training challenges
- Load balancing needed
- Gradient issues

**Use case**: Large-scale models

### Comparison Table

| Aspect | Dense MoE | Sparse MoE |
|--------|-----------|------------|
| **Experts Used** | All (E) | Top-k (k << E) |
| **Computation** | O(E × d²) | O(k × d²) |
| **Parameters** | E × d² | E × d² |
| **Active Parameters** | E × d² | k × d² |
| **Efficiency** | 1× | E/k × |
| **Training** | Easy | Challenging |
| **Typical E** | 2-8 | 16-256 |
| **Used In** | Early MoE | Modern large models |

---

## Load Balancing

### Metrics

**1. Balance Factor**

$$B = \frac{\sum_{i=1}^{E} f_i^2}{(1/E) \sum_{i=1}^{E} f_i^2} = E \cdot \sum_{i=1}^{E} f_i^2$$

where $f_i$ is fraction of tokens to expert $i$.

- Perfect balance: $B = 1$
- Worse balance: $B > 1$

**2. Coefficient of Variation**

$$\text{CV} = \frac{\sigma(\{f_1, ..., f_E\})}{\mu(\{f_1, ..., f_E\})}$$

Lower CV = better balance.

**3. Entropy of Routing Distribution**

$$H = -\sum_{i=1}^{E} f_i \log f_i$$

- Maximum entropy: $H = \log E$ (perfect balance)
- Lower entropy: worse balance

### Load Balancing Techniques

**1. Auxiliary Loss**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{balance}}$$

**2. Expert Capacity**

Hard limit on tokens per expert:

```python
capacity = (batch_size × seq_len × k / num_experts) × capacity_factor

if expert_i.count > capacity:
    # Option 1: Drop excess tokens
    # Option 2: Route to residual connection
    # Option 3: Route to next-best expert
```

**3. Load Balancing Loss (Detailed)**

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

Minimize when $f_i = P_i = 1/E$ for all $i$.

**Derivation**:

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

If perfectly balanced: $f_i = P_i = 1/E$

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = E \cdot E \cdot \frac{1}{E^2} = \frac{1}{E}$$

This is the minimum value. Any imbalance increases the loss.

**4. Random Token Permutation**

Shuffle token order to prevent positional bias in routing.

**5. Noise in Routing**

Add noise to router logits to encourage exploration:

$$h_i \leftarrow h_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

---

## Modern Implementations

### GPT-4 (Rumored Architecture)

**Speculation based on leaks and analysis**:
- 8 experts per MoE layer
- ~1.8T total parameters
- ~280B active parameters per forward pass
- Top-2 routing
- Expert specialization: different experts for different languages/domains

### Mixtral 8x7B (Mistral AI, 2023)

**Confirmed Architecture**:
- 8 experts, each 7B parameters
- Total: 46.7B parameters
- Active: 12.9B parameters per token (2 experts)
- Top-2 routing
- MoE only in FFN layers
- Open source!

**Performance**:
- Matches GPT-3.5 on many tasks
- 6× faster inference than equivalent dense model
- Outperforms Llama 2 70B on most benchmarks

### Switch Transformer

**Specifications**:
- Up to 1.6T parameters
- 2048 experts (largest variant)
- Top-1 routing
- 7× pre-training speedup vs T5-XXL

**Architecture Details**:
```python
# Pseudocode
def switch_layer(x):
    # Router
    router_logits = linear(x)  # (batch, seq_len, num_experts)
    expert_index = argmax(router_logits, dim=-1)  # (batch, seq_len)
    
    # Dispatch to experts
    for i in range(num_experts):
        mask = (expert_index == i)
        if mask.any():
            expert_input = x[mask]
            expert_output = experts[i](expert_input)
            output[mask] = expert_output
    
    return output
```

### GShard (Google, 2020)

**Scale**: 600B parameters

**Key Features**:
- Top-2 routing
- Expert parallelism (experts on different devices)
- Data parallelism (batches across devices)
- Random routing for load balance

**Parallelism Strategy**:
```
Device 0: Expert 0, 4, 8, 12, ...
Device 1: Expert 1, 5, 9, 13, ...
Device 2: Expert 2, 6, 10, 14, ...
Device 3: Expert 3, 7, 11, 15, ...
```

Tokens are routed to appropriate device based on expert selection.

---

## Practical Considerations

### When to Use MoE?

**✅ Use MoE when**:
- Training very large models (> 100B parameters)
- Have distributed training infrastructure
- Inference budget is limited
- Want conditional computation
- Data has diverse domains/tasks

**❌ Avoid MoE when**:
- Model is small (< 1B parameters)
- Single GPU training
- Don't have engineering resources for complexity
- Serving infrastructure can't handle expert parallelism

### Implementation Challenges

**1. System Complexity**

- Expert parallelism across devices
- All-to-all communication patterns
- Load balancing in distributed setting

**2. Memory Management**

- Each device must hold multiple experts
- Dynamic batching based on routing
- Capacity buffers

**3. Inference Optimization**

- Batching tokens for same expert
- Minimizing expert switches
- KV cache management with sparse activation

### Training Tips

**1. Start with Dense Model**

Train small dense model first, then upscale to MoE.

**2. Gradual Expert Increase**

Start with few experts (4-8), gradually increase.

**3. Monitor Load Balance**

Track routing statistics every checkpoint:
- Expert utilization
- Load balance metrics
- Router entropy

**4. Auxiliary Loss Scheduling**

Start with high α, gradually decrease:
```python
alpha = alpha_max * (1 - step / total_steps)
```

**5. Expert Regularization**

Add L2 regularization to prevent experts from becoming too different:

$$\mathcal{L}_{\text{reg}} = \sum_{i,j} \|W_i - W_j\|^2$$

### Inference Optimization

**1. Expert Caching**

Keep frequently-used experts in fast memory.

**2. Batching Strategy**

Group tokens going to same expert for efficient processing.

**3. Speculative Routing**

Predict which experts will be needed and prefetch.

**4. Expert Pruning**

Remove rarely-used experts after training for smaller deployment.

---

## Mathematical Analysis

### Capacity and Efficiency

**Total Parameters**: $P_{\text{total}} = P_{\text{base}} + E \times P_{\text{expert}}$

**Active Parameters**: $P_{\text{active}} = P_{\text{base}} + k \times P_{\text{expert}}$

**Efficiency Gain**: $\frac{P_{\text{total}}}{P_{\text{active}}} = \frac{P_{\text{base}} + E \times P_{\text{expert}}}{P_{\text{base}} + k \times P_{\text{expert}}}$

For large E and small k: $\approx \frac{E}{k}$

**Example** (Mixtral 8x7B):
- Total: 46.7B parameters
- Active: 12.9B parameters  
- Efficiency: 46.7/12.9 ≈ 3.6× more parameters for same compute

### Load Balance Theory

**Optimal Load**: $f_i^* = \frac{1}{E}$ for all $i$

**Deviation**: $\Delta_i = f_i - \frac{1}{E}$

**Balance Loss (Squared Deviation)**:

$$\mathcal{L}_{\text{balance}} = \sum_{i=1}^{E} \Delta_i^2 = \sum_{i=1}^{E} \left(f_i - \frac{1}{E}\right)^2$$

Minimum when all $\Delta_i = 0$.

### Gradient Flow Analysis

**For selected expert $i \in \mathcal{T}_k$**:

$$\frac{\partial \mathcal{L}}{\partial W_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial E_i(\mathbf{x})} \cdot \frac{\partial E_i(\mathbf{x})}{\partial W_i}$$

**For non-selected expert $j \notin \mathcal{T}_k$**:

$$\frac{\partial \mathcal{L}}{\partial W_j} = 0$$ (direct path)

But router receives gradient:

$$\frac{\partial \mathcal{L}}{\partial W_g} = \sum_{i \in \mathcal{T}_k} \frac{\partial \mathcal{L}}{\partial g_i} \cdot \frac{\partial g_i}{\partial W_g}$$

This allows router to learn which experts to select.

---

## Key Takeaways

### Advantages of MoE

1. ✅ **Scalability**: Trillion-parameter models with manageable compute
2. ✅ **Efficiency**: 3-10× more parameters for same FLOPS
3. ✅ **Specialization**: Experts learn different aspects
4. ✅ **Conditional Compute**: Only activate what's needed

### Challenges

1. ❌ **Training Complexity**: Load balancing, routing stability
2. ❌ **System Complexity**: Distributed training, expert parallelism
3. ❌ **Inference**: Batching, memory management
4. ❌ **Tuning**: Many hyperparameters (α, capacity factor, k, E)

### Design Choices

| Choice | Value | Trade-off |
|--------|-------|-----------|
| **Num Experts (E)** | 8-256 | More experts → more capacity, harder to balance |
| **Top-k** | 1-4 | Lower k → more efficient, less capacity |
| **Capacity Factor** | 1.0-2.0 | Higher → fewer drops, more compute |
| **Balance Loss α** | 0.01-0.1 | Higher → better balance, worse task loss |

### Modern Trends

1. **Larger E**: 2048 experts in Switch Transformer
2. **Top-1 routing**: Simplicity wins (Switch, Mixtral)
3. **Expert choice**: Flip the paradigm
4. **Soft MoE**: Fully differentiable alternative
5. **Sparse Upcycling**: Reuse dense pre-training

---

## References

1. **Shazeer, N., et al.** (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR*.

2. **Fedus, W., et al.** (2021). "Switch Transformers: Scaling to Trillion Parameter Models." *JMLR*.

3. **Lepikhin, D., et al.** (2020). "GShard: Scaling Giant Models with Conditional Computation." *ICLR*.

4. **Du, N., et al.** (2021). "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." *ICML*.

5. **Zhou, Y., et al.** (2022). "Mixture-of-Experts with Expert Choice Routing." *NeurIPS*.

6. **Mistral AI** (2023). "Mixtral of Experts." Technical Report.

---

