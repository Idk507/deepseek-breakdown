# DeepSeek: Complete Technical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Training Pipeline](#training-pipeline)
5. [Key Innovations](#key-innovations)
6. [Performance Metrics](#performance-metrics)

---

## Introduction

DeepSeek is a series of large language models (LLMs) developed by DeepSeek AI, a Chinese AI research company. The DeepSeek family includes several variants, with DeepSeek-V3 being one of the most recent and advanced iterations.

### Key Models
- **DeepSeek-V1**: Initial release (2024)
- **DeepSeek-V2**: Enhanced version with MoE architecture
- **DeepSeek-V3**: Latest model with 671B total parameters
- **DeepSeek-R1**: Reasoning-focused variant

---

## Architecture Overview

### 1. Transformer Foundation

DeepSeek is built on the **Transformer architecture**, which uses self-attention mechanisms to process sequential data.

#### Basic Transformer Equation
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = Dimension of key vectors

### 2. Mixture of Experts (MoE)

DeepSeek-V2 and V3 employ **Mixture of Experts** architecture for efficiency.

#### MoE Architecture Parameters (DeepSeek-V3)
- **Total Parameters**: 671 billion
- **Active Parameters per Token**: 37 billion
- **Number of Experts**: 256
- **Experts Activated per Token**: 8
- **Expert Size**: ~2.6B parameters each

#### MoE Computation

For each token, the router selects top-k experts:

```
Router_Score(x) = Softmax(W_g · x)
Output = Σ(i=1 to k) G_i(x) · Expert_i(x)
```

Where:
- x = input token embedding
- W_g = gating network weights
- G_i(x) = gating probability for expert i
- k = number of active experts (8 in DeepSeek-V3)

**Efficiency Calculation:**
```
Active_Ratio = Active_Parameters / Total_Parameters
             = 37B / 671B
             ≈ 5.5%
```

This means only 5.5% of the model is active for any given token, drastically reducing computational cost.

---

## Mathematical Foundations

### 1. Multi-Head Attention

DeepSeek uses **Multi-Head Attention (MHA)** or **Grouped Query Attention (GQA)** for efficient processing.

#### Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Parameters:
- h = number of attention heads
- W_i^Q, W_i^K, W_i^V = projection matrices for head i
- W^O = output projection matrix

#### Grouped Query Attention (GQA)

GQA reduces memory usage by sharing key/value heads across multiple query heads:

```
Number of KV heads = h / group_size
Memory_Savings = 1 - (1/group_size)
```

For group_size = 8:
```
Memory_Savings = 1 - (1/8) = 87.5% reduction in KV cache
```

### 2. Positional Encoding

DeepSeek uses **Rotary Position Embedding (RoPE)** for position awareness.

#### RoPE Mathematics

For a vector x at position m:

```
RoPE(x_m) = [
  x_0 · cos(mθ_0) - x_1 · sin(mθ_0),
  x_0 · sin(mθ_0) + x_1 · cos(mθ_0),
  x_2 · cos(mθ_1) - x_3 · sin(mθ_1),
  x_2 · sin(mθ_1) + x_3 · cos(mθ_1),
  ...
]
```

Where θ_i = 10000^(-2i/d) for dimension d

### 3. Layer Normalization

DeepSeek uses **RMSNorm** (Root Mean Square Normalization) instead of LayerNorm:

```
RMSNorm(x) = x / RMS(x) · γ

where RMS(x) = √(1/n Σ(x_i²))
```

Benefits:
- 10-15% faster than LayerNorm
- Simpler computation (no mean subtraction)
- Better numerical stability

### 4. Activation Functions

DeepSeek typically uses **SwiGLU** activation:

```
SwiGLU(x) = Swish(xW_1) ⊙ (xW_2)
where Swish(x) = x · sigmoid(x)
```

Or expressed mathematically:
```
SwiGLU(x) = (x · σ(x)) ⊙ (xW)
where σ(x) = 1/(1 + e^(-x))
```

---

## Training Pipeline

### 1. Pre-training

#### Dataset
- **Size**: ~15 trillion tokens (estimated for V3)
- **Composition**: 
  - Code: ~30%
  - English text: ~40%
  - Chinese text: ~20%
  - Other languages: ~10%

#### Training Objective

Cross-entropy loss for next-token prediction:

```
L = -Σ(i=1 to T) log P(x_i | x_1, ..., x_(i-1))
```

Where:
- T = sequence length
- x_i = token at position i
- P = predicted probability distribution

#### Optimization

**AdamW Optimizer** with the following hyperparameters:

```
Learning_Rate Schedule:
lr(t) = lr_max · min(t/warmup, 1) · decay_factor

where:
- lr_max = 3e-4 (typical)
- warmup = 2000 steps
- decay_factor = (1 - t/total_steps)^0.5
```

**Gradient Clipping:**
```
g' = g · min(1, max_norm / ||g||)
where max_norm = 1.0
```

### 2. Fine-tuning

#### Supervised Fine-Tuning (SFT)

Loss function:
```
L_SFT = -Σ log P(y_i | x, y_<i)
```

Where:
- x = input prompt
- y = target completion
- y_<i = previous tokens in completion

#### Reinforcement Learning from Human Feedback (RLHF)

**Reward Model Training:**
```
L_RM = -E[log σ(r_w - r_l)]
```

Where:
- r_w = reward for winning response
- r_l = reward for losing response
- σ = sigmoid function

**PPO (Proximal Policy Optimization):**
```
L_PPO = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)
- A = advantage function
- ε = clipping parameter (0.2)
```

### 3. Training Efficiency Metrics

#### FLOPs Calculation

For a single forward pass:

```
FLOPs ≈ 2 · P · T

where:
- P = number of parameters
- T = number of training tokens
```

For DeepSeek-V3:
```
FLOPs ≈ 2 · 671B · 15T
     ≈ 2 · 10^13 FLOPs
```

#### GPU Hours

Estimated training time:
```
GPU_Hours = FLOPs / (GPU_FLOPs/s · 3600)

For H100 (1000 TFLOPS):
GPU_Hours ≈ 2·10^13 / (10^15 · 3600)
         ≈ 5.5 million GPU hours
```

---

## Key Innovations

### 1. Multi-Token Prediction (MTP)

DeepSeek-V3 introduces **Multi-Token Prediction** during training.

#### Standard Training
```
P(x_i | x_<i)  [predicts only next token]
```

#### Multi-Token Prediction
```
P(x_i, x_(i+1), ..., x_(i+n) | x_<i)  [predicts n future tokens]
```

**Benefits:**
- Better long-range planning
- Improved coherence
- 15-20% better sample efficiency

**Loss Function:**
```
L_MTP = -Σ(j=1 to n) α_j · log P(x_(i+j) | x_<i)

where α_j = decay weight for token j positions ahead
```

### 2. Expert Specialization

DeepSeek's MoE learns domain-specific experts automatically.

#### Load Balancing Loss

To prevent expert collapse:
```
L_load = α · Σ(i=1 to N) (f_i - 1/N)²

where:
- f_i = fraction of tokens routed to expert i
- N = total number of experts
- α = balancing coefficient (0.01)
```

**Total Training Loss:**
```
L_total = L_CE + λ_1·L_load + λ_2·L_aux
```

### 3. Auxiliary Loss

Auxiliary prediction heads improve representation learning:

```
L_aux = Σ L_task_specific

Examples:
- Next sentence prediction
- Masked language modeling
- Code completion
```

### 4. FP8 Mixed Precision Training

DeepSeek pioneered efficient **FP8 training** for large models.

#### Number Representation
- **FP32**: 1 sign + 8 exponent + 23 mantissa bits
- **FP16**: 1 sign + 5 exponent + 10 mantissa bits
- **FP8**: 1 sign + 4 exponent + 3 mantissa bits

**Dynamic Range:**
```
FP8 range = 2^(-bias) to 2^(max_exp - bias)
         ≈ 2^(-6) to 2^8
         ≈ 0.015625 to 256
```

**Memory Savings:**
```
Savings = 1 - (FP8_size / FP32_size)
        = 1 - (8/32)
        = 75% memory reduction
```

---

## Performance Metrics

### 1. Benchmark Performance

#### Mathematical Reasoning (GSM8K, MATH)

**GSM8K Accuracy:**
```
Accuracy = Correct_Solutions / Total_Problems
DeepSeek-V3: ~92%
```

**MATH Dataset:**
```
Pass@1 = Problems_Solved_First_Try / Total_Problems
DeepSeek-V3: ~78%
```

### 2. Code Generation (HumanEval)

```
Pass@k = 1 - (N-C choose k) / (N choose k)

where:
- N = total samples generated
- C = correct samples
- k = number of attempts
```

**DeepSeek-V3 Performance:**
- Pass@1: ~85%
- Pass@10: ~95%

### 3. Inference Speed

#### Tokens per Second

```
TPS = Batch_Size · Seq_Length / Inference_Time

Typical DeepSeek-V3 on 8×A100:
TPS ≈ 45-60 tokens/second (batch=1)
```

#### Latency Calculation

```
First_Token_Latency = Prefill_Time
Next_Token_Latency = Decode_Time

Total_Latency = Prefill_Time + (N-1) · Decode_Time
```

For 2048 token generation:
```
Total_Latency ≈ 0.5s + 2047 · 0.02s
              ≈ 41 seconds
```

### 4. Perplexity

Perplexity measures model uncertainty:

```
PPL = exp(-1/T · Σ log P(x_i | x_<i))
    = exp(Cross_Entropy_Loss)
```

Lower perplexity = better model

**DeepSeek-V3 Perplexity:**
- English: ~8.5
- Code: ~12.3
- Chinese: ~11.2

---

## Advanced Topics

### 1. Scaling Laws

DeepSeek follows neural scaling laws:

```
L(N, D) = A·N^(-α) + B·D^(-β) + C

where:
- L = loss
- N = model size (parameters)
- D = dataset size (tokens)
- α, β = scaling exponents (~0.076, ~0.095)
```

**Compute-Optimal Scaling:**
```
N_optimal ∝ C^α
D_optimal ∝ C^β

where C = compute budget
```

### 2. Context Window Extension

DeepSeek extends context using **YaRN** (Yet another RoPE extensioN):

```
θ'_i = θ_i · s(i) · t^λ

where:
- s(i) = interpolation scale
- t = temperature
- λ = extension factor
```

This allows extending from 4K to 32K+ context with minimal degradation.

### 3. Quantization

#### Post-Training Quantization

Convert FP16 weights to INT8:

```
x_q = round(x / scale) + zero_point

Quantization_Error = ||x - Dequant(x_q)||²
```

**GPTQ (Gradient-based PTQ):**
```
min ||WX - W_q X||²

Solved layer-by-layer with Hessian information
```

**Compression Ratio:**
```
Ratio = Original_Size / Quantized_Size
      = 16 bits / 8 bits
      = 2× compression
```

### 4. Distillation

Creating smaller models from DeepSeek:

```
L_distill = α·L_CE + (1-α)·KL(P_student || P_teacher)

where:
- L_CE = cross-entropy loss
- KL = Kullback-Leibler divergence
- α = balance parameter
```

**Temperature Scaling:**
```
P(x_i) = exp(z_i/T) / Σ exp(z_j/T)

Higher T → softer probabilities → better distillation
```

---

## Practical Implementation

### 1. Memory Requirements

#### Parameter Memory

```
Param_Memory = Num_Params · Bytes_Per_Param

For FP16:
- DeepSeek-V3: 671B × 2 bytes = 1,342 GB
- Active (37B): 37B × 2 bytes = 74 GB
```

#### KV Cache Memory

```
KV_Cache = 2 · Layers · Heads · Head_Dim · Seq_Len · Batch · Bytes

Example (32K context, batch=1):
KV_Cache = 2 · 60 · 128 · 128 · 32768 · 1 · 2
         ≈ 64 GB
```

#### Total Inference Memory

```
Total = Param_Memory + KV_Cache + Activations + Overhead
      ≈ 74 GB + 64 GB + 10 GB + 10 GB
      ≈ 158 GB
```

### 2. Batch Processing

#### Throughput Optimization

```
Effective_Throughput = (Batch_Size · Seq_Length) / Time

Optimal_Batch_Size = GPU_Memory / (Model_Size + KV_Cache)
```

### 3. Pipeline Parallelism

For distributed inference:

```
Num_Devices = ⌈Total_Memory / Device_Memory⌉
Latency_Increase = Num_Devices · Communication_Overhead
```

---

## Comparison with Other Models

### Parameter Efficiency

| Model | Total Params | Active Params | Efficiency |
|-------|-------------|---------------|------------|
| GPT-4 | ~1.8T (rumored) | ~1.8T | 100% |
| DeepSeek-V3 | 671B | 37B | 5.5% |
| Mixtral-8x7B | 47B | 13B | 27.7% |

### Cost Efficiency

```
Cost_Per_Token = (GPU_Hours · GPU_Cost) / Total_Tokens

DeepSeek advantage:
- 3-5× cheaper training than comparable models
- 2-3× cheaper inference due to MoE
```

---

## Summary

DeepSeek represents a significant advancement in efficient large language model design through:

1. **Mixture of Experts**: 671B total, 37B active parameters
2. **Advanced Training**: Multi-token prediction, FP8 precision
3. **Optimized Inference**: Fast token generation, low memory
4. **Strong Performance**: Competitive with larger models
5. **Cost Effective**: Significantly cheaper to train and run

### Key Mathematical Innovations

- **MoE Routing**: Top-k expert selection with load balancing
- **RoPE + YaRN**: Extended context windows
- **Multi-Token Prediction**: Improved sample efficiency
- **FP8 Training**: 75% memory reduction

### Future Directions

- Larger context windows (128K+)
- Better reasoning capabilities
- Multimodal extensions
- Further efficiency improvements

---

## References

- DeepSeek Technical Reports (2024)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Mixtral of Experts" (Jiang et al., 2024)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)

---

*Last Updated: February 2026*
*Document Version: 1.0*
