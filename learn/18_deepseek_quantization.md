# DeepSeek's LLM Quantization: Complete Implementation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [DeepSeek's FP8 Training](#deepseeks-fp8-training)
4. [Post-Training Quantization](#post-training-quantization)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Per-Tensor vs Per-Channel Quantization](#per-tensor-vs-per-channel-quantization)
7. [Dynamic vs Static Quantization](#dynamic-vs-static-quantization)
8. [KV Cache Quantization](#kv-cache-quantization)
9. [Activation Quantization](#activation-quantization)
10. [Weight Quantization](#weight-quantization)
11. [Mixed Precision Strategies](#mixed-precision-strategies)
12. [Training Techniques](#training-techniques)
13. [Inference Optimization](#inference-optimization)
14. [Detailed Examples](#detailed-examples)

---

## Introduction

### What is LLM Quantization?

**Quantization** is the process of reducing the precision of model weights and activations from high-precision formats (FP32, FP16) to lower-precision formats (INT8, FP8, INT4) to reduce memory and computational requirements.

**Core Idea:**
> Represent model parameters and activations with fewer bits while maintaining acceptable accuracy.

### DeepSeek's Quantization Journey

```
DeepSeek-V2 (May 2024):
├── Training: FP16 mixed precision
├── Inference: FP16/BF16
├── Post-training: INT8/INT4 quantization
└── KV cache: INT8 quantization

DeepSeek-V3 (December 2024):
├── Training: FP8 mixed precision (breakthrough!)
├── Inference: FP8 native support
├── Post-training: INT8/INT4 with advanced techniques
├── KV cache: INT8/INT4 with group quantization
└── Activation: Dynamic FP8 quantization

Key Achievement:
- First large-scale (671B) FP8 training
- <0.1% quality loss with FP8
- 2× memory reduction
- 2× speedup on H100/H800
```

### Why Quantization Matters

```
DeepSeek-V3 (671B parameters):

FP32 Precision:
  Weights: 671B × 4 bytes = 2.68 TB
  Activations: ~100 GB per batch
  Total: ~2.78 TB per batch
  → Impossible to run!

FP16 Precision:
  Weights: 671B × 2 bytes = 1.34 TB
  Activations: ~50 GB per batch
  Total: ~1.39 TB per batch
  → Still very expensive

FP8 Precision (DeepSeek-V3):
  Weights: 671B × 1 byte = 671 GB
  Activations: ~25 GB per batch
  Total: ~696 GB per batch
  → Fits on 8× H100 (640 GB total)

INT4 Quantization (Post-training):
  Weights: 671B × 0.5 bytes = 335.5 GB
  Activations: FP16 (25 GB)
  Total: ~360.5 GB
  → Can run on 4× A100 (320 GB total)

Impact: 7.7× memory reduction (FP32 → INT4)
```

---

## Quantization Fundamentals

### Number Representation

**Floating Point Formats:**

```
FP32 (32-bit float):
├── Sign: 1 bit
├── Exponent: 8 bits
├── Mantissa: 23 bits
├── Range: ~10^-38 to 10^38
└── Precision: ~7 decimal digits

FP16 (16-bit float):
├── Sign: 1 bit
├── Exponent: 5 bits
├── Mantissa: 10 bits
├── Range: ~10^-8 to 65504
└── Precision: ~3 decimal digits

BF16 (Brain Float 16):
├── Sign: 1 bit
├── Exponent: 8 bits (same as FP32!)
├── Mantissa: 7 bits
├── Range: Same as FP32
└── Precision: ~2 decimal digits

FP8 E4M3 (8-bit float):
├── Sign: 1 bit
├── Exponent: 4 bits
├── Mantissa: 3 bits
├── Range: -448 to 448
└── Precision: Limited

FP8 E5M2 (8-bit float):
├── Sign: 1 bit
├── Exponent: 5 bits
├── Mantissa: 2 bits
├── Range: -57344 to 57344
└── Precision: Very limited
```

**Integer Formats:**

```
INT8 (8-bit integer):
├── Signed: -128 to 127
├── Unsigned: 0 to 255
└── No decimal precision

INT4 (4-bit integer):
├── Signed: -8 to 7
├── Unsigned: 0 to 15
└── No decimal precision
```

### Basic Quantization Formula

```
Symmetric Quantization:

Q(x) = round(x / s)
x̂ = Q(x) × s

where:
  x: original value (FP32/FP16)
  s: scale factor
  Q(x): quantized value (INT8/INT4)
  x̂: dequantized value (approximation of x)

Scale factor:
  s = max(|x|) / (2^(bits-1) - 1)

Example (FP16 → INT8):
  x = [0.5, -1.2, 0.8, -0.3]
  max(|x|) = 1.2
  s = 1.2 / 127 = 0.00945
  
  Q(x) = round([0.5, -1.2, 0.8, -0.3] / 0.00945)
       = round([52.9, -127.0, 84.7, -31.7])
       = [53, -127, 85, -32]
  
  x̂ = [53, -127, 85, -32] × 0.00945
    = [0.501, -1.200, 0.803, -0.302]

Error = |x - x̂| = [0.001, 0.000, 0.003, 0.002]
```

**Asymmetric Quantization:**

```
Q(x) = round((x - z) / s)
x̂ = Q(x) × s + z

where:
  z: zero-point offset
  s: scale factor

Scale and zero-point:
  s = (x_max - x_min) / (2^bits - 1)
  z = x_min

Used when data is not centered around zero
```

---

## DeepSeek's FP8 Training

### FP8 Format Selection

DeepSeek-V3 uses two FP8 formats:

```
E4M3 (4-bit exponent, 3-bit mantissa):
├── Range: -448 to 448
├── Better precision
├── Used for: Gradients
└── Why: Gradients need precision more than range

E5M2 (5-bit exponent, 2-bit mantissa):
├── Range: -57344 to 57344
├── Better range
├── Used for: Forward activations
└── Why: Activations can have large outliers

Hybrid Strategy:
  Forward pass: E5M2 for activations
  Backward pass: E4M3 for gradients
  Weights: E4M3 (stored in FP8, computed in FP32)
```

### FP8 Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              DeepSeek-V3 FP8 Training Pipeline              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Master Weights (FP32)                                       │
│    │                                                         │
│    ↓ Cast                                                    │
│  Working Weights (FP8-E4M3)                                  │
│    │                                                         │
│    ↓                                                         │
│  ┌────────────────────────────────┐                         │
│  │      Forward Pass              │                         │
│  │                                │                         │
│  │  Input (FP8-E5M2)              │                         │
│  │    ↓                           │                         │
│  │  MatMul (FP8 × FP8)            │                         │
│  │    ↓                           │                         │
│  │  Activations (FP8-E5M2)        │                         │
│  │    ↓                           │                         │
│  │  Layer Norm (FP32)             │ ← Higher precision      │
│  │    ↓                           │                         │
│  │  Output (FP8-E5M2)             │                         │
│  └────────────────────────────────┘                         │
│    │                                                         │
│    ↓                                                         │
│  Loss Computation (FP32)                                     │
│    │                                                         │
│    ↓                                                         │
│  ┌────────────────────────────────┐                         │
│  │      Backward Pass             │                         │
│  │                                │                         │
│  │  Gradient (FP8-E4M3)           │                         │
│  │    ↓                           │                         │
│  │  MatMul (FP8 × FP8)            │                         │
│  │    ↓                           │                         │
│  │  Accumulate (FP32)             │ ← Higher precision      │
│  │    ↓                           │                         │
│  │  Gradient Clipping (FP32)      │                         │
│  └────────────────────────────────┘                         │
│    │                                                         │
│    ↓                                                         │
│  Optimizer Step (FP32)                                       │
│    │                                                         │
│    ↓                                                         │
│  Update Master Weights (FP32)                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Key Points:
- Master weights always in FP32 (for optimizer)
- Working weights cast to FP8 each forward pass
- Activations in FP8 (E5M2 for range)
- Gradients in FP8 (E4M3 for precision)
- Critical ops (LayerNorm, Loss) in FP32
- Gradient accumulation in FP32
```

### Per-Tensor Dynamic Scaling

```
Dynamic Scaling Algorithm:

For each tensor T in forward/backward:
  
  1. Compute statistics:
     T_max = max(|T|)
     T_amax = running_max(T_max)  # Exponential moving average
  
  2. Compute scale:
     s = T_amax / FP8_MAX
     
     where FP8_MAX:
       E4M3: 448
       E5M2: 57344
  
  3. Quantize:
     T_fp8 = clamp(T / s, -FP8_MAX, FP8_MAX)
  
  4. Store scale for dequantization:
     scales[tensor_name] = s

Example:
  Activation tensor: [-2.5, 1.8, -0.9, 3.2, 0.5]
  T_max = 3.2
  T_amax = 3.5 (from moving average)
  
  For E5M2:
    s = 3.5 / 57344 = 6.1e-5
    T_fp8 = [-2.5, 1.8, -0.9, 3.2, 0.5] / 6.1e-5
          = [-40983, 29508, -14754, 52459, 8197]
    Clamp to E5M2 range and represent in FP8

Dequantization:
  T_fp16 = T_fp8 × s
```

### Mixed Precision Strategy

```
Operation-Level Precision Assignment:

FP8 Operations:
├── Matrix multiplication (matmul)
├── Convolution
├── Element-wise multiplication
└── Embedding lookup

FP32 Operations (for stability):
├── Layer Normalization
├── Batch Normalization
├── Softmax
├── Loss computation
├── Gradient clipping
└── Optimizer updates

FP16/BF16 Operations (optional):
├── Residual connections
├── Dropout
└── GELU activation (can be FP8)

Rationale:
  FP8: High compute, tolerant to precision loss
  FP32: Normalization needs precision for stability
  FP16: Good balance for other ops
```

---

## Post-Training Quantization

### Weight-Only Quantization

```
DeepSeek INT8 Weight Quantization:

Per-Channel Quantization:

For each weight matrix W ∈ ℝ^(m×n):
  
  For each output channel i (row):
    1. Compute scale:
       s_i = max(|W[i, :]|) / 127
    
    2. Quantize:
       W_q[i, :] = round(W[i, :] / s_i)
    
    3. Store:
       - W_q[i, :] as INT8
       - s_i as FP16

Dequantization during inference:
  W_fp16[i, :] = W_q[i, :] × s_i

Memory:
  Original: m × n × 2 bytes (FP16)
  Quantized: m × n × 1 + m × 2 bytes (INT8 + scales)
  Ratio: ~50% reduction

Example:
  W = [[0.5, -1.2, 0.8],
       [-0.3, 0.9, -0.6]]
  
  Channel 0: max = 1.2, s_0 = 1.2/127 = 0.00945
    W_q[0,:] = [53, -127, 85]
  
  Channel 1: max = 0.9, s_1 = 0.9/127 = 0.00709
    W_q[1,:] = [-42, 127, -85]
```

### INT4 Weight Quantization

```
Group Quantization for INT4:

Group size: 128 (typical)

For weight matrix W:
  1. Divide into groups of 128 elements
  2. Quantize each group separately
  
  For each group G:
    s_G = max(|G|) / 7  # INT4 range: -8 to 7
    G_q = round(G / s_G)
    Clip to [-8, 7]

Storage:
  - Quantized weights: 0.5 bytes per weight
  - Scales: 1 scale per 128 weights (FP16)

Example (group_size=4 for demo):
  W = [0.5, -1.2, 0.8, -0.3, 0.9, -0.6, 0.4, -0.7]
  
  Group 1: [0.5, -1.2, 0.8, -0.3]
    s_1 = 1.2 / 7 = 0.171
    G1_q = round([0.5, -1.2, 0.8, -0.3] / 0.171)
         = [3, -7, 5, -2]
  
  Group 2: [0.9, -0.6, 0.4, -0.7]
    s_2 = 0.9 / 7 = 0.129
    G2_q = round([0.9, -0.6, 0.4, -0.7] / 0.129)
         = [7, -5, 3, -5]

Accuracy vs FP16: <1% perplexity increase
Memory: 4× reduction
```

### GPTQ (Gradient-based Post-Training Quantization)

```
DeepSeek uses GPTQ for high-quality INT4:

Algorithm:
  For each layer:
    1. Collect calibration data (1024 samples)
    2. Compute Hessian matrix H = X^T X
    3. Quantize weights sequentially:
       
       For each column j:
         a. Quantize w_j
         b. Compute error: e = w_j - w_q_j
         c. Update remaining weights:
            w_k = w_k - (H[j,k]/H[j,j]) × e
            for all k > j

Mathematical formulation:
  Minimize: ||XW - XW_q||²
  
  Where:
    X: calibration activations
    W: original weights
    W_q: quantized weights

Benefits:
  - Considers layer interactions
  - Minimizes activation error (not just weight error)
  - Better than naive rounding
  - ~0.5% perplexity increase vs FP16

DeepSeek-V3 INT4 GPTQ results:
  Perplexity increase: 0.3-0.5%
  Memory: 4× reduction
  Speed: 2-3× faster than FP16
```

---

## Mathematical Foundation

### Quantization Error Analysis

```
Quantization Error:

E = ||x - x̂||²

where:
  x: original value
  x̂: quantized then dequantized value

For uniform quantization:
  E ≈ (s²/12) for each element
  
  where s = scale factor

Total error for tensor:
  E_total = n × (s²/12)
  
  where n = number of elements

Signal-to-Quantization-Noise Ratio:
  SQNR = 10 × log₁₀(σ²_signal / σ²_error)
  
  For b bits:
    SQNR ≈ 6.02b + 1.76 dB

Examples:
  8-bit: SQNR ≈ 50 dB (excellent)
  4-bit: SQNR ≈ 26 dB (acceptable)
  2-bit: SQNR ≈ 14 dB (poor)
```

### Optimal Scale Computation

```
Minimize quantization error:

Optimal scale for symmetric quantization:
  s* = argmin_s E(s)
     = argmin_s Σ(x_i - round(x_i/s)×s)²

Closed form (for uniform distribution):
  s* = (max(|x|) + ε) / (2^(b-1) - 1)
  
  where:
    b: number of bits
    ε: small constant for numerical stability

For asymmetric quantization:
  s* = (x_max - x_min) / (2^b - 1)
  z* = -round(x_min / s*)

DeepSeek uses percentile-based clipping:
  Instead of max(|x|), use 99.9th percentile
  Reduces effect of outliers
  Better SQNR in practice
```

### Matrix Multiplication Quantization

```
Quantized Matrix Multiplication:

Standard:
  Y = XW
  
  where:
    X ∈ ℝ^(m×k): activations (FP16)
    W ∈ ℝ^(k×n): weights (FP16)
    Y ∈ ℝ^(m×n): output (FP16)

INT8 Quantization:
  Y_q = (X_q × W_q) × (s_x × s_w)
  
  where:
    X_q ∈ INT8^(m×k): quantized activations
    W_q ∈ INT8^(k×n): quantized weights
    s_x, s_w: scale factors
    Y_q: quantized output (then dequantized)

Computational cost:
  FP16: m × k × n MACs (multiply-accumulate)
  INT8: m × k × n MACs (but faster!)
  
  Speedup: 2-4× on modern hardware

Accumulation:
  Must use INT32 for accumulation to avoid overflow
  
  For each output element:
    acc = 0 (INT32)
    for i in range(k):
      acc += X_q[i] × W_q[i]  # INT8 × INT8 → INT16 → INT32
    
    Y[j] = acc × (s_x × s_w)  # Scale back to FP16
```

---

## Per-Tensor vs Per-Channel Quantization

### Per-Tensor Quantization

```
Single scale for entire tensor:

Quantization:
  s = max(|W|) / 127
  W_q = round(W / s)

Storage:
  - Tensor: INT8
  - Scale: 1 FP16 value

Pros:
  + Simple implementation
  + Minimal overhead
  + Fast dequantization

Cons:
  - Less accurate for heterogeneous tensors
  - Outliers affect entire tensor

Example:
  W = [[0.1, 0.2],
       [5.0, 4.8]]
  
  max = 5.0
  s = 5.0 / 127 = 0.0394
  
  W_q = [[3, 5],      # Fine for small values
         [127, 122]]   # Good for large values
  
  But: small values lose precision
```

### Per-Channel Quantization

```
Separate scale for each channel:

For weights W ∈ ℝ^(out_ch × in_ch):
  
  For each output channel i:
    s_i = max(|W[i, :]|) / 127
    W_q[i, :] = round(W[i, :] / s_i)

Storage:
  - Tensor: INT8
  - Scales: out_ch FP16 values

Pros:
  + Much better accuracy
  + Handles heterogeneous channels
  + Industry standard

Cons:
  - Slightly more complex
  - More scale factors to store

Example (same data):
  W = [[0.1, 0.2],     # Channel 0
       [5.0, 4.8]]     # Channel 1
  
  Channel 0: s_0 = 0.2 / 127 = 0.00157
    W_q[0, :] = [64, 127]  # Better precision!
  
  Channel 1: s_1 = 5.0 / 127 = 0.0394
    W_q[1, :] = [127, 122]

DeepSeek uses per-channel for weights
```

### Group Quantization (for INT4)

```
Compromise between per-tensor and per-channel:

Group size: 32, 64, 128 (typical)

Quantization:
  Divide channel into groups
  Each group has own scale
  
  For channel with 1024 elements, group_size=128:
    - 8 groups
    - 8 scales per channel

Storage overhead:
  Scales: (tensor_size / group_size) × 2 bytes
  
  For 1B parameters, group_size=128:
    Scales: (1B / 128) × 2 = 15.6 MB (negligible)

Accuracy:
  Better than per-tensor
  Slightly worse than per-element
  Good trade-off

DeepSeek-V3 uses group_size=128 for INT4
```

---

## Dynamic vs Static Quantization

### Static Quantization

```
Calibration phase:
  1. Run model on calibration dataset
  2. Collect statistics (min, max, histogram)
  3. Compute optimal scales
  4. Quantize model
  5. Fixed scales for inference

Process:
  # Calibration
  for batch in calibration_data:
    output = model(batch)
    collect_stats(activations)
  
  scales = compute_scales(stats)
  
  # Quantize
  model_q = quantize(model, scales)
  
  # Inference (scales fixed)
  output = model_q(new_input)

Pros:
  + Fast inference (no runtime overhead)
  + Predictable performance
  + Can optimize scales offline

Cons:
  - Requires calibration data
  - May not adapt to distribution shift
  - One-size-fits-all scales

Used for: Weight quantization in DeepSeek
```

### Dynamic Quantization

```
Runtime quantization:
  For each forward pass:
    1. Compute input statistics
    2. Calculate scales
    3. Quantize
    4. Compute
    5. Dequantize

Process:
  def forward(x):
    # Compute scale at runtime
    s_x = max(|x|) / 127
    
    # Quantize
    x_q = round(x / s_x)
    
    # Compute (using quantized values)
    y_q = matmul_int8(x_q, w_q)
    
    # Dequantize
    y = y_q × (s_x × s_w)
    
    return y

Pros:
  + Adapts to input distribution
  + Better accuracy for varying inputs
  + No calibration needed

Cons:
  - Runtime overhead for scale computation
  - Slightly slower

Used for: Activation quantization in DeepSeek
```

---

## KV Cache Quantization

### Problem: KV Cache Memory

```
DeepSeek-V3 KV cache (FP16):
  
  Per layer: 512 × seq_len × 2 bytes
  60 layers: 60 × 512 × seq_len × 2
  
  For 128K context:
    = 60 × 512 × 131072 × 2
    = 8.05 GB per sample

For batch of 8:
  = 64.4 GB just for KV cache!

Even with MLA compression, still significant
```

### INT8 KV Cache Quantization

```
DeepSeek's KV Cache Quantization:

Per-token quantization:
  For each new token t:
    1. Compute C_kv (compressed KV)
    2. Quantize:
       s_t = max(|C_kv[t, :]|) / 127
       C_kv_q[t, :] = round(C_kv[t, :] / s_t)
    
    3. Store:
       - C_kv_q[t, :] as INT8
       - s_t as FP16

Dequantization (during attention):
  C_kv_fp16[t, :] = C_kv_q[t, :] × s_t

Memory reduction:
  FP16: 512 × 131072 × 2 = 134 MB per layer
  INT8: 512 × 131072 × 1 + 131072 × 2 = 67.6 MB
  Reduction: ~50%

Quality impact: <0.5% perplexity increase
```

### INT4 KV Cache (Aggressive)

```
Group quantization for KV cache:

Group size: 32

For each token:
  Divide C_kv into groups of 32
  Quantize each group to INT4
  
  For each group G:
    s_G = max(|G|) / 7
    G_q = round(G / s_G)

Memory:
  INT4: 512 × 131072 × 0.5 = 33.5 MB
  Scales: (512 × 131072 / 32) × 2 = 4.2 MB
  Total: 37.7 MB per layer
  
  Reduction: 3.6× vs FP16

Quality: ~1% perplexity increase
Acceptable for most applications
```

---

## Activation Quantization

### Challenge: Outliers

```
Activation statistics:

Typical distribution:
  99% of values: [-3, 3]
  0.9% of values: [-10, 10]
  0.1% of values: up to ±1000 (outliers!)

Problem with naive quantization:
  s = 1000 / 127 = 7.87
  
  Most values quantized to:
    Q(2.0) = round(2.0 / 7.87) = 0
    Q(1.5) = round(1.5 / 7.87) = 0
  
  Severe precision loss for normal values!
```

### Outlier-Aware Quantization

```
DeepSeek's approach:

1. Percentile Clipping:
   Instead of max, use 99.9th percentile
   
   x_clip = percentile(|x|, 99.9)
   s = x_clip / 127
   
   Values beyond percentile clipped:
     x_clipped = clip(x, -x_clip, x_clip)

2. Mixed-Precision Outliers:
   Keep outliers in FP16
   Quantize rest to INT8
   
   outlier_mask = |x| > threshold
   x_q[~outlier_mask] = quantize(x[~outlier_mask])
   x_fp16[outlier_mask] = x[outlier_mask]

3. Smoothing:
   Apply scaling before quantization
   Reduces outlier magnitude
   
   x_smooth = x / sqrt(Var(x) + ε)
   x_q = quantize(x_smooth)
```

---

## Weight Quantization

### Layer-Wise Quantization

```
DeepSeek quantizes different layers differently:

Attention Layers:
  - Q, K, V projections: INT8 per-channel
  - Output projection: INT8 per-channel
  - Rationale: High compute, benefits from quantization

MoE Expert Layers:
  - Gate/Up projections: INT4 group quantization
  - Down projections: INT4 group quantization
  - Rationale: Huge parameter count, need aggressive compression

Shared Experts:
  - Higher precision: INT8 per-channel
  - Rationale: Always active, quality critical

Embedding Layer:
  - Keep FP16
  - Rationale: Lookup table, no computation benefit

Layer Norm:
  - Keep FP32
  - Rationale: Normalization needs precision
```

### Weight Smoothing

```
SmoothQuant technique:

Problem:
  Activations have outliers → hard to quantize
  Weights smooth → easy to quantize

Solution:
  Transfer difficulty from activations to weights
  
  For layer with weights W and activations X:
    Y = X W
  
  Transform:
    s = smooth_factor (per-channel)
    Y = (X / s) × (s × W)
  
  Now:
    X̃ = X / s  (smoother activations)
    W̃ = s × W  (rougher weights, but still ok)
  
  Quantize:
    X̃_q = quantize(X̃)  # Easier!
    W̃_q = quantize(W̃)  # Still ok

DeepSeek uses this for attention layers
```

---

## Mixed Precision Strategies

### Sensitive Layer Detection

```
Sensitivity analysis:

For each layer:
  1. Quantize layer to INT8
  2. Measure perplexity change
  3. If change > threshold: keep FP16
  
Results for DeepSeek-V3:
  - Most layers: <0.1% perplexity increase (INT8 ok)
  - First 2 layers: 0.5% increase (keep FP16)
  - Last 2 layers: 0.3% increase (keep FP16)
  - Normalization layers: 1.0% increase (keep FP32)

Final configuration:
  - Layers 1-2: FP16
  - Layers 3-59: INT8
  - Layer 60: FP16
  - All norms: FP32
  - Result: <0.2% total perplexity increase
```

### Dynamic Precision Selection

```
Runtime precision adjustment:

Based on input perplexity:
  if perplexity < 10:  # Easy input
    use INT8 for all layers
  elif perplexity < 50:  # Medium
    use INT8 for middle layers, FP16 for first/last
  else:  # Hard input
    use FP16 for all layers

Adaptive overhead: negligible
Quality improvement: 0.5-1.0% on difficult inputs
```

---

## Training Techniques

### FP8 Training Recipe

```
DeepSeek-V3 FP8 Training:

Initialization:
  - Master weights: FP32
  - Working weights: FP8-E4M3
  - Activations: FP8-E5M2
  - Gradients: FP8-E4M3

Training loop:
  for batch in data:
    # Forward
    w_fp8 = cast_fp8(master_weights)
    x_fp8 = cast_fp8(input, format='E5M2')
    
    y = forward(x_fp8, w_fp8)  # FP8 matmuls
    y_fp32 = cast_fp32(y)
    
    loss = criterion(y_fp32, target)  # FP32 loss
    
    # Backward
    grad = backward(loss)
    grad_fp8 = cast_fp8(grad, format='E4M3')
    
    # Accumulate in FP32
    grad_accum_fp32 += cast_fp32(grad_fp8)
    
    # Update (every N steps)
    if step % N == 0:
      optimizer.step(master_weights, grad_accum_fp32)
      grad_accum_fp32 = 0

Stability techniques:
  - Loss scaling: multiply loss by 1024 before backward
  - Gradient clipping: clip in FP32 before accumulation
  - Delayed updates: accumulate gradients over multiple steps
```

### Quantization-Aware Training

```
Train with quantization in the loop:

class QuantizedLinear(nn.Module):
    def forward(self, x):
        # Quantize weights
        w_q = quantize(self.weight)
        w_dq = dequantize(w_q)
        
        # Quantize activations
        x_q = quantize(x)
        x_dq = dequantize(x_q)
        
        # Compute with dequantized values
        y = F.linear(x_dq, w_dq)
        
        return y
    
    def backward(self, grad_output):
        # Straight-through estimator
        grad_weight = grad_output @ self.input.T
        grad_input = self.weight @ grad_output.T
        
        return grad_input, grad_weight

Benefits:
  - Model learns to be robust to quantization
  - Better accuracy than post-training quantization
  - Can use lower precision (INT4) with <1% loss
```

---

## Inference Optimization

### Fused Quantized Kernels

```
Optimized INT8 matmul kernel:

Instead of:
  x_fp16 → quantize → x_int8
  w_fp16 → quantize → w_int8
  y_int32 = matmul(x_int8, w_int8)
  y_fp16 = dequantize(y_int32)

Fused kernel:
  y_fp16 = quantized_matmul(x_fp16, w_int8, scales)
  
  Internal:
    - Quantize x on-the-fly
    - Use INT8 DP4A instructions (4 multiplies per cycle)
    - Accumulate in INT32
    - Dequantize and write FP16

Speedup: 2-4× vs FP16 on modern GPUs
```

### Batched Quantization

```
Optimize for throughput:

Batch quantization operations:
  For batch of 32 sequences:
    - Compute scales for all in parallel
    - Quantize all tensors together
    - Single kernel launch

vs Sequential:
  - 32 separate scale computations
  - 32 separate quantize operations
  - 32 kernel launches

Throughput improvement: 3-5×
```

---

## Detailed Examples

### Example 1: Weight Quantization

```python
import torch

# Original FP16 weight
W_fp16 = torch.tensor([
    [0.123, -0.456, 0.789],
    [-0.234, 0.567, -0.891]
], dtype=torch.float16)

print("Original weights (FP16):")
print(W_fp16)
print(f"Memory: {W_fp16.numel() * 2} bytes")

# Per-channel INT8 quantization
def quantize_per_channel(W):
    W_q = torch.zeros_like(W, dtype=torch.int8)
    scales = torch.zeros(W.shape[0], dtype=torch.float16)
    
    for i in range(W.shape[0]):
        # Compute scale
        s = W[i].abs().max() / 127
        scales[i] = s
        
        # Quantize
        W_q[i] = torch.round(W[i] / s).clamp(-128, 127).to(torch.int8)
    
    return W_q, scales

W_q, scales = quantize_per_channel(W_fp16)

print("\nQuantized weights (INT8):")
print(W_q)
print(f"Scales: {scales}")
print(f"Memory: {W_q.numel() + scales.numel() * 2} bytes")

# Dequantize
W_dq = W_q.float() * scales.unsqueeze(1)
print("\nDequantized weights:")
print(W_dq)

# Error
error = (W_fp16 - W_dq).abs().mean()
print(f"\nMean absolute error: {error:.6f}")
print(f"Relative error: {(error / W_fp16.abs().mean() * 100):.2f}%")
print(f"Memory reduction: {(1 - (W_q.numel() + scales.numel()*2)/(W_fp16.numel()*2))*100:.1f}%")
```

### Example 2: KV Cache Quantization

```python
# Simulate KV cache
batch_size = 1
seq_len = 100
latent_dim = 512

# FP16 KV cache
kv_cache_fp16 = torch.randn(batch_size, seq_len, latent_dim, dtype=torch.float16)

print("FP16 KV cache:")
print(f"  Shape: {kv_cache_fp16.shape}")
print(f"  Memory: {kv_cache_fp16.numel() * 2 / 1024:.2f} KB")

# Per-token INT8 quantization
def quantize_kv_cache(cache):
    cache_q = torch.zeros_like(cache, dtype=torch.int8)
    scales = torch.zeros(batch_size, seq_len, dtype=torch.float16)
    
    for b in range(batch_size):
        for t in range(seq_len):
            # Compute scale per token
            s = cache[b, t].abs().max() / 127
            scales[b, t] = s
            
            # Quantize
            cache_q[b, t] = torch.round(cache[b, t] / s).clamp(-128, 127).to(torch.int8)
    
    return cache_q, scales

kv_cache_q, kv_scales = quantize_kv_cache(kv_cache_fp16)

print("\nINT8 KV cache:")
print(f"  Shape: {kv_cache_q.shape}")
print(f"  Memory: {(kv_cache_q.numel() + kv_scales.numel() * 2) / 1024:.2f} KB")
print(f"  Reduction: {(1 - (kv_cache_q.numel() + kv_scales.numel()*2)/(kv_cache_fp16.numel()*2))*100:.1f}%")

# Dequantize
kv_cache_dq = kv_cache_q.float() * kv_scales.unsqueeze(-1)

# Error
error = (kv_cache_fp16 - kv_cache_dq).abs().mean()
print(f"\nQuantization error: {error:.6f}")
```

---

## Summary

### Key Techniques

```
DeepSeek Quantization Arsenal:

1. FP8 Training:
   - E4M3 for weights/gradients
   - E5M2 for activations
   - Dynamic per-tensor scaling
   - <0.1% quality loss

2. INT8 Post-Training:
   - Per-channel for weights
   - Per-token for KV cache
   - <0.5% quality loss

3. INT4 Aggressive:
   - Group quantization (size=128)
   - GPTQ for weights
   - ~1% quality loss
   - 4× memory reduction

4. Mixed Precision:
   - Sensitive layers in FP16
   - Most layers in INT8
   - Norms in FP32
   - Optimal quality/efficiency
```

### Performance Impact

```
DeepSeek-V3 (671B parameters):

Quantization    Memory    Speed     Quality Loss
────────────────────────────────────────────────
FP16 (baseline) 1.34 TB   1.0×      0%
FP8 (training)  671 GB    2.0×      <0.1%
INT8 (inference) 671 GB   2.5×      <0.5%
INT4 (aggressive) 335 GB  3.0×      ~1.0%

Real-world deployment:
  - Training: FP8 (2× faster, 2× less memory)
  - Serving: INT4 (fits on fewer GPUs)
  - Critical apps: INT8 (best quality/speed)
```

---
