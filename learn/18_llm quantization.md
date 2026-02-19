# LLM Quantization: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Fundamentals of Quantization](#fundamentals-of-quantization)
3. [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
4. [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
5. [Advanced Techniques](#advanced-techniques)
6. [GPTQ and AWQ](#gptq-and-awq)
7. [GGUF and llama.cpp](#gguf-and-llamacpp)
8. [Mixed Precision Quantization](#mixed-precision-quantization)
9. [Practical Implementation](#practical-implementation)
10. [Performance Analysis](#performance-analysis)

---

## Introduction

### What is Quantization?

**Quantization** is the process of reducing the numerical precision of model weights and activations from high-precision (e.g., FP32, FP16) to lower-precision (e.g., INT8, INT4).

### Why Quantize LLMs?

**Problem**: Large Language Models are huge
- GPT-3 175B: ~350GB (FP16)
- LLaMA 70B: ~140GB (FP16)
- LLaMA 7B: ~14GB (FP16)

**Memory too large for**:
- Consumer GPUs (most have 8-24GB)
- Mobile devices
- Edge deployment
- Cost-effective inference

**Solution**: Quantization
- 4-bit quantization: ~4× smaller
- LLaMA 70B: 140GB → 35GB (fits on single GPU!)
- LLaMA 7B: 14GB → 3.5GB (fits on laptop)

### Types of Quantization

| Type | When Applied | Complexity | Quality |
|------|-------------|------------|---------|
| **Post-Training Quantization (PTQ)** | After training | Low | Good |
| **Quantization-Aware Training (QAT)** | During training | High | Best |
| **Dynamic Quantization** | Runtime | Medium | Variable |

---

## Fundamentals of Quantization

### Numerical Precision Formats

**Floating Point Formats**:

| Format | Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|------|----------|----------|-------|-----------|
| **FP32** | 32 | 1 | 8 | 23 | ±1.4×10⁻⁴⁵ to ±3.4×10³⁸ | ~7 digits |
| **FP16** | 16 | 1 | 5 | 10 | ±5.96×10⁻⁸ to ±6.55×10⁴ | ~3 digits |
| **BF16** | 16 | 1 | 8 | 7 | Same as FP32 | ~2 digits |

**Integer Formats**:

| Format | Bits | Range | Values |
|--------|------|-------|--------|
| **INT8** | 8 | -128 to 127 | 256 |
| **INT4** | 4 | -8 to 7 | 16 |
| **INT2** | 2 | -2 to 1 | 4 |

### Quantization Formula

**Basic symmetric quantization**:

$$x_q = \text{round}\left(\frac{x}{s}\right)$$

$$\hat{x} = x_q \times s$$

where:
- $x$ is original FP32/FP16 value
- $s$ is **scale factor**
- $x_q$ is quantized integer
- $\hat{x}$ is dequantized (reconstructed) value

**Scale factor computation**:

$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

where $b$ is number of bits.

### Quantization Error

**Quantization error (reconstruction error)**:

$$\text{Error} = |x - \hat{x}| = |x - x_q \times s|$$

**Mean Squared Error** (typical metric):

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2$$

### Example: INT8 Quantization

**Original weights**: $W = [-0.5, 0.3, 1.2, -0.8, 0.6]$

**Step 1**: Find max absolute value
$$\max(|W|) = 1.2$$

**Step 2**: Compute scale
$$s = \frac{1.2}{127} = 0.00945$$

**Step 3**: Quantize
$$W_q = \text{round}\left(\frac{W}{s}\right) = [-53, 32, 127, -85, 63]$$

**Step 4**: Dequantize (verify)
$$\hat{W} = W_q \times s = [-0.501, 0.302, 1.200, -0.803, 0.595]$$

**Errors**:
$$E = [-0.001, 0.002, 0.000, -0.003, -0.005]$$

---

## Post-Training Quantization (PTQ)

### Symmetric Quantization

**Per-tensor quantization**:

$$W_q = \text{clip}\left(\text{round}\left(\frac{W}{s}\right), -128, 127\right)$$

$$s = \frac{\max(|W|)}{127}$$

**Advantages**:
- Simple
- Fast
- No zero-point needed

**Disadvantages**:
- Wastes range if distribution asymmetric
- One scale for entire tensor

### Asymmetric Quantization

**Formula**:

$$W_q = \text{clip}\left(\text{round}\left(\frac{W - z}{s}\right), 0, 255\right)$$

where:
- $s = \frac{\max(W) - \min(W)}{255}$ is scale
- $z = -\text{round}\left(\frac{\min(W)}{s}\right)$ is zero-point

**Dequantization**:

$$\hat{W} = s \times (W_q - z)$$

**Advantages**:
- Better for asymmetric distributions
- Uses full integer range

**Disadvantages**:
- More complex (need zero-point)
- Slightly slower

### Per-Channel Quantization

**Problem with per-tensor**: Different channels have different ranges.

**Example**:
```
Channel 0: [-0.1, 0.1]  → small range
Channel 1: [-2.0, 2.0]  → large range

Per-tensor scale: 2.0 / 127 = 0.0157
Channel 0 uses only [-6, 6] out of [-127, 127] → wasted precision!
```

**Solution: Per-channel quantization**

For weight matrix $W \in \mathbb{R}^{C_{out} \times C_{in}}$:

$$s_i = \frac{\max_j |W_{i,j}|}{127} \quad \text{for } i = 1, ..., C_{out}$$

Each output channel gets its own scale!

**Benefits**:
- Higher precision per channel
- Better preservation of model quality
- Minimal overhead (only $C_{out}$ scales)

### Calibration

**Problem**: Activations change with input data.

**Solution**: Calibration with representative dataset.

**Algorithm**:
```
1. Collect statistics on calibration set:
   - Min/max values of activations
   - Or: Histogram of activation distributions

2. Compute optimal scale factors:
   - Minimize quantization error
   - Or: Minimize KL-divergence from original

3. Apply quantization with computed scales
```

**Calibration methods**:

**Min-Max**:
$$s = \frac{\max(x) - \min(x)}{255}$$

**Percentile** (e.g., 99.9%):
$$s = \frac{\text{percentile}_{99.9}(|x|)}{127}$$
Clips outliers for better overall accuracy.

**MSE Minimization**:
$$s^* = \arg\min_s \mathbb{E}[(x - \text{quant}(x, s))^2]$$

---

## Quantization-Aware Training (QAT)

### Concept

**Idea**: Simulate quantization during training so model adapts.

**Training flow**:
```
Forward pass:
  Weights (FP32) → Fake Quantize → Quantized Weights (still FP32)
  
Backward pass:
  Gradients flow through (straight-through estimator)
  Update FP32 weights
```

### Straight-Through Estimator (STE)

**Problem**: Quantization function is non-differentiable.

$$\frac{\partial \text{round}(x)}{\partial x} = 0 \text{ almost everywhere}$$

**Solution**: Use identity in backward pass.

**Forward**:
$$y = \text{round}(x)$$

**Backward**:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times 1$$

### QAT Algorithm

```python
def qat_forward(weight, scale):
    # Quantize
    weight_q = round(weight / scale)
    weight_q = clip(weight_q, -128, 127)
    
    # Dequantize (still in FP32 for training)
    weight_fake_quant = weight_q * scale
    
    return weight_fake_quant

def qat_backward(grad):
    # Straight-through estimator
    return grad  # Identity function
```

### Benefits of QAT

**Comparison**:

| Method | Accuracy Loss | Training Time | Complexity |
|--------|---------------|---------------|------------|
| PTQ | 2-5% | None | Low |
| QAT | 0.5-1% | +20-30% | Medium |

**When to use QAT**:
- Critical applications requiring minimal accuracy loss
- Have compute budget for retraining
- Targeting very low precision (4-bit, 2-bit)

---

## Advanced Techniques

### GPTQ (Accurate Post-Training Quantization)

**Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)

**Key Idea**: Use second-order information (Hessian) to minimize quantization error.

**Algorithm**:

For each layer weight matrix $W$:

1. **Compute Hessian** (approximated):
   $$H = 2XX^T$$
   where $X$ is layer input from calibration data.

2. **Cholesky decomposition**:
   $$H = LL^T$$

3. **Iterative quantization**:
   For each column $i$:
   ```
   a. Quantize column i
   b. Compute error
   c. Update remaining columns to compensate for error
   ```

**Compensation formula**:

$$W_{:,j} \leftarrow W_{:,j} - \frac{w_i - \text{quant}(w_i)}{H_{ii}}H_{:,i} \quad \text{for } j > i$$

**Benefits**:
- 3-4 bit quantization with <1% perplexity loss
- No retraining needed
- Layer-wise processing (memory efficient)

**GPTQ Results**:

| Model | Bits | Perplexity (original) | Perplexity (GPTQ) | Loss |
|-------|------|----------------------|-------------------|------|
| LLaMA 7B | 16 | 5.68 | 5.68 | 0% |
| LLaMA 7B | 4 | 5.68 | 5.78 | +1.8% |
| LLaMA 7B | 3 | 5.68 | 6.12 | +7.7% |

### AWQ (Activation-aware Weight Quantization)

**Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)

**Key Insight**: Not all weights are equally important!

**Observation**:
- 1% of weights contribute to 50% of activation magnitude
- These "salient" weights need higher precision

**Algorithm**:

1. **Identify salient weights**:
   $$s_i = \frac{1}{n}\sum_{x \in \text{calib}} |w_i \cdot x|$$
   
   Weights with high $s_i$ are salient.

2. **Per-channel scaling**:
   $$W^{(s)} = W \times \text{diag}(s)$$
   $$X^{(s)} = X / s$$
   
   Scale weights up, activations down (mathematically equivalent).

3. **Quantize scaled weights**:
   $$W_q = \text{quant}(W^{(s)})$$

4. **Store scales** for runtime.

**Benefit**: Salient weights get more quantization range.

**AWQ Results**:

| Model | Method | Bits | WikiText-2 PPL |
|-------|--------|------|----------------|
| LLaMA-7B | FP16 | 16 | 5.68 |
| LLaMA-7B | RTN | 4 | 7.89 |
| LLaMA-7B | GPTQ | 4 | 5.78 |
| LLaMA-7B | **AWQ** | 4 | **5.71** |

AWQ achieves near-FP16 quality at 4-bit!

### SmoothQuant

**Paper**: "SmoothQuant: Accurate and Efficient Post-Training Quantization" (2023)

**Problem**: Activations have outliers that hurt quantization.

**Example activation distribution**:
```
Most values: [-1, 1]
Outliers: [-50, 50]

Scale = 50/127 = 0.39
Most values use only [-3, 3] out of [-127, 127] → bad precision!
```

**Solution**: Migrate difficulty from activations to weights.

**Mathematical smoothing**:

$$Y = XW = (X \text{diag}(s))(W \text{diag}(s^{-1})) = X'W'$$

where $s$ is per-channel smoothing factor.

**Smoothing factor**:
$$s_j = \max(|X_{:,j}|)^\alpha / \max(|W_{j,:}|)^{1-\alpha}$$

Typical: $\alpha = 0.5$ (geometric mean)

**Effect**:
- Reduces activation outliers
- Slightly increases weight range (but weights easier to quantize)
- Net improvement in quantization quality

---

## GGUF and llama.cpp

### GGUF Format

**GGUF** (GPT-Generated Unified Format) is a file format for quantized models.

**Design**:
- Self-contained (metadata + weights)
- Multiple quantization types in one file
- Optimized for CPU inference
- Memory-mapped loading

### Quantization Types in GGUF

| Type | Bits/weight | Description |
|------|-------------|-------------|
| **Q4_0** | 4.5 | 4-bit, 32-weight blocks |
| **Q4_1** | 5.0 | 4-bit with min/max per block |
| **Q5_0** | 5.5 | 5-bit, 32-weight blocks |
| **Q5_1** | 6.0 | 5-bit with min/max |
| **Q8_0** | 8.5 | 8-bit |
| **Q2_K** | 2.5-3.5 | 2-bit, k-quant (mixed) |
| **Q3_K** | 3.0-4.0 | 3-bit, k-quant |
| **Q4_K** | 4.0-5.0 | 4-bit, k-quant |
| **Q5_K** | 5.0-6.0 | 5-bit, k-quant |
| **Q6_K** | 6.0-7.0 | 6-bit, k-quant |

### K-Quant (Mixed Precision)

**Idea**: Different parts of same tensor at different precision.

**Structure** (Q4_K example):
```
Tensor divided into blocks of 256 weights:
  - First 32 weights: 6-bit (important)
  - Next 224 weights: 4-bit (normal)
  - Scales: FP16
```

**Automatically determines** which weights need higher precision.

### Block Quantization

**Q4_0 format** (most popular):

```
Block size: 32 weights
Per block:
  - 1 FP16 scale (2 bytes)
  - 32 × 4-bit weights (16 bytes)
  Total: 18 bytes for 32 weights

Effective: 18/32 = 4.5 bits per weight
```

**Quantization**:

$$\text{scale} = \frac{\max(|\text{block}|)}{8}$$
$$w_q = \text{round}\left(\frac{w}{\text{scale}}\right)$$
$$w_q = \text{clip}(w_q, -8, 7)$$

**Dequantization**:

$$w = w_q \times \text{scale}$$

### Quality Comparison

**LLaMA-7B on WikiText-2**:

| Format | Size | Perplexity | Speed (tok/s) |
|--------|------|------------|---------------|
| FP16 | 13.0 GB | 5.68 | 15 |
| Q8_0 | 6.7 GB | 5.69 | 28 |
| Q6_K | 5.2 GB | 5.72 | 32 |
| Q5_K_M | 4.4 GB | 5.75 | 35 |
| Q4_K_M | 3.8 GB | 5.83 | 40 |
| Q3_K_M | 3.0 GB | 6.15 | 45 |
| Q2_K | 2.4 GB | 7.89 | 50 |

**Sweet spot**: Q4_K_M (3.8GB, 5.83 PPL, 40 tok/s)

---

## Mixed Precision Quantization

### Motivation

**Not all layers are equally sensitive**:

```
Attention layers:     More sensitive (use 8-bit)
FFN layers:           Less sensitive (use 4-bit)
First/last layers:    Most sensitive (keep FP16)
Embedding layer:      Can be 4-bit
```

### Layer-wise Precision Assignment

**Manual assignment**:
```python
precision_map = {
    'model.embed_tokens': 16,
    'model.layers.0': 16,        # First layer
    'model.layers.1-30.attn': 8,  # Attention
    'model.layers.1-30.ffn': 4,   # FFN
    'model.layers.31': 16,        # Last layer
    'lm_head': 16
}
```

**Automatic assignment** (sensitivity-based):

1. Measure sensitivity:
   $$\text{Sensitivity}_i = \frac{\Delta \text{Loss}_i}{\Delta w_i}$$

2. Sort layers by sensitivity

3. Assign precision:
   - Top 10%: 16-bit
   - Next 30%: 8-bit
   - Remaining: 4-bit

### Mixed Precision Benefits

**Example (LLaMA-7B)**:

| Configuration | Avg Bits | Size | Perplexity |
|--------------|----------|------|------------|
| All 4-bit | 4.0 | 3.8 GB | 5.95 |
| Mixed (auto) | 5.2 | 4.5 GB | 5.73 |
| All 8-bit | 8.0 | 7.0 GB | 5.69 |

**18% more memory for 0.22 PPL improvement** (worthwhile!)

---

## Practical Implementation

### PyTorch Quantization API

**Dynamic Quantization** (easiest):

```python
import torch

# Original model
model = MyLLM()

# Dynamically quantize (activations at runtime)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8    # Target dtype
)
```

**Static Quantization** (requires calibration):

```python
# Prepare model
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibrate
for batch in calibration_data:
    model_prepared(batch)

# Convert
model_quantized = torch.quantization.convert(model_prepared)
```

### bitsandbytes Library

**4-bit quantization with NormalFloat (NF4)**:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_use_double_quant=True,      # Quantize scales too
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Features**:
- NF4: Quantization scheme optimized for normal distributions
- Double quantization: Quantize the scales themselves
- Compute in BF16: Maintain quality for matrix multiplications

### GPTQ-for-LLaMa

**Quantize a model**:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)

# Quantize with calibration data
model.quantize(calibration_data)

# Save
model.save_quantized("llama-2-7b-gptq-4bit")
```

### AWQ Quantization

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4
}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data
)

# Save
model.save_quantized("llama-2-7b-awq-4bit")
```

---

## Performance Analysis

### Memory Savings

**Formula**:

$$\text{Memory} = \text{Parameters} \times \frac{\text{Bits}}{8}$$

**Example (LLaMA-7B, 7 billion parameters)**:

| Precision | Bits | Memory |
|-----------|------|--------|
| FP32 | 32 | 28 GB |
| FP16 | 16 | 14 GB |
| INT8 | 8 | 7 GB |
| INT4 | 4 | 3.5 GB |
| INT2 | 2 | 1.75 GB |

**With overhead** (scales, KV cache):

| Precision | Actual Size |
|-----------|-------------|
| FP16 | ~14 GB |
| INT8 | ~8 GB |
| INT4 | ~4.5 GB |

### Speed Improvements

**Inference speedup** (relative to FP16):

| Precision | CPU | GPU (CUDA) |
|-----------|-----|------------|
| FP16 | 1.0× | 1.0× |
| INT8 | 2-3× | 1.5-2× |
| INT4 | 4-6× | 2-3× |

**Note**: GPU speedup less due to memory bandwidth still being bottleneck.

### Accuracy Trade-offs

**Typical perplexity increases**:

| Model Size | 8-bit | 4-bit | 3-bit | 2-bit |
|------------|-------|-------|-------|-------|
| 7B | +0.1% | +1-3% | +5-10% | +20-40% |
| 13B | +0.05% | +0.5-2% | +3-7% | +15-30% |
| 70B | +0.02% | +0.3-1% | +2-5% | +10-20% |

**Larger models are more robust to quantization!**

### Benchmark Results

**LLaMA-2-7B on various benchmarks**:

| Method | Bits | MMLU | HellaSwag | ARC-C | Avg |
|--------|------|------|-----------|-------|-----|
| FP16 | 16 | 45.3 | 77.2 | 44.4 | 55.6 |
| RTN | 4 | 43.1 | 73.8 | 41.2 | 52.7 |
| GPTQ | 4 | 44.8 | 76.5 | 43.9 | 55.1 |
| AWQ | 4 | 45.0 | 76.9 | 44.1 | 55.3 |
| NF4 | 4 | 44.5 | 76.1 | 43.5 | 54.7 |

**AWQ and GPTQ are very close to FP16!**

---

## Best Practices

### Choosing Quantization Method

**Decision tree**:

```
Do you need to retrain?
├─ No → Post-Training Quantization
│  ├─ Target 8-bit → Simple PTQ (per-channel)
│  ├─ Target 4-bit, quality critical → GPTQ or AWQ
│  └─ Target 4-bit, speed critical → RTN (Round-to-Nearest)
└─ Yes → Quantization-Aware Training
   ├─ Resources available → Full QAT
   └─ Limited resources → LoRA + QAT
```

### Recommended Configurations

**For different use cases**:

**1. Maximum Quality** (minimal accuracy loss):
- Method: AWQ or GPTQ
- Bits: 4-bit (group size 128)
- Size: ~4.5 GB (7B model)
- Loss: <1% perplexity

**2. Balanced** (good quality, good speed):
- Method: Q4_K_M (GGUF)
- Bits: 4-bit mixed
- Size: ~3.8 GB (7B model)
- Loss: ~2-3% perplexity

**3. Maximum Compression** (for edge devices):
- Method: Q2_K or Q3_K_S
- Bits: 2-3 bit
- Size: ~2.4 GB (7B model)
- Loss: ~10-20% perplexity

**4. Production Inference**:
- Method: INT8 (bitsandbytes or GPTQ)
- Bits: 8-bit
- Size: ~7 GB (7B model)
- Loss: <0.5% perplexity

### Common Pitfalls

**1. Not calibrating properly**:
- Use diverse calibration data (1000+ samples)
- Match calibration distribution to inference distribution

**2. Quantizing everything**:
- Keep first/last layers at higher precision
- Keep embeddings at reasonable precision

**3. Ignoring runtime**:
- 4-bit faster on CPU than GPU (limited kernels)
- Some quantization schemes require special hardware

**4. Over-quantizing small models**:
- 7B models more sensitive than 70B
- Consider 8-bit for critical applications with small models

---

## Future Directions

### Emerging Techniques

**1. Sub-4-bit Quantization**:
- 3-bit, 2-bit, even 1-bit models
- Requires specialized methods (SpQR, QuIP)

**2. Learned Quantization**:
- Learn quantization parameters end-to-end
- Vector quantization

**3. Dynamic Bit Allocation**:
- Allocate bits based on importance
- Automatically find optimal bit-width per layer

**4. Hardware-Aware Quantization**:
- Optimize for specific accelerators
- Co-design software and hardware

### Open Problems

1. **Maintaining quality at <4 bits** for small models
2. **Efficient kernels** for mixed-precision inference
3. **Quantizing attention** (KV cache quantization)
4. **Long-context quantization** (128K+ tokens)

---

## Key Takeaways

### Summary Table

| Aspect | 8-bit | 4-bit | 2-bit |
|--------|-------|-------|-------|
| **Size Reduction** | 2× | 4× | 8× |
| **Speed Increase** | 2× | 3× | 4× |
| **Accuracy Loss** | <0.5% | 1-3% | 10-30% |
| **Use Case** | Production | Consumer devices | Extreme edge |
| **Best Method** | Simple PTQ | GPTQ/AWQ | Specialized |

### Practical Recommendations

1. **Start with 4-bit GPTQ or AWQ** for most applications
2. **Use GGUF/llama.cpp** for CPU inference
3. **Keep 8-bit for critical applications** requiring minimal loss
4. **Calibrate with diverse data** (1000+ samples)
5. **Test on your specific use case** - benchmarks don't tell everything

### Quick Reference

**Need maximum quality?** → 4-bit AWQ
**Need maximum speed?** → Q2_K or Q3_K (GGUF)
**Need balanced?** → 4-bit GPTQ or Q4_K_M
**Running on GPU?** → bitsandbytes (NF4)
**Running on CPU?** → GGUF (llama.cpp)

---

## References

1. **GPTQ**: Frantar et al. (2023), "GPTQ: Accurate Post-Training Quantization for GPT"
2. **AWQ**: Lin et al. (2023), "AWQ: Activation-aware Weight Quantization"
3. **SmoothQuant**: Xiao et al. (2023), "SmoothQuant: Accurate and Efficient Post-Training Quantization"
4. **GGUF/llama.cpp**: Gerganov et al. (2023), GitHub repository
5. **bitsandbytes**: Dettmers et al. (2022), "LLM.int8(): 8-bit Matrix Multiplication"

---

