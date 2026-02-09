# DeepSeek LLM: Complete End-to-End Basics (with Mathematics)

## 1. What is DeepSeek?

DeepSeek (DeepSeek AI) is a Chinese AI lab that develops **open-source**, **extremely efficient**, and **high-performance** Large Language Models (LLMs).

**Signature Philosophy**:
- Massive total parameters, but **very sparse activation** (Mixture-of-Experts)
- State-of-the-art performance in **Math, Coding, and Reasoning**
- Dramatically lower training & inference cost than dense models (Llama, Qwen, GPT)

### Major Models (2024–2025)
| Model              | Total Params | Active Params/Token | KV Cache Compression | Context | Release     |
|--------------------|--------------|----------------------|-----------------------|---------|-------------|
| DeepSeek-V2        | 236B        | 21B                 | ~8–10× (MLA)          | 128K    | May 2024    |
| DeepSeek-V3        | 671B        | 37B                 | ~8–10× (MLA)          | 128K+   | Dec 2024    |
| DeepSeek-R1        | 671B        | 37B                 | Same as V3            | 128K    | Jan 2025    |

## 2. Base Architecture (Decoder-Only Transformer)

DeepSeek models are **decoder-only** Transformers with these standard components:

- RMSNorm
- RoPE (Rotary Position Embeddings)
- SwiGLU activation in FFN
- Parallel attention + FFN (like Llama 3)

The **revolutionary changes** are in two places:
1. **Attention** → Multi-Head Latent Attention (MLA)
2. **FFN** → DeepSeekMoE

## 3. Innovation 1: DeepSeekMoE (Mixture of Experts)

### Standard MoE vs DeepSeekMoE

**Standard MoE (Mixtral, DeepSeek-V1 style)**:
- 8 large experts
- Top-2 routing

**DeepSeekMoE (V2 → V3)**:
- Extremely **fine-grained** experts
- Shared experts + routed experts

### DeepSeek-V3 MoE Configuration
- 256 routed experts
- 1 shared expert (always active)
- Activate **Top-8 routed experts** + 1 shared → **Total active FFN params ~37B**

### Mathematical Formulation

For a token hidden state \(\mathbf{h}_t \in \mathbb{R}^d\):

1. **Routing logits**:
   \[
   \mathbf{g} = \mathbf{h}_t \mathbf{W}_g \in \mathbb{R}^{N_e} \quad (N_e = 256)
   \]

2. **Top-K selection** (K=8):
   \[
   \text{Top-8 indices} = \arg\max_k (\mathbf{g})
   \]

3. **Expert output** (each expert is SwiGLU FFN):
   \[
   \mathbf{E}_i(\mathbf{x}) = \mathbf{W}_2 (\sigma(\mathbf{W}_1 \mathbf{x}) \odot \mathbf{W}_3 \mathbf{x})
   \]

4. **Final MoE output**:
   \[
   \text{MoE}(\mathbf{h}_t) = \sum_{i \in \text{Top-8}} g_i \cdot \mathbf{E}_i(\mathbf{h}_t) + \mathbf{E}_{\text{shared}}(\mathbf{h}_t)
   \]

### V3 Innovation: Auxiliary-Loss-Free Load Balancing
Traditional MoE uses an auxiliary loss:
\[
\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot p_i
\]
DeepSeek-V3 **removes** this loss entirely and instead uses:
- Biased routing initialization
- Sequence-level balancing
- Expert-level capacity factors

Result: Better expert specialization + no extra training overhead.

## 4. Innovation 2: Multi-Head Latent Attention (MLA)

This is **the most important efficiency breakthrough**.

### Problem with Standard MHA
KV cache size = `2 × num_layers × seq_len × num_heads × head_dim × bytes`

For V3 (128K context):
→ ~100+ GB KV cache in BF16 → impossible on consumer GPUs.

### MLA Solution: Low-Rank Joint Compression

Instead of caching full **K** and **V** per head, cache **one single latent vector** per token.

#### MLA Equations (per layer)

Let:
- \(d =\) hidden dim (7168 in V3)
- \(d_c =\) compression dim (**512** in V3)
- \(n_h =\) heads (128)
- \(d_h =\) head dim (128)

For token \(\mathbf{h}_t\):

```python
# 1. Compress Q, K, V jointly into latent
c_t^{KV} = h_t @ W^{DKV}          # (d,) → (d_c,)     [Down projection]

# 2. Decompress to full K and V
K_t^C = c_t^{KV} @ W^{UK}         # (d_c,) → (d,)
V_t^C = c_t^{KV} @ W^{UV}         # (d_c,) → (d,)

# 3. Q is NOT compressed (remains full)
Q_t = h_t @ W^Q                   # (d,) → (d,)
