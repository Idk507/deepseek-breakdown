# DeepSeek Basics: Complete End-to-End Explanation (2026 Edition)

Here's a detailed, up-to-date, math-heavy guide covering **DeepSeek-V3** (671B MoE) + **DeepSeek-R1** (reasoning model).

## 1. DeepSeek Model Evolution

- **DeepSeek-V1/V2** (2024): Introduced **DeepSeekMoE** + **MLA**
- **DeepSeek-V3** (Dec 2024): 671B total, ~37B active params → SOTA open model (beats Llama-3.1-405B on most benchmarks)
- **DeepSeek-R1** (Jan 2025): Reasoning model based on V3-Base + GRPO RL

## 2. Core Architecture Innovations

### A. Multi-head Latent Attention (MLA) – The KV Cache Killer

**Problem**: Standard MHA → KV Cache = 2 × seq_len × n_heads × d_head × bytes (huge for 128k context)

**MLA Solution**: Low-rank joint compression of K & V.

**Exact Equations (DeepSeek-V3)**:

Input hidden: $ h_t \in \mathbb{R}^d $ (d = 7168)

- KV Latent Compression:
  $$
  c_{KV}^t = W_{DKV} h^t \in \mathbb{R}^{d_c}, \quad d_c = 512
  $$

- Compressed K/V:
  $$
  k_C^t = W_{UK} c_{KV}^t \in \mathbb{R}^{n_h \cdot d_h}
  $$
  $$
  v_C^t = W_{UV} c_{KV}^t \in \mathbb{R}^{n_h \cdot d_h}
  $$
  (W_UK, W_UV: up-projection)

- Decoupled RoPE part:
  $$
  k_R^t = \text{RoPE}(W_{KR} h^t), \quad d_{Rh} = 64
  $$
  Final head key: $ k_i^t = [k_C^t_i ; k_R^t] $

- Query (slightly different):
  $$
  c_Q^t \in \mathbb{R}^{d'_c=1536}, \quad q_C^t = W_{UQ} c_Q^t
  $$
  $$
  q_R^t = \text{RoPE}(W_{QR} c_Q^t)
  $$

**Attention per head**:
$$
o_i^t = \sum_j \text{softmax}\left( \frac{q_i^{t\top} k_i^j }{\sqrt{d_h + d_{Rh}}} \right) v_C^j_i
$$

**KV Cache Only Stores**:
- $ c_{KV}^t $ (512 dim)
- $ k_R^t $ (64 dim per head)

→ **~20-30× smaller KV cache** than MHA/GQA. Huge win for long context.

### B. DeepSeekMoE (Ultimate Expert Specialization)

**Standard MoE** (Mixtral/Switch): N experts, top-k routing → experts become redundant.

**DeepSeekMoE Key Ideas**:

1. **Fine-grained Segmentation**:
   - Split each expert into `m` smaller experts (m=8 typical)
   - Total routed experts = 256 instead of 32
   - More combinatorial flexibility

2. **Shared Experts**:
   - 1 shared expert always activated (captures common knowledge)
   - Routed experts: 256, activate **Top-8**

**Per Layer**:
- Total experts: 257 (1 shared + 256 routed)
- Activated params: 1 shared + 8 routed
- Expert FFN intermediate size: **2048** (much smaller than dense 4×d)

**Routing**:
$$
s_{i,t} = \text{softmax}_i (u_t^\top e_i)
$$
Top-K routing with auxiliary-loss-free balancing (V3 innovation)

### C. V3 Load Balancing (Auxiliary-Loss-Free)

Instead of noisy auxiliary loss:

- Add **learnable bias** $ b_i $ to each routed expert
- Dynamically: increase bias if underused, decrease if overused
- Small sequence-wise balance loss ($ \alpha = 0.0001 $)

This leads to near-perfect expert utilization without hurting performance.

## 3. DeepSeek-V3 Hyperparameters (Key Ones)

- Hidden dim: 7168
- Layers: 61 (first 3 dense, rest MoE)
- Heads: 128 (head dim = 128)
- Vocab: 129,280
- Total params: **671B**
- Active params: **~37B**
- MoE intermediate: 2048
- Context: 128K (post-training)

## 4. Training Pipeline (End-to-End)

**Pre-training**:
- 14.8T tokens
- FP8 mixed precision
- Multi-Token Prediction (MTP) loss (next 2 tokens) with weight 0.1-0.3

**Post-training**:
- SFT + Length extrapolation
- GRPO RL (for R1)

## 5. DeepSeek-R1 & GRPO

**GRPO Objective** (Group Relative Policy Optimization):
$$
J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( r_i \cdot \frac{\pi_\theta}{\pi_{old}}, \ \text{clip}(...) \cdot r_i \right) - \beta D_{KL} \right]
$$
where advantage:
$$
A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

**Key Features**:
- No critic network (unlike PPO)
- Group size G=16
- Rule-based verifiable rewards only (no RM for reasoning stage)
- Cold-start data → RL (R1-Zero) → Multi-stage RL (R1)

This leads to **emergent reasoning** (self-reflection, verification, long CoT).

## 6. Performance Summary (V3)

- Outperforms Llama-3.1-405B, Qwen2.5-72B, GPT-4o-0513 on many evals
- Extremely cheap inference (~1/5th of Llama-405B)

