### Introduction to DeepSeek

DeepSeek is a family of open-source large language models (LLMs) developed by DeepSeek AI, a Chinese AI research company. The series includes models like DeepSeek-V2, DeepSeek-V3, DeepSeek-Coder, DeepSeek-Math, and DeepSeek-R1, which are designed for tasks such as natural language processing, coding, mathematical reasoning, and complex problem-solving. These models are built on the Transformer architecture but incorporate innovations for efficiency, scalability, and performance, particularly in math and code. DeepSeek-V3, for example, is a Mixture-of-Experts (MoE) model with 671 billion total parameters, activating only 37 billion per token, making it cost-effective for training and inference. This allows it to rival models like GPT-4o in benchmarks while being cheaper to run.

The "basics" of DeepSeek refer to its core architecture, training process, and inference mechanism. We'll cover this end-to-end: from input processing, through the model's layers, to output generation, with mathematical details.

### Basic Transformer Architecture in DeepSeek

DeepSeek models are fundamentally based on the Transformer architecture introduced in Vaswani et al. (2017). Transformers process sequences (e.g., text tokens) using self-attention and feed-forward networks (FFNs), without recurrence, enabling parallel computation.

#### Key Components
1. **Input Embedding**: Text is tokenized into a sequence of tokens \( t_1, t_2, \dots, t_T \). Each token is embedded into a vector \( \mathbf{e}_i \in \mathbb{R}^d \), where \( d \) is the model dimension (e.g., 4096 in smaller models). Positional encodings (e.g., Rotary Position Embeddings, RoPE) are added to capture order:
   \[
   \mathbf{h}_0^i = \mathbf{e}_i + \operatorname{RoPE}(\mathbf{e}_i)
   \]
   RoPE applies rotations to embeddings based on position, preserving relative distances.

2. **Stacked Layers**: The model has \( L \) layers (e.g., 60 in DeepSeek-V3). Each layer consists of:
   - **Self-Attention**: Computes relationships between tokens.
   - **Feed-Forward Network (FFN)**: Applies non-linear transformations.
   Layers use residual connections and normalization (e.g., RMSNorm) for stability:
   \[
   \mathbf{h}_{l+1} = \operatorname{LayerNorm}(\mathbf{h}_l + \operatorname{FFN}(\operatorname{LayerNorm}(\mathbf{h}_l + \operatorname{Attention}(\mathbf{h}_l))))
   \]

#### Self-Attention Mechanism with Math
Self-attention allows each token to attend to others. For input hidden states \( \mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_T] \in \mathbb{R}^{T \times d} \):

- Project to Queries (Q), Keys (K), Values (V):
  \[
  \mathbf{Q} = \mathbf{H} W^Q, \quad \mathbf{K} = \mathbf{H} W^K, \quad \mathbf{V} = \mathbf{H} W^V
  \]
  where \( W^Q, W^K, W^V \in \mathbb{R}^{d \times d} \).

- Multi-Head Attention (MHA) splits into \( n_h \) heads (e.g., 32), each with dimension \( d_h = d / n_h \):
  For head \( i \):
  \[
  \mathbf{A}_i = \operatorname{Softmax}\left( \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_h}} \right) \mathbf{V}_i
  \]
  Concatenate and project:
  \[
  \operatorname{MHA}(\mathbf{H}) = \operatorname{Concat}(\mathbf{A}_1, \dots, \mathbf{A}_{n_h}) W^O
  \]
  where \( W^O \in \mathbb{R}^{d \times d} \).

This captures dependencies, but standard MHA is memory-intensive for long contexts due to the KV cache (storing K and V for all past tokens).

#### Feed-Forward Network (FFN)
A simple MLP per token:
\[
\operatorname{FFN}(\mathbf{x}) = W_2 \cdot \sigma(W_1 \mathbf{x})
\]
where \( \sigma \) is an activation like SwiGLU (a variant of GLU), and dimensions expand to \( d_{ff} \) (e.g., 4x \( d \)).

### Innovations in DeepSeek: Efficiency and Specialization

DeepSeek improves the Transformer with three key innovations: Multi-Head Latent Attention (MLA), DeepSeekMoE, and Multi-Token Prediction (MTP). These make it efficient for long contexts (up to 128K tokens) and specialized tasks like math.

#### 1. Multi-Head Latent Attention (MLA)
MLA compresses the KV cache to reduce memory (critical for inference). Instead of storing full K/V (\( O(T d) \) memory), it uses latent compression.

- **KV Compression** for token \( t \):
  \[
  \mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c} \quad (d_c \ll d)
  \]
  \[
  \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}
  \]
  \[
  \mathbf{k}_t^R = \operatorname{RoPE}(W^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_h^R}
  \]
  Per head \( i \):
  \[
  \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R], \quad \mathbf{v}_{t,i} = [\mathbf{v}_{t,i}^C; \mathbf{v}_t^R]
  \]
  Cache only \( \mathbf{c}_t^{KV} \) and \( \mathbf{k}_t^R \), reducing memory by ~5x compared to standard MHA.

- **Query Compression**:
  Similar process for Q:
  \[
  \mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \in \mathbb{R}^{d_c'}
  \]
  \[
  \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q, \quad \mathbf{q}_t^R = \operatorname{RoPE}(W^{QR} \mathbf{c}_t^Q)
  \]
  \[
  \mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]
  \]

- **Attention Computation**:
  \[
  \mathbf{o}_{t,i} = \sum_{j=1}^t \operatorname{Softmax}_j \left( \frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}} \right) \mathbf{v}_{j,i}^C
  \]
  \[
  \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \dots; \mathbf{o}_{t,n_h}]
  \]
  This maintains performance while enabling faster inference on long sequences.

#### 2. DeepSeekMoE (Mixture-of-Experts)
MoE replaces dense FFNs with sparse experts, activating only a subset per token for efficiency. DeepSeekMoE uses finer-grained experts: shared (\( N_s \)) and routed (\( N_r \), e.g., 512 total, activating \( K_r = 32 \)).

- **FFN with Routing**:
  Input \( \mathbf{u}_t \) after attention:
  \[
  \mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \operatorname{FFN}^{(s)}_i(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \operatorname{FFN}^{(r)}_i(\mathbf{u}_t)
  \]
  Routing gates:
  \[
  s_{i,t} = \operatorname{Sigmoid}(\mathbf{u}_t^T \mathbf{e}_i) \quad (\mathbf{e}_i: \text{expert centroid})
  \]
  \[
  g'_{i,t} = \begin{cases} s_{i,t}, & \text{if } s_{i,t} \in \operatorname{TopK}(\{s_{j,t}\}_{j=1}^{N_r}, K_r) \\ 0, & \text{otherwise} \end{cases}
  \]
  \[
  g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}
  \]

- **Auxiliary-Loss-Free Load Balancing**: Adds learnable bias \( b_i \) to routing to prevent overload:
  \[
  g'_{i,t} = \begin{cases} s_{i,t}, & \text{if } s_{i,t} + b_i \in \operatorname{TopK}(\{s_{j,t} + b_j\}, K_r) \\ 0, & \text{otherwise} \end{cases}
  \]
  Bias updated dynamically (e.g., decrease by \( \gamma = 0.001 \) if expert overloaded). A small balance loss \( \mathcal{L}_{\mathrm{Bal}} = \alpha \sum_i f_i P_i \) (with \( \alpha = 0.0001 \)) ensures sequence-wise fairness, where \( f_i \) is activation frequency and \( P_i \) is average affinity.

This allows expert specialization (e.g., some for math, others for code) without traditional auxiliary losses, reducing training cost.

- **Node-Limited Routing**: Limits tokens to \( M=4 \) GPU nodes, optimizing distributed training.

#### 3. Multi-Token Prediction (MTP)
Predicts multiple future tokens (e.g., next 2-4) to speed up generation. For depth \( D=1 \), predict up to \( k=4 \):

- Module:
  \[
  \mathbf{h}_{i}^{\prime k} = M_k [\operatorname{RMSNorm}(\mathbf{h}_i^{k-1}); \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]
  \]
  \[
  \mathbf{h}_{1:T-k}^k = \operatorname{TRM}_k(\mathbf{h}_{1:T-k}^{\prime k})
  \]
  \[
  P_{i+k+1}^k = \operatorname{OutHead}(\mathbf{h}_i^k)
  \]

- Loss:
  \[
  \mathcal{L}_{\text{MTP}}^k = \operatorname{CrossEntropy}(P_{2+k:T+1}^k, t_{2+k:T+1})
  \]
  \[
  \mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^D \mathcal{L}_{\text{MTP}}^k \quad (\lambda=0.3 \to 0.1)
  \]

Combined with speculative decoding, this boosts throughput by 1.8x with high acceptance rates.

### Training Pipeline

DeepSeek's training is end-to-end: pretraining, context extension, post-training.

1. **Pretraining**: On 14.8 trillion tokens (text, code, math). Uses AdamW optimizer, global batch size up to 15,360. Mixed precision (FP8) with quantization. Total: ~2.66M GPU hours on H800s. Objective: Next-token prediction + MTP loss.

2. **Context Extension**: Two phases to extend from 4K to 128K tokens using YaRN (dynamic RoPE scaling). Maintains perplexity.

3. **Post-Training**:
   - **Supervised Fine-Tuning (SFT)**: On 1.5M high-quality instances (math, code, reasoning). Loss: Cross-entropy on responses.
   - **Reinforcement Learning (RL)**: Uses Group Relative Policy Optimization (GRPO) for alignment. GRPO improves on PPO by grouping preferences:
     \[
     \mathcal{L}_{\text{GRPO}} = -\mathbb{E} \left[ \log \sigma \left( \frac{1}{\tau} (r(y_w) - r(y_l) - \beta \log \frac{\pi(y_w)}{\pi_0(y_w)} + \beta \log \frac{\pi(y_l)}{\pi_0(y_l)}) \right) \right]
     \]
     where \( r \) is reward, \( y_w/y_l \) winning/losing responses, \( \tau \) temperature, \( \beta \) KL penalty, \( \pi/\pi_0 \) policy/reference.
   - Hybrid Reward Model (RM): Rule-based for math/code + model-based for general.
   - Distillation from DeepSeek-R1 (reasoning model) boosts math/code scores (e.g., GSM8K: 74.0%, MATH: 39.8%).

For math-specific models like DeepSeek-Math, additional pretraining on 120B math tokens, then RL with verifiers.

### End-to-End Process: From Input to Output

1. **Input**: User query tokenized into sequence \( t_1, \dots, t_T \). Embedded with RoPE.

2. **Forward Pass**:
   - Through \( L \) layers: MLA computes attention (compressed KV cache for efficiency).
   - DeepSeekMoE routes to experts (only 37B params active).
   - Residuals and norms stabilize.

3. **Output Generation**:
   - Final hidden state \( \mathbf{h}_T \) projected to logits: \( \logits = W_{\text{out}} \mathbf{h}_T \).
   - Sample next token(s) via softmax: \( P(t_{T+1}) = \operatorname{Softmax}(\logits / \tau) \).
   - With MTP: Predict multiple, verify/accept speculatively.
   - Autoregressive loop until EOS or max length.

4. **Inference Optimizations**: FP8, node-limited routing, no token dropping. For math/reasoning, chain-of-thought (CoT) prompting elicits step-by-step outputs.

| Benchmark | DeepSeek-V3 Score | Comparison |
|-----------|-------------------|------------|
| GSM8K (Math) | 74.0% | Near GPT-4o |
| MATH | 39.8% | Strong for open-source |
| HumanEval (Code) | High 80s% | Competitive |

