# Tokenization, Positional Encoding, and Embeddings: A Complete Mathematical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Tokenization](#tokenization)
3. [Embedding Techniques](#embedding-techniques)
4. [Positional Encoding](#positional-encoding)
5. [Complete Pipeline](#complete-pipeline)
6. [Advanced Topics](#advanced-topics)

---

## Introduction

Before any neural network can process text, three fundamental transformations must occur:

1. **Tokenization**: Converting raw text into discrete units (tokens)
2. **Embedding**: Mapping tokens to continuous vector representations
3. **Positional Encoding**: Adding sequence order information to embeddings

These steps form the foundation of all modern NLP models, from word2vec to GPT-4.

### The Complete Pipeline

```
Raw Text: "Hello world"
    ↓
Tokenization: ["Hello", "world"]
    ↓
Token IDs: [15496, 995]
    ↓
Embedding Lookup: [[0.2, -0.5, ..., 0.3], [-0.1, 0.8, ..., 0.4]]
    ↓
Positional Encoding: Add position information
    ↓
Final Input: Ready for transformer/model
```

---

## Tokenization

### 1.1 What is Tokenization?

**Tokenization** is the process of breaking down text into smaller units called **tokens**. These tokens can be:
- Words
- Subwords
- Characters
- Bytes

### 1.2 Why Tokenization Matters

**Challenges with raw text:**
- Variable-length inputs
- Infinite vocabulary (new words constantly emerge)
- Spelling variations
- Morphological complexity

**Solution:** Map text to a fixed vocabulary of tokens with IDs.

---

### 1.3 Tokenization Approaches

#### A. Word-Level Tokenization

**Method:** Split text by whitespace and punctuation.

**Example:**
```
Text: "Hello, world!"
Tokens: ["Hello", ",", "world", "!"]
```

**Advantages:**
- Simple and intuitive
- Preserves word meanings

**Disadvantages:**
- Huge vocabulary (100k-1M+ words)
- Cannot handle out-of-vocabulary (OOV) words
- Poor for morphologically rich languages

**Mathematical Representation:**

Vocabulary: $V = \{w_1, w_2, ..., w_{|V|}\}$

Tokenization function: $f_{\text{word}}: \text{Text} \rightarrow [t_1, t_2, ..., t_n]$ where $t_i \in V$

---

#### B. Character-Level Tokenization

**Method:** Split text into individual characters.

**Example:**
```
Text: "Hello"
Tokens: ["H", "e", "l", "l", "o"]
```

**Advantages:**
- Tiny vocabulary (~100 characters for English)
- No OOV problem
- Captures spelling patterns

**Disadvantages:**
- Very long sequences
- Loses semantic meaning
- Computationally expensive

**Vocabulary Size:**
- English: ~100 characters (a-z, A-Z, 0-9, punctuation, space)
- Unicode: 143,859+ characters

---

#### C. Subword Tokenization

**Key Idea:** Find middle ground between words and characters.

Common words stay whole; rare words split into meaningful subunits.

**Example:**
```
Text: "unhappiness"
Tokens: ["un", "happiness"]  or  ["un", "happi", "ness"]
```

---

### 1.4 Byte Pair Encoding (BPE)

**Algorithm:** Iteratively merge the most frequent adjacent character pairs.

#### Mathematical Formulation

**Input:**
- Corpus: $C = \{s_1, s_2, ..., s_N\}$ (collection of strings)
- Target vocabulary size: $|V|$

**Initialization:**
- Start with character-level vocabulary: $V_0 = \{\text{all unique characters}\}$

**Iteration:**

For $i = 1$ to $\text{num\_merges}$:
1. Count all adjacent token pairs in corpus:
   $$\text{freq}(t_i, t_j) = \sum_{s \in C} \text{count of } (t_i, t_j) \text{ in } s$$

2. Find most frequent pair:
   $$(t_a, t_b) = \arg\max_{(t_i, t_j)} \text{freq}(t_i, t_j)$$

3. Merge into new token:
   $$t_{\text{new}} = t_a \circ t_b$$
   
4. Update vocabulary:
   $$V_{i} = V_{i-1} \cup \{t_{\text{new}}\}$$

5. Replace all occurrences of $(t_a, t_b)$ with $t_{\text{new}}$ in corpus

**Final vocabulary:** $V = V_{\text{num\_merges}}$

#### Step-by-Step Example

**Corpus:** `["low", "low", "low", "lower", "lower", "newest", "newest", "newest", "widest"]`

**Step 1: Initialize (character level)**
```
Vocabulary: {l, o, w, e, r, n, s, t, i, d}
Tokenized corpus:
  "low" → [l, o, w]
  "lower" → [l, o, w, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]
```

**Step 2: Count pairs**
```
(l, o): 5    (from 3×"low" + 2×"lower")
(o, w): 5    (from 3×"low" + 2×"lower")
(e, w): 3    (from 3×"newest")
(w, e): 3    (from 3×"newest")
(e, s): 3    (from 3×"newest")
(s, t): 4    (from 3×"newest" + 1×"widest")
...
```

**Step 3: Merge most frequent → (l, o) or (o, w) (tie)**

Choose (l, o):
```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, lo}
Tokenized corpus:
  "low" → [lo, w]
  "lower" → [lo, w, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]
```

**Step 4: Merge (lo, w)**
```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, lo, low}
Tokenized corpus:
  "low" → [low]
  "lower" → [low, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]
```

Continue until desired vocabulary size...

**Advantages:**
- Efficient compression
- Handles OOV gracefully
- Data-driven (learns from corpus)

**Disadvantages:**
- Training time intensive
- Language-specific
- May split semantically coherent words

---

### 1.5 WordPiece

**Used by:** BERT, DistilBERT

**Difference from BPE:** Instead of frequency, maximize **likelihood** of training data.

#### Mathematical Formulation

**Objective:** Choose merges that maximize log-likelihood:

$$\mathcal{L} = \sum_{s \in C} \log P(s)$$

where $P(s)$ is probability of sentence under unigram language model:

$$P(s) = \prod_{t \in \text{tokens}(s)} P(t)$$

**Merge criterion:**

$$\text{score}(t_a, t_b) = \frac{P(t_a \circ t_b)}{P(t_a) \cdot P(t_b)}$$

Merge pair with highest score (maximizes likelihood gain).

**Example:**

If $P(\text{"un"}) = 0.01$, $P(\text{"able"}) = 0.02$, $P(\text{"unable"}) = 0.005$:

$$\text{score}(\text{"un"}, \text{"able"}) = \frac{0.005}{0.01 \times 0.02} = \frac{0.005}{0.0002} = 25$$

High score → merge is beneficial.

**Special tokens:**
- `[CLS]`: Classification token (start of sequence)
- `[SEP]`: Separator token (between sentences)
- `[PAD]`: Padding token
- `[UNK]`: Unknown token
- `[MASK]`: Masked token (for MLM)

---

### 1.6 Unigram Language Model

**Used by:** SentencePiece (T5, ALBERT, XLNet)

**Method:** Start with large vocabulary, iteratively remove tokens.

#### Mathematical Formulation

**Model:** Unigram probabilities

$$P(x) = \prod_{i=1}^{|x|} P(x_i)$$

where $x = [x_1, ..., x_{|x|}]$ is tokenization of text.

**Training:**

1. Initialize with large vocabulary $V$
2. For each token $t \in V$:
   - Compute loss if $t$ is removed: $\mathcal{L}_{-t}$
3. Remove tokens with smallest impact (lowest loss increase)
4. Repeat until reaching target vocabulary size

**Tokenization (inference):**

For text $s$, find tokenization that maximizes probability:

$$\text{tokens}^* = \arg\max_{x \in \text{all\_tokenizations}(s)} P(x)$$

Found using **Viterbi algorithm** (dynamic programming).

**Advantages:**
- Reversible (can detokenize perfectly)
- Language-agnostic
- Multiple tokenization candidates

---

### 1.7 SentencePiece

**Key Innovation:** Treat text as **byte stream** (no pre-tokenization needed).

**Benefits:**
- Language-agnostic (works for all languages)
- No whitespace assumption
- Handles any Unicode text

**Implementation:**
- Can use BPE or Unigram algorithm
- Adds whitespace as special character: `▁`

**Example:**
```
Text: "Hello world"
Tokens: ["▁Hello", "▁world"]
```

The `▁` indicates word boundary, allowing perfect reconstruction.

---

### 1.8 Tokenization Comparison Table

| Method | Vocab Size | OOV Handling | Used By | Advantages |
|--------|-----------|--------------|---------|------------|
| Word | 100k-1M | ❌ Poor | Traditional NLP | Preserves meaning |
| Character | ~100 | ✅ Perfect | CharRNN, CharCNN | Tiny vocab |
| BPE | 32k-50k | ✅ Good | GPT, RoBERTa | Efficient compression |
| WordPiece | 30k | ✅ Good | BERT | Likelihood-based |
| Unigram | 32k | ✅ Good | T5, ALBERT | Probabilistic |
| SentencePiece | 32k | ✅ Perfect | mT5, XLM-R | Language-agnostic |

---

## Embedding Techniques

### 2.1 What are Embeddings?

**Embeddings** map discrete tokens to continuous vector representations.

**Formal definition:**

$$\text{Embedding}: V \rightarrow \mathbb{R}^d$$

where:
- $V$ is vocabulary (discrete space)
- $d$ is embedding dimension
- Each token $t \in V$ maps to vector $\mathbf{e}_t \in \mathbb{R}^d$

**Why embeddings?**
- Neural networks need continuous inputs
- Similar words should have similar representations
- Enable arithmetic in semantic space

---

### 2.2 One-Hot Encoding (Baseline)

**Method:** Represent each token as binary vector with single 1.

**Mathematical definition:**

For vocabulary $V$ with $|V| = n$, token $t_i$ maps to:

$$\mathbf{v}_i = [0, 0, ..., 1, ..., 0] \in \{0, 1\}^n$$

where the $i$-th position is 1.

**Example:**

Vocabulary: $\{\text{cat}, \text{dog}, \text{bird}\}$ (size 3)

```
"cat"  → [1, 0, 0]
"dog"  → [0, 1, 0]
"bird" → [0, 0, 1]
```

**Disadvantages:**
- High dimensionality (|V| can be 50k+)
- Sparse (only one 1, rest 0s)
- No semantic similarity: 
  $$\text{similarity}(\text{cat}, \text{dog}) = \mathbf{v}_{\text{cat}} \cdot \mathbf{v}_{\text{dog}} = 0$$
- Same as similarity between any two different words!

---

### 2.3 Learned Embeddings (Embedding Layer)

**Method:** Learn dense vector for each token during training.

**Architecture:**

$$\mathbf{E} \in \mathbb{R}^{|V| \times d}$$

where:
- Rows: vocabulary tokens
- Columns: embedding dimensions
- $\mathbf{E}[i, :]$ = embedding for token $i$

**Lookup operation:**

For token with ID $i$:

$$\mathbf{e}_i = \mathbf{E}[i, :] \in \mathbb{R}^d$$

**Example:**

Vocabulary size: 10,000
Embedding dimension: 512

$$\mathbf{E} \in \mathbb{R}^{10000 \times 512}$$

Token ID 42:
$$\mathbf{e}_{42} = \mathbf{E}[42, :] = [0.2, -0.5, 0.3, ..., 0.1] \in \mathbb{R}^{512}$$

**Training:**

Embeddings are learned via backpropagation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{E}[i, j]} = \frac{\partial \mathcal{L}}{\partial \mathbf{e}_i[j]}$$

**Properties:**
- Dense (all values non-zero)
- Low-dimensional (typically 128-1024)
- Captures semantics through training

---

### 2.4 Word2Vec

**Two architectures:** CBOW (Continuous Bag of Words) and Skip-gram

#### A. Skip-gram

**Objective:** Predict context words from center word.

**Mathematical formulation:**

Given word sequence $w_1, w_2, ..., w_T$, maximize:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where:
- $c$ is context window size
- $w_t$ is center word
- $w_{t+j}$ is context word

**Probability model (softmax):**

$$P(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_w^\top \mathbf{v}_{w_I})}$$

where:
- $\mathbf{v}_{w_I}$ is input (center) word embedding
- $\mathbf{v}_{w_O}$ is output (context) word embedding

**Example:**

Sentence: "The quick brown fox jumps"
Center word: "brown"
Context window: 2

Predict: $P(\text{quick} | \text{brown})$, $P(\text{The} | \text{brown})$, $P(\text{fox} | \text{brown})$, $P(\text{jumps} | \text{brown})$

**Problem:** Softmax over full vocabulary is expensive!

**Solution: Negative Sampling**

Instead of full softmax, sample negative examples:

$$\mathcal{L}_{\text{NS}} = \log \sigma(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_i}^\top \mathbf{v}_{w_I})]$$

where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid)
- $k$ is number of negative samples (typically 5-20)
- $P_n(w)$ is noise distribution (often $P(w)^{3/4}$)

---

#### B. CBOW (Continuous Bag of Words)

**Objective:** Predict center word from context words.

**Mathematical formulation:**

$$P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$$

**Context representation (average):**

$$\mathbf{h} = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} \mathbf{v}_{w_{t+j}}$$

**Prediction:**

$$P(w_t | \text{context}) = \frac{\exp(\mathbf{v}_{w_t}^\top \mathbf{h})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_w^\top \mathbf{h})}$$

**Comparison:**

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Predict | Context from word | Word from context |
| Speed | Slower | Faster |
| Rare words | Better | Worse |
| Small data | Better | Worse |

---

### 2.5 GloVe (Global Vectors)

**Key idea:** Factorize word co-occurrence matrix.

**Mathematical formulation:**

**Co-occurrence matrix:** $X \in \mathbb{R}^{|V| \times |V|}$

$$X_{ij} = \text{number of times word } j \text{ appears in context of word } i$$

**Objective:** Learn embeddings such that:

$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log(X_{ij})$$

where:
- $\mathbf{w}_i$ is word embedding
- $\tilde{\mathbf{w}}_j$ is context embedding
- $b_i, \tilde{b}_j$ are bias terms

**Loss function:**

$$\mathcal{L} = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log(X_{ij}) \right)^2$$

**Weighting function:**

$$f(x) = \begin{cases}
(x / x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}$$

where $\alpha = 0.75$, $x_{\max} = 100$ (typical values).

**Purpose of $f(x)$:**
- Downweight very frequent co-occurrences
- Avoid overweighting rare words

**Properties:**
- Captures global statistics (unlike Word2Vec's local context)
- Linear relationships: $\mathbf{king} - \mathbf{man} + \mathbf{woman} \approx \mathbf{queen}$

---

### 2.6 FastText

**Extension of Word2Vec** that uses **subword information**.

**Key innovation:** Represent word as bag of character n-grams.

**Mathematical formulation:**

For word $w$, extract character n-grams: $\mathcal{G}_w = \{g_1, g_2, ..., g_m\}$

**Embedding:**

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

where $\mathbf{z}_g$ is n-gram embedding.

**Example:**

Word: "where" (with 3-grams)
N-grams: `<wh`, `whe`, `her`, `ere`, `re>`

$$\mathbf{v}_{\text{where}} = \mathbf{z}_{\text{<wh}} + \mathbf{z}_{\text{whe}} + \mathbf{z}_{\text{her}} + \mathbf{z}_{\text{ere}} + \mathbf{z}_{\text{re>}}$$

**Advantages:**
- Handles OOV words (rare words)
- Captures morphology ("running", "runs", "runner" share n-grams)
- Better for morphologically rich languages

**Example with OOV:**

Training: "run", "running", "runner"
Test: "runs" (not in training)

Can still compute embedding:
$$\mathbf{v}_{\text{runs}} = \mathbf{z}_{\text{<ru}} + \mathbf{z}_{\text{run}} + \mathbf{z}_{\text{uns}} + \mathbf{z}_{\text{ns>}}$$

---

### 2.7 Contextual Embeddings (BERT, GPT)

**Problem with static embeddings:** Same word always has same embedding.

**Example:**
- "I went to the **bank** to deposit money" (financial institution)
- "I sat by the river **bank**" (land beside river)

Word2Vec/GloVe give same embedding for both!

**Solution: Contextual embeddings**

**Definition:** Embedding depends on entire sentence context.

$$\mathbf{e}_{\text{token}} = f(\text{entire sequence})$$

**BERT example:**

Input: "The cat sat on the mat"

$$[\mathbf{e}_{\text{The}}, \mathbf{e}_{\text{cat}}, \mathbf{e}_{\text{sat}}, \mathbf{e}_{\text{on}}, \mathbf{e}_{\text{the}}, \mathbf{e}_{\text{mat}}] = \text{BERT}(\text{input tokens})$$

Each $\mathbf{e}$ is different for each occurrence because it depends on context!

**Comparison:**

| Type | Word2Vec/GloVe | BERT/GPT |
|------|---------------|----------|
| Representation | Static | Contextual |
| Same word | Same vector | Different vector |
| Training | Unsupervised (co-occurrence) | Self-supervised (MLM/CLM) |
| Dimension | 100-300 | 768-1024+ |

---

## Positional Encoding

### 3.1 Why Positional Encoding?

**Problem:** Self-attention is **permutation invariant**.

For sequences $x = [x_1, x_2, x_3]$ and $x' = [x_2, x_1, x_3]$:

$$\text{Attention}(x) = \text{Attention}(x')$$

But word order matters!
- "Dog bites man" ≠ "Man bites dog"

**Solution:** Add positional information to embeddings.

---

### 3.2 Learned Positional Embeddings

**Method:** Learn a separate embedding for each position.

**Mathematical formulation:**

$$\mathbf{P} \in \mathbb{R}^{L \times d}$$

where:
- $L$ is maximum sequence length
- $d$ is embedding dimension
- $\mathbf{P}[i, :]$ is positional embedding for position $i$

**Final embedding:**

$$\mathbf{x}_i^{\text{final}} = \mathbf{x}_i^{\text{token}} + \mathbf{P}[i, :]$$

**Example:**

Token embeddings: $\mathbf{x}_{\text{cat}} = [0.1, 0.5, -0.2, 0.3]$

Position 2: $\mathbf{P}[2] = [0.05, -0.1, 0.15, 0.0]$

Final: $\mathbf{x}_2 = [0.15, 0.4, -0.05, 0.3]$

**Used by:** BERT, GPT-1, GPT-2

**Advantages:**
- Simple
- Flexible (learned from data)

**Disadvantages:**
- Fixed maximum length $L$
- Cannot extrapolate to longer sequences

---

### 3.3 Sinusoidal Positional Encoding

**Used by:** Original Transformer, T5

**Key idea:** Use deterministic function of position.

**Mathematical formulation:**

For position $\text{pos}$ and dimension $i$:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

where:
- $\text{pos}$ is position in sequence (0, 1, 2, ...)
- $i$ is dimension index (0, 1, 2, ..., $d/2 - 1$)
- $d$ is embedding dimension

**Intuition:**

Different dimensions use different frequencies:
- Low dimensions: high frequency (change quickly with position)
- High dimensions: low frequency (change slowly with position)

**Example computation:**

Position: $\text{pos} = 0$
Dimension: $d = 512$

For $i = 0$ (dimension 0):
$$\text{PE}(0, 0) = \sin\left(\frac{0}{10000^{0/512}}\right) = \sin(0) = 0$$

For $i = 0$ (dimension 1):
$$\text{PE}(0, 1) = \cos\left(\frac{0}{10000^{0/512}}\right) = \cos(0) = 1$$

For $i = 1$ (dimension 2):
$$\text{PE}(0, 2) = \sin\left(\frac{0}{10000^{2/512}}\right) = 0$$

For $i = 1$ (dimension 3):
$$\text{PE}(0, 3) = \cos\left(\frac{0}{10000^{2/512}}\right) = 1$$

---

Position: $\text{pos} = 1$

For $i = 0$ (dimension 0):
$$\text{PE}(1, 0) = \sin\left(\frac{1}{10000^{0/512}}\right) = \sin(1) \approx 0.841$$

For $i = 0$ (dimension 1):
$$\text{PE}(1, 1) = \cos\left(\frac{1}{10000^{0/512}}\right) = \cos(1) \approx 0.540$$

---

**Wavelength interpretation:**

Dimension $2i$ has wavelength:

$$\lambda_i = 2\pi \cdot 10000^{2i/d}$$

- Dimension 0: wavelength $= 2\pi$ (period = 1)
- Dimension 510: wavelength $= 2\pi \cdot 10000$ (period = 10000)

This creates a unique "fingerprint" for each position.

**Properties:**

1. **Bounded:** $\text{PE}(\text{pos}, i) \in [-1, 1]$

2. **Unique:** Each position has unique encoding

3. **Relative positions:** For fixed offset $k$:
   $$\text{PE}(\text{pos} + k) = M_k \cdot \text{PE}(\text{pos})$$
   where $M_k$ is linear transformation (allows model to learn relative positions)

4. **Extrapolation:** Works for positions beyond training (no maximum length)

**Visual representation:**

```
Position 0: [0.000,  1.000,  0.000,  1.000, ...]
Position 1: [0.841,  0.540,  0.010,  1.000, ...]
Position 2: [0.909, -0.416,  0.020,  1.000, ...]
Position 3: [0.141, -0.990,  0.030,  0.999, ...]
...
```

Each position is a unique wave pattern!

---

### 3.4 Rotary Positional Embedding (RoPE)

**Used by:** LLaMA, PaLM, GPT-NeoX

**Key innovation:** Encode position by **rotating** query and key vectors.

**Mathematical formulation:**

For position $m$, apply rotation to each pair of dimensions:

$$\begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix}_{\text{rotated}} = \begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix} \begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix}$$

where:
$$\theta_i = 10000^{-2i/d}$$

**For keys, same rotation:**

$$\begin{pmatrix}
k_{2i} \\
k_{2i+1}
\end{pmatrix}_{\text{rotated}} = \begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix} \begin{pmatrix}
k_{2i} \\
k_{2i+1}
\end{pmatrix}$$

**Key property:** Dot product encodes relative position!

For query at position $m$ and key at position $n$:

$$q_m^\top k_n = \text{same as } q_0^\top k_{n-m}$$

The attention depends only on relative distance $n - m$, not absolute positions.

**Advantages:**
- Encodes relative positions naturally
- Better extrapolation to longer sequences
- Efficient implementation

**Example (2D case):**

Position 0: No rotation
$$\begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

Position 1: Rotate by $\theta$
$$\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} = \begin{pmatrix}
q_0\cos\theta - q_1\sin\theta \\
q_0\sin\theta + q_1\cos\theta
\end{pmatrix}$$

Position 2: Rotate by $2\theta$
$$\begin{pmatrix}
\cos(2\theta) & -\sin(2\theta) \\
\sin(2\theta) & \cos(2\theta)
\end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

---

### 3.5 ALiBi (Attention with Linear Biases)

**Used by:** BLOOM, MPT

**Method:** Add bias to attention scores based on distance.

**Mathematical formulation:**

Standard attention score:
$$\text{score}(q_i, k_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

ALiBi attention score:
$$\text{score}_{\text{ALiBi}}(q_i, k_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m \cdot |i - j|$$

where:
- $m$ is head-specific slope (different for each attention head)
- $|i - j|$ is distance between positions

**Head-specific slopes:**

For head $h$ out of $H$ total heads:

$$m_h = 2^{-8h/H}$$

Example with 8 heads:
```
Head 1: m = 2^(-8/8) = 0.5
Head 2: m = 2^(-16/8) = 0.25
Head 3: m = 2^(-24/8) = 0.125
...
Head 8: m = 2^(-64/8) = 0.0039
```

**Effect:**
- Closer positions: small penalty
- Farther positions: large penalty
- Different heads have different "attention spans"

**Advantages:**
- No positional embeddings needed
- Excellent extrapolation to longer sequences
- Simple and efficient

---

### 3.6 Positional Encoding Comparison

| Method | Type | Max Length | Extrapolation | Relative | Used By |
|--------|------|------------|---------------|----------|---------|
| Learned | Absolute | Fixed | ❌ Poor | ❌ No | BERT, GPT-2 |
| Sinusoidal | Absolute | Unlimited | ✅ Good | ⚠️ Implicit | Transformer |
| RoPE | Relative | Unlimited | ✅ Excellent | ✅ Yes | LLaMA, PaLM |
| ALiBi | Relative | Unlimited | ✅ Excellent | ✅ Yes | BLOOM |

---

## Complete Pipeline

### 4.1 End-to-End Example

**Input:** "Hello world"

**Step 1: Tokenization** (using BPE)
```
Tokens: ["Hello", "world"]
Token IDs: [15496, 995]
```

**Step 2: Embedding Lookup**

Embedding matrix: $\mathbf{E} \in \mathbb{R}^{50000 \times 512}$

```
Token "Hello" (ID 15496):
  e_15496 = E[15496, :] = [0.23, -0.51, 0.18, ..., 0.42] ∈ ℝ^512

Token "world" (ID 995):
  e_995 = E[995, :] = [-0.11, 0.64, -0.32, ..., 0.19] ∈ ℝ^512
```

**Step 3: Positional Encoding** (sinusoidal)

Position 0:
```
PE(0, :) = [sin(0/10000^0), cos(0/10000^0), sin(0/10000^(2/512)), ...]
         = [0.000, 1.000, 0.000, 1.000, ...]
```

Position 1:
```
PE(1, :) = [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/512)), ...]
         = [0.841, 0.540, 0.010, 1.000, ...]
```

**Step 4: Combine**

```
x_0 = e_15496 + PE(0) = [0.23, -0.51, ...] + [0.00, 1.00, ...]
                      = [0.23, 0.49, ...]

x_1 = e_995 + PE(1) = [-0.11, 0.64, ...] + [0.84, 0.54, ...]
                    = [0.73, 1.18, ...]
```

**Step 5: Feed to Transformer**

```
Input to model: [x_0, x_1] ∈ ℝ^(2×512)
```

---

### 4.2 Mathematical Summary

**Complete transformation:**

$$\text{Raw text} \xrightarrow{\text{tokenize}} \text{Token IDs} \xrightarrow{\text{embed}} \text{Token vectors} \xrightarrow{\text{+pos}} \text{Input embeddings}$$

**Formally:**

1. Tokenization: $f_{\text{tok}}: \text{String} \rightarrow [t_1, ..., t_n]$ where $t_i \in \{1, ..., |V|\}$

2. Embedding: $\mathbf{e}_i = \mathbf{E}[t_i, :] \in \mathbb{R}^d$

3. Positional: $\mathbf{p}_i = \text{PE}(i) \in \mathbb{R}^d$

4. Final: $\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i \in \mathbb{R}^d$

**Input to model:**

$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n] \in \mathbb{R}^{n \times d}$$

---

## Advanced Topics

### 5.1 Embedding Dimension Selection

**Tradeoff:**
- Small $d$: Less parameters, faster, but limited capacity
- Large $d$: More parameters, slower, but richer representations

**Typical values:**
- Word2Vec: 100-300
- BERT-base: 768
- BERT-large: 1024
- GPT-3: 12,288
- GPT-4: Unknown (estimated 10k-20k)

**Rule of thumb:**

$$d \approx 4 \sqrt[4]{|V|}$$

For $|V| = 50000$: $d \approx 4 \times 15 = 60$... but this is too small!

Modern models use much larger $d$ for capacity.

---

### 5.2 Subword Regularization

**Problem:** Fixed tokenization may not be optimal.

**Solution:** During training, randomly sample different tokenizations.

**Example:**

Text: "unhappiness"

Possible tokenizations:
1. ["un", "happiness"] (probability 0.4)
2. ["unhap", "piness"] (probability 0.3)
3. ["un", "happi", "ness"] (probability 0.2)
4. ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"] (probability 0.1)

Sample based on probabilities during training → more robust model.

---

### 5.3 Byte-Level Encoding

**Used by:** GPT-2, RoBERTa

**Method:** Use bytes (256 values) as base vocabulary.

**Advantages:**
- True universal tokenizer (any text, any language)
- No unknown tokens ever
- Handles emojis, special characters perfectly

**Example:**

Text: "Hello 😊"

UTF-8 bytes: `[72, 101, 108, 108, 111, 32, 240, 159, 152, 138]`

Then apply BPE on byte sequence.

---

### 5.4 Vocabulary Size Impact

**Small vocabulary (< 10k):**
- Pros: Fewer parameters in embedding layer, faster training
- Cons: Longer sequences, more subword splits, loss of semantic units

**Large vocabulary (> 100k):**
- Pros: Shorter sequences, preserve word meanings
- Cons: More parameters, rare tokens undertrained, OOV issues

**Sweet spot:** 30k-50k for most languages

---

## Key Takeaways

1. **Tokenization** converts text to discrete units
   - BPE/WordPiece balance vocabulary size and sequence length
   - Modern models use subword tokenization (32k-50k vocab)

2. **Embeddings** map tokens to continuous vectors
   - Learned embeddings capture semantics through training
   - Contextual embeddings (BERT) depend on full context

3. **Positional encoding** adds sequence order
   - Sinusoidal: deterministic, unlimited length
   - Learned: flexible but fixed max length
   - RoPE/ALiBi: better extrapolation

4. **Complete pipeline:**
   Text → Tokens → IDs → Embeddings → +Position → Model Input

---

## References

- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Schuster & Nakajima (2012). "Japanese and Korean Voice Search" (WordPiece)
- Kudo & Richardson (2018). "SentencePiece" (Unigram)
- Mikolov et al. (2013). "Efficient Estimation of Word Representations" (Word2Vec)
- Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"
- Bojanowski et al. (2017). "Enriching Word Vectors with Subword Information" (FastText)
- Vaswani et al. (2017). "Attention Is All You Need" (Sinusoidal PE)
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE)
- Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases" (ALiBi)

---

*This document provides a complete mathematical treatment. For implementation, see the accompanying code file.*
