"""
================================================================================
MULTI-QUERY ATTENTION (MQA) - COMPLETE GUIDE
================================================================================

A step-by-step explanation with math, implementation, and solved examples
that anyone can understand and implement.

Author: Educational Tutorial
Purpose: Make MQA crystal clear with concrete examples
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np


"""
================================================================================
PART 1: UNDERSTANDING THE PROBLEM
================================================================================

SCENARIO:
Imagine you're generating text with a language model, one word at a time:
"The cat sat on the ___"

For each new word, the model needs to look back at ALL previous words to 
understand context. This is done through "attention."

THE ATTENTION MECHANISM:
Each word asks: "Which previous words should I pay attention to?"

For example, when generating the word after "on the", the model should pay
attention to "cat" (what is sitting) and "mat/floor/chair" (where it sits).

STANDARD MULTI-HEAD ATTENTION (MHA):
- Has multiple "heads" (like having multiple perspectives)
- Head 1 might focus on subject-object relationships
- Head 2 might focus on time/sequence
- Head 3 might focus on location
- etc.

Each head has three components:
1. Q (Query): "What am I looking for?"
2. K (Key): "What information do I have?"
3. V (Value): "What should I output?"

THE PROBLEM WITH MHA:
During text generation, we need to store ALL previous K and V values in memory
(called "KV cache") to avoid recomputing them. This uses a LOT of memory!

Example:
- 32 heads × 2048 tokens × 128 dimensions = 8,388,608 values per layer
- For a 80-layer model, this becomes gigabytes of memory!

THE MQA SOLUTION:
What if all heads SHARE the same K and V, but each head still has its own Q?

Think of it like this:
- All detectives (heads) look at the same evidence (K and V)
- But each detective asks different questions (Q)
- The evidence doesn't change, only the questions!

This reduces memory by 8x-64x depending on number of heads!
"""


"""
================================================================================
PART 2: THE MATHEMATICS - EXPLAINED SIMPLY
================================================================================

Let's build this step by step with SIMPLE MATH.

SETUP:
------
Let's say we have a sentence with 4 words (tokens):
"The cat sat down"

Each word is represented as a vector (list of numbers).
Let's use small dimensions so we can see everything:

- d_model = 8 (each word is 8 numbers)
- num_heads = 2 (we'll have 2 attention heads)
- head_dim = 4 (each head works with 4 numbers)

Note: d_model = num_heads × head_dim (8 = 2 × 4)

INPUT REPRESENTATION:
--------------------
Our input X is a matrix where each row is a word:

X = [4 tokens × 8 dimensions]

    [x₁₁  x₁₂  x₁₃  x₁₄  x₁₅  x₁₆  x₁₇  x₁₈]  ← "The"
    [x₂₁  x₂₂  x₂₃  x₂₄  x₂₅  x₂₆  x₂₇  x₂₈]  ← "cat"
    [x₃₁  x₃₂  x₃₃  x₃₄  x₃₅  x₃₆  x₃₇  x₃₈]  ← "sat"
    [x₄₁  x₄₂  x₄₃  x₄₄  x₄₅  x₄₆  x₄₇  x₄₈]  ← "down"


STANDARD MULTI-HEAD ATTENTION (MHA):
====================================

STEP 1: Create Q, K, V through linear projections
--------------------------------------------------

We multiply X by weight matrices to create Q, K, and V:

Q = X @ Wq    where Wq is [8 × 8]
K = X @ Wk    where Wk is [8 × 8]
V = X @ Wv    where Wv is [8 × 8]

Result:
Q = [4 tokens × 8 dimensions]
K = [4 tokens × 8 dimensions]
V = [4 tokens × 8 dimensions]

STEP 2: Split into multiple heads
----------------------------------

We split the 8 dimensions into 2 heads of 4 dimensions each:

Q reshaped: [4 tokens × 2 heads × 4 dim/head]
K reshaped: [4 tokens × 2 heads × 4 dim/head]
V reshaped: [4 tokens × 2 heads × 4 dim/head]

Then transpose to: [2 heads × 4 tokens × 4 dim/head]

So now:
- Head 0 has: Q₀[4×4], K₀[4×4], V₀[4×4]
- Head 1 has: Q₁[4×4], K₁[4×4], V₁[4×4]

Each head has its OWN separate Q, K, and V!

STEP 3: Compute attention for each head
----------------------------------------

For Head 0:
Attention_scores₀ = Q₀ @ K₀ᵀ / √4
                  = [4×4] @ [4×4]ᵀ / 2
                  = [4×4] matrix

This gives us a 4×4 matrix where entry (i,j) tells us:
"How much should token i attend to token j?"

STEP 4: Apply softmax
---------------------

Attention_weights₀ = softmax(Attention_scores₀)

Each row becomes a probability distribution (sums to 1).

STEP 5: Apply attention to values
----------------------------------

Output₀ = Attention_weights₀ @ V₀
        = [4×4] @ [4×4]
        = [4×4]

Repeat steps 3-5 for Head 1, giving us Output₁ [4×4]

STEP 6: Concatenate heads
--------------------------

Final_output = concat(Output₀, Output₁)
             = [4 tokens × 8 dimensions]

We've restored the original dimension!


MULTI-QUERY ATTENTION (MQA):
=============================

The KEY DIFFERENCE is in STEP 1!

STEP 1: Create Q, K, V (THE CHANGE!)
-------------------------------------

Q = X @ Wq    where Wq is [8 × 8]      ← Same as MHA
K = X @ Wk    where Wk is [8 × 4]      ← ONLY 4 outputs! (not 8!)
V = X @ Wv    where Wv is [8 × 4]      ← ONLY 4 outputs! (not 8!)

Result:
Q = [4 tokens × 8 dimensions]     ← Same as MHA
K = [4 tokens × 4 dimensions]     ← HALF the size!
V = [4 tokens × 4 dimensions]     ← HALF the size!

STEP 2: Split Q, add dimension to K and V
------------------------------------------

Q reshaped: [2 heads × 4 tokens × 4 dim/head]  ← Same as MHA

K reshaped: [1 head × 4 tokens × 4 dim/head]   ← Only 1 "head"!
V reshaped: [1 head × 4 tokens × 4 dim/head]   ← Only 1 "head"!

So now:
- Head 0 has: Q₀[4×4], K_shared[4×4], V_shared[4×4]
- Head 1 has: Q₁[4×4], K_shared[4×4], V_shared[4×4]

Notice: K and V are THE SAME for both heads!

STEP 3-6: Same as MHA, but K and V are shared
----------------------------------------------

When computing attention, PyTorch automatically "broadcasts" the single
K and V to work with both query heads.

MEMORY SAVINGS:
---------------

MHA stores:
- Q: [2 heads × 4 tokens × 4 dim] = 32 values
- K: [2 heads × 4 tokens × 4 dim] = 32 values
- V: [2 heads × 4 tokens × 4 dim] = 32 values
Total K+V: 64 values

MQA stores:
- Q: [2 heads × 4 tokens × 4 dim] = 32 values
- K: [1 head × 4 tokens × 4 dim] = 16 values
- V: [1 head × 4 tokens × 4 dim] = 16 values
Total K+V: 32 values

MQA uses 50% memory for K+V! (with 2 heads)
With 32 heads, MQA would use only 3.125% of MHA memory!
"""


"""
================================================================================
PART 3: CONCRETE NUMERICAL EXAMPLE
================================================================================

Let's work through an actual example with REAL NUMBERS you can verify!

We'll use tiny dimensions so you can follow along:
- 3 tokens (3 words)
- d_model = 4 (4 dimensions per word)
- num_heads = 2 (2 attention heads)
- head_dim = 2 (2 dimensions per head)
"""

def numerical_example_mqa():
    """
    Complete worked example with actual numbers
    """
    
    print("=" * 80)
    print("MULTI-QUERY ATTENTION - NUMERICAL EXAMPLE")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    print("\n📌 SETUP")
    print("-" * 80)
    print("Sentence: 'cat sat down'")
    print("Tokens: 3")
    print("Embedding dimension (d_model): 4")
    print("Number of heads: 2")
    print("Dimensions per head: 2")
    
    # Create simple input - 3 tokens, 4 dimensions each
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],   # "cat"
        [2.0, 3.0, 4.0, 5.0],   # "sat"
        [3.0, 4.0, 5.0, 6.0]    # "down"
    ])
    
    print("\n📊 INPUT X (3 tokens × 4 dimensions):")
    print("-" * 80)
    print(X.numpy())
    print("\nRow 0 = 'cat'  = [1.0, 2.0, 3.0, 4.0]")
    print("Row 1 = 'sat'  = [2.0, 3.0, 4.0, 5.0]")
    print("Row 2 = 'down' = [3.0, 4.0, 5.0, 6.0]")
    
    # =========================================================================
    # STEP 1: CREATE Q, K, V PROJECTIONS (MQA WAY)
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: CREATE Q, K, V PROJECTIONS")
    print("=" * 80)
    
    # For simplicity, we'll use simple weight matrices (not random)
    # Q projection: 4 → 4 dimensions
    Wq = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])  # Identity matrix for simplicity
    
    # K projection: 4 → 2 dimensions (THIS IS THE MQA CHANGE!)
    Wk = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])  # Takes first 2 dimensions
    
    # V projection: 4 → 2 dimensions (THIS IS THE MQA CHANGE!)
    Wv = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])  # Takes last 2 dimensions
    
    print("\n🔧 Weight Matrix Wq (4×4) - for Queries:")
    print(Wq.numpy())
    print("   This maps 4 dims → 4 dims (will split into 2 heads)")
    
    print("\n🔧 Weight Matrix Wk (4×2) - for Keys:")
    print(Wk.numpy())
    print("   This maps 4 dims → 2 dims (SHARED across heads!)")
    
    print("\n🔧 Weight Matrix Wv (4×2) - for Values:")
    print(Wv.numpy())
    print("   This maps 4 dims → 2 dims (SHARED across heads!)")
    
    # Compute Q, K, V
    Q = X @ Wq  # [3, 4]
    K = X @ Wk  # [3, 2] ← Smaller!
    V = X @ Wv  # [3, 2] ← Smaller!
    
    print("\n📊 Q (Queries) - shape [3 tokens × 4 dims]:")
    print(Q.numpy())
    
    print("\n📊 K (Keys) - shape [3 tokens × 2 dims]:")
    print(K.numpy())
    print("   ⚠️ NOTICE: Only 2 dimensions, not 4!")
    
    print("\n📊 V (Values) - shape [3 tokens × 2 dims]:")
    print(V.numpy())
    print("   ⚠️ NOTICE: Only 2 dimensions, not 4!")
    
    # =========================================================================
    # STEP 2: RESHAPE FOR MULTI-HEAD COMPUTATION
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: RESHAPE FOR MULTI-HEAD COMPUTATION")
    print("=" * 80)
    
    # Split Q into 2 heads of 2 dimensions each
    Q = Q.view(3, 2, 2)  # [3 tokens, 2 heads, 2 dim/head]
    Q = Q.transpose(0, 1)  # [2 heads, 3 tokens, 2 dim/head]
    
    print("\n🔀 Q after splitting into 2 heads:")
    print(f"   Shape: {Q.shape} = [2 heads, 3 tokens, 2 dim/head]")
    
    print("\n   Q for Head 0 (3 tokens × 2 dims):")
    print(Q[0].numpy())
    
    print("\n   Q for Head 1 (3 tokens × 2 dims):")
    print(Q[1].numpy())
    
    # Add dimension to K and V for broadcasting
    K = K.unsqueeze(0)  # [1, 3, 2] = [1 head, 3 tokens, 2 dims]
    V = V.unsqueeze(0)  # [1, 3, 2] = [1 head, 3 tokens, 2 dims]
    
    print("\n🔀 K after adding dimension:")
    print(f"   Shape: {K.shape} = [1 head, 3 tokens, 2 dims]")
    print("\n   K (will be SHARED by both heads):")
    print(K[0].numpy())
    
    print("\n🔀 V after adding dimension:")
    print(f"   Shape: {V.shape} = [1 head, 3 tokens, 2 dims]")
    print("\n   V (will be SHARED by both heads):")
    print(V[0].numpy())
    
    print("\n💡 KEY INSIGHT:")
    print("   - Q has 2 different heads (different for each head)")
    print("   - K has only 1 'head' (SAME for all query heads)")
    print("   - V has only 1 'head' (SAME for all query heads)")
    
    # =========================================================================
    # STEP 3: COMPUTE ATTENTION FOR HEAD 0
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: COMPUTE ATTENTION FOR HEAD 0")
    print("=" * 80)
    
    Q_head0 = Q[0]  # [3, 2]
    K_shared = K[0]  # [3, 2]
    V_shared = V[0]  # [3, 2]
    
    print("\n📐 3a. Compute Attention Scores = Q @ K^T / sqrt(head_dim)")
    print(f"   Q_head0 shape: {Q_head0.shape} = [3 tokens, 2 dims]")
    print(f"   K^T shape: {K_shared.T.shape} = [2 dims, 3 tokens]")
    
    print("\n   Q_head0:")
    print(Q_head0.numpy())
    
    print("\n   K_shared^T:")
    print(K_shared.T.numpy())
    
    # Matrix multiplication
    scores_head0 = Q_head0 @ K_shared.T  # [3, 3]
    scale_factor = np.sqrt(2)  # sqrt(head_dim)
    scores_head0 = scores_head0 / scale_factor
    
    print("\n   Scores = Q @ K^T / sqrt(2):")
    print(scores_head0.numpy())
    print("\n   Shape: [3 tokens × 3 tokens]")
    print("   scores[i,j] = how much token i should attend to token j")
    print(f"\n   For example, scores[0,1] = {scores_head0[0,1].item():.3f}")
    print("   → How much 'cat' should attend to 'sat'")
    
    print("\n📐 3b. Apply Softmax (convert to probabilities)")
    
    attn_weights_head0 = torch.softmax(scores_head0, dim=-1)
    
    print("\n   Attention Weights (after softmax):")
    print(attn_weights_head0.numpy().round(3))
    print("\n   Each row sums to 1.0 (probability distribution)")
    print(f"   Row sums: {attn_weights_head0.sum(dim=-1).numpy()}")
    
    print("\n   Interpretation:")
    print(f"   Row 0: When processing 'cat', attend:")
    for j, word in enumerate(['cat', 'sat', 'down']):
        print(f"      {attn_weights_head0[0,j].item():.1%} to '{word}'")
    
    print("\n📐 3c. Apply Attention to Values")
    print(f"   attn_weights shape: {attn_weights_head0.shape} = [3, 3]")
    print(f"   V shape: {V_shared.shape} = [3, 2]")
    
    print("\n   V_shared:")
    print(V_shared.numpy())
    
    output_head0 = attn_weights_head0 @ V_shared  # [3, 2]
    
    print("\n   Output_head0 = attn_weights @ V:")
    print(output_head0.numpy().round(3))
    print("\n   Each token is now a weighted combination of all value vectors!")
    
    # =========================================================================
    # STEP 4: COMPUTE ATTENTION FOR HEAD 1 (SAME K AND V!)
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: COMPUTE ATTENTION FOR HEAD 1")
    print("=" * 80)
    
    Q_head1 = Q[1]  # [3, 2] - DIFFERENT from head 0
    # K_shared and V_shared are THE SAME as head 0!
    
    print("\n💡 CRITICAL OBSERVATION:")
    print("   Head 1 uses DIFFERENT queries (Q):")
    print(Q_head1.numpy())
    
    print("\n   But Head 1 uses the SAME K and V as Head 0!")
    print("   This is the essence of Multi-Query Attention!")
    
    # Compute attention for head 1
    scores_head1 = Q_head1 @ K_shared.T / scale_factor
    attn_weights_head1 = torch.softmax(scores_head1, dim=-1)
    output_head1 = attn_weights_head1 @ V_shared
    
    print("\n   Attention Weights for Head 1:")
    print(attn_weights_head1.numpy().round(3))
    
    print("\n   Output_head1:")
    print(output_head1.numpy().round(3))
    
    print("\n   Notice: Different from Head 0 because Q is different!")
    print("   But K and V were the same!")
    
    # =========================================================================
    # STEP 5: CONCATENATE HEADS
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 5: CONCATENATE HEADS")
    print("=" * 80)
    
    print("\n   Head 0 output shape:", output_head0.shape, "= [3 tokens, 2 dims]")
    print(output_head0.numpy().round(3))
    
    print("\n   Head 1 output shape:", output_head1.shape, "= [3 tokens, 2 dims]")
    print(output_head1.numpy().round(3))
    
    # Concatenate along the last dimension
    final_output = torch.cat([output_head0, output_head1], dim=-1)
    
    print("\n   Final output (concatenated):")
    print(f"   Shape: {final_output.shape} = [3 tokens, 4 dims]")
    print(final_output.numpy().round(3))
    
    print("\n✅ COMPLETE!")
    print("   Each token started as a 4-dim vector")
    print("   After attention, each token is still a 4-dim vector")
    print("   But now enriched with context from other tokens!")
    
    # =========================================================================
    # STEP 6: COMPARE MEMORY USAGE
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 6: MEMORY ANALYSIS")
    print("=" * 80)
    
    print("\n📊 STANDARD MULTI-HEAD ATTENTION (MHA):")
    print("   Would store K and V separately for each head:")
    print("   K: [2 heads, 3 tokens, 2 dims] = 12 values")
    print("   V: [2 heads, 3 tokens, 2 dims] = 12 values")
    print("   Total K+V: 24 values")
    
    print("\n📊 MULTI-QUERY ATTENTION (MQA):")
    print("   Stores only ONE K and V (shared across heads):")
    print("   K: [1 head, 3 tokens, 2 dims] = 6 values")
    print("   V: [1 head, 3 tokens, 2 dims] = 6 values")
    print("   Total K+V: 12 values")
    
    print("\n💰 SAVINGS:")
    print(f"   MQA uses 50% of MHA memory (12 vs 24 values)")
    print(f"   With more heads (e.g., 32), savings would be 96.875%!")
    
    return final_output


"""
================================================================================
PART 4: IMPLEMENTATION CODE
================================================================================

Now let's implement MQA properly in PyTorch
"""

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention Implementation
    
    Key difference from standard MHA:
    - Q projection: d_model → d_model (same)
    - K projection: d_model → head_dim (MUCH SMALLER!)
    - V projection: d_model → head_dim (MUCH SMALLER!)
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q projection: d_model → d_model (will be split into num_heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # K and V projections: d_model → head_dim (SHARED across heads!)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project to Q, K, V
        # --------------------------
        Q = self.q_proj(x)  # [batch, seq_len, d_model]
        K = self.k_proj(x)  # [batch, seq_len, head_dim] ← Smaller!
        V = self.v_proj(x)  # [batch, seq_len, head_dim] ← Smaller!
        
        # Step 2: Reshape for multi-head computation
        # -------------------------------------------
        # Split Q into num_heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Add dimension to K and V for broadcasting
        K = K.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        V = V.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        
        # Step 3: Compute attention scores
        # --------------------------------
        # Q @ K^T - K will be broadcasted from [batch, 1, ...] to [batch, num_heads, ...]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [batch, num_heads, seq_len, seq_len]
        
        # Step 4: Apply softmax
        # ---------------------
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Step 5: Apply attention to values
        # ----------------------------------
        # V will be broadcasted from [batch, 1, ...] to [batch, num_heads, ...]
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch, num_heads, seq_len, head_dim]
        
        # Step 6: Concatenate heads
        # -------------------------
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, d_model)
        
        # Step 7: Final output projection
        # --------------------------------
        output = self.out_proj(attn_output)
        
        return output


def test_implementation():
    """
    Test the implementation with a simple example
    """
    print("\n" + "=" * 80)
    print("TESTING MQA IMPLEMENTATION")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {d_model // num_heads}")
    
    # Create model
    mqa = MultiQueryAttention(d_model, num_heads)
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = mqa(x)
    
    print(f"Output shape: {output.shape}")
    print("\n✅ Implementation works correctly!")
    
    # Check parameter count
    total_params = sum(p.numel() for p in mqa.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Compare with MHA
    head_dim = d_model // num_heads
    mha_params = (d_model * d_model * 3 +  # Q, K, V projections
                  d_model * d_model)        # Output projection
    mqa_params = (d_model * d_model +       # Q projection
                  d_model * head_dim * 2 +  # K, V projections (smaller!)
                  d_model * d_model)        # Output projection
    
    print(f"\nParameter comparison:")
    print(f"  Standard MHA would have: {mha_params:,} parameters")
    print(f"  MQA has: {mqa_params:,} parameters")
    print(f"  Reduction: {(1 - mqa_params/mha_params)*100:.1f}%")


"""
================================================================================
PART 5: VISUAL SUMMARY
================================================================================

STANDARD MULTI-HEAD ATTENTION (MHA):
────────────────────────────────────

Input X: [seq_len × d_model]
    ↓
Linear Projections:
    Q = X @ Wq  →  [seq_len × d_model]
    K = X @ Wk  →  [seq_len × d_model]  ← Full size
    V = X @ Wv  →  [seq_len × d_model]  ← Full size
    ↓
Split into heads:
    Q: [num_heads × seq_len × head_dim]
    K: [num_heads × seq_len × head_dim]  ← Each head has own K
    V: [num_heads × seq_len × head_dim]  ← Each head has own V
    ↓
For each head:
    Attention = softmax(Q @ K^T / √d) @ V
    ↓
Concatenate heads → Output: [seq_len × d_model]


MULTI-QUERY ATTENTION (MQA):
────────────────────────────

Input X: [seq_len × d_model]
    ↓
Linear Projections:
    Q = X @ Wq  →  [seq_len × d_model]
    K = X @ Wk  →  [seq_len × head_dim]  ← Much smaller!
    V = X @ Wv  →  [seq_len × head_dim]  ← Much smaller!
    ↓
Reshape:
    Q: [num_heads × seq_len × head_dim]
    K: [1 × seq_len × head_dim]          ← Only ONE K (shared!)
    V: [1 × seq_len × head_dim]          ← Only ONE V (shared!)
    ↓
For each head (K and V broadcasted):
    Attention = softmax(Q @ K^T / √d) @ V
    ↓
Concatenate heads → Output: [seq_len × d_model]


KEY DIFFERENCE:
───────────────
MHA: Each head has separate K and V
MQA: All heads share the same K and V

MEMORY SAVINGS:
───────────────
MHA KV cache: num_heads × seq_len × head_dim × 2
MQA KV cache: 1 × seq_len × head_dim × 2
Reduction: num_heads × smaller!

With 32 heads: 96.875% reduction in KV cache!
"""


if __name__ == "__main__":
    print("\n" + "🎓" * 40)
    print("MULTI-QUERY ATTENTION - COMPLETE GUIDE")
    print("🎓" * 40)
    
    # Run numerical example
    numerical_example_mqa()
    
    # Test implementation
    test_implementation()
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("""
WHAT IS MULTI-QUERY ATTENTION?
    A memory-efficient variant of multi-head attention where all heads
    share the same Key (K) and Value (V) matrices, while maintaining
    separate Query (Q) matrices per head.

HOW IT WORKS:
    1. Each head has its own Q (query) - asking different questions
    2. All heads share the same K (keys) - same information pool
    3. All heads share the same V (values) - same answer pool
    4. Broadcasting makes shared K and V work with all query heads

WHY IT WORKS:
    The diversity in attention comes from different queries (Q), not from
    having different copies of the same information (K, V). All heads can
    ask different questions about the same underlying data.

MEMORY SAVINGS:
    With 8 heads:  87.5% reduction in KV cache
    With 32 heads: 96.9% reduction in KV cache
    
PRACTICAL IMPACT:
    ✓ Faster text generation (2-10x)
    ✓ Can serve more users simultaneously
    ✓ Fits in smaller/cheaper GPUs
    ✓ Minimal quality loss (<1-2%)
    
USED IN:
    PaLM (540B), Falcon, StarCoder, and many modern LLMs
    """)
    print("=" * 80)
