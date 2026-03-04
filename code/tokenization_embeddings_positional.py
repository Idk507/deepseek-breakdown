"""
Tokenization, Embeddings, and Positional Encoding Implementation
==================================================================

This file contains:
1. Tokenization implementations (BPE, WordPiece simulation, character)
2. Embedding techniques (learned, Word2Vec-style, GloVe-style)
3. Positional encoding implementations (sinusoidal, learned, RoPE, ALiBi)
4. Complete end-to-end pipeline
5. Visualization and analysis tools

Author: Educational Implementation
Date: 2026
"""

import numpy as np
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import re

# ============================================================================
# PART 1: TOKENIZATION
# ============================================================================

class CharacterTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Get all unique characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Create mappings (sorted for consistency)
        chars = sorted(chars)
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for char, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
        print(f"Character vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join([self.id_to_char.get(id, '?') for id in ids])


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    
    Implements the BPE algorithm for subword tokenization.
    """
    
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
    
    def _get_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Get initial vocabulary with character frequencies."""
        vocab = defaultdict(int)
        for text in texts:
            # Split into words
            words = text.lower().split()
            for word in words:
                # Add end-of-word marker
                word = ' '.join(list(word)) + ' </w>'
                vocab[word] += 1
        return vocab
    
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent token pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge all occurrences of the most frequent pair."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def fit(self, texts: List[str]):
        """Learn BPE merges from texts."""
        print(f"Training BPE with target vocab size: {self.vocab_size}")
        
        # Get initial vocabulary
        vocab = self._get_vocab(texts)
        
        # Get all unique characters as base vocabulary
        chars = set()
        for word in vocab.keys():
            chars.update(word.split())
        
        # Initialize token mappings with characters
        self.token_to_id = {char: i for i, char in enumerate(sorted(chars))}
        self.id_to_token = {i: char for char, i in self.token_to_id.items()}
        
        num_merges = self.vocab_size - len(self.token_to_id)
        
        # Learn merges
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge it
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            # Add new token to vocabulary
            new_token = ''.join(best_pair)
            if new_token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[new_token] = token_id
                self.id_to_token[token_id] = new_token
            
            if (i + 1) % 50 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {best_pair} -> {new_token}")
        
        print(f"Final vocabulary size: {len(self.token_to_id)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            # Start with characters
            word_tokens = list(word) + ['</w>']
            word_str = ' '.join(word_tokens)
            
            # Apply learned merges
            for pair in self.merges:
                bigram = ' '.join(pair)
                replacement = ''.join(pair)
                word_str = word_str.replace(bigram, replacement)
            
            # Convert to IDs
            tokens = word_str.split()
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Unknown token - use first character
                    if token[0] in self.token_to_id:
                        token_ids.append(self.token_to_id[token[0]])
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(id, '?') for id in ids]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text


class SimpleWordTokenizer:
    """Simple word-level tokenizer."""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Create mappings (sorted by frequency)
        words = [word for word, _ in word_counts.most_common()]
        self.word_to_id = {word: i for i, word in enumerate(words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        
        print(f"Word vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = text.lower().split()
        return [self.word_to_id.get(word, 0) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ' '.join([self.id_to_word.get(id, '<UNK>') for id in ids])


# ============================================================================
# PART 2: EMBEDDINGS
# ============================================================================

class TokenEmbedding:
    """
    Learned token embedding layer.
    
    Maps discrete token IDs to continuous vectors.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        """
        Initialize embedding matrix.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            embedding_dim: Dimension of embedding vectors
            seed: Random seed for initialization
        """
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * scale
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Array of token IDs (batch_size, seq_len) or (seq_len,)
            
        Returns:
            embeddings: (batch_size, seq_len, embedding_dim) or (seq_len, embedding_dim)
        """
        return self.embeddings[token_ids]
    
    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding for a single token."""
        return self.embeddings[token_id]
    
    def similarity(self, id1: int, id2: int) -> float:
        """Compute cosine similarity between two tokens."""
        emb1 = self.embeddings[id1]
        emb2 = self.embeddings[id2]
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)


class Word2VecSkipGram:
    """
    Simplified Word2Vec Skip-gram implementation.
    
    Predicts context words from center word.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, window_size: int = 2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        # Input embeddings (center words)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Output embeddings (context words)
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def train_step(self, center_id: int, context_ids: List[int], 
                    negative_ids: List[int], learning_rate: float = 0.01):
        """
        Single training step with negative sampling.
        
        Args:
            center_id: Center word ID
            context_ids: Context word IDs (positive samples)
            negative_ids: Negative sample IDs
            learning_rate: Learning rate
        """
        center_vec = self.W_in[center_id]
        
        # Positive samples
        for context_id in context_ids:
            context_vec = self.W_out[context_id]
            
            # Sigmoid of dot product
            score = np.dot(center_vec, context_vec)
            pred = 1.0 / (1.0 + np.exp(-score))
            
            # Gradient (target = 1 for positive)
            grad = learning_rate * (1 - pred)
            
            # Update
            self.W_in[center_id] += grad * context_vec
            self.W_out[context_id] += grad * center_vec
        
        # Negative samples
        for negative_id in negative_ids:
            negative_vec = self.W_out[negative_id]
            
            # Sigmoid of dot product
            score = np.dot(center_vec, negative_vec)
            pred = 1.0 / (1.0 + np.exp(-score))
            
            # Gradient (target = 0 for negative)
            grad = learning_rate * (-pred)
            
            # Update
            self.W_in[center_id] += grad * negative_vec
            self.W_out[negative_id] += grad * center_vec
    
    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get final embedding (average of input and output)."""
        return (self.W_in[token_id] + self.W_out[token_id]) / 2


# ============================================================================
# PART 3: POSITIONAL ENCODING
# ============================================================================

class SinusoidalPositionalEncoding:
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
        """
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute positional encodings
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create positional encoding matrix."""
        pe = np.zeros((self.max_len, self.d_model))
        
        # Position indices
        position = np.arange(0, self.max_len)[:, np.newaxis]
        
        # Dimension indices
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional encodings for sequence.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encodings (seq_len, d_model)
        """
        return self.pe[:seq_len, :]
    
    def visualize_frequencies(self, positions: int = 100):
        """Print positional encoding for visualization."""
        print(f"Sinusoidal Positional Encoding (first 8 dims, {positions} positions)")
        print("=" * 80)
        
        pe = self.pe[:positions, :8]
        
        for pos in range(min(10, positions)):
            print(f"Position {pos:2d}: ", end="")
            for dim in range(8):
                print(f"{pe[pos, dim]:7.3f} ", end="")
            print()
        
        if positions > 10:
            print("...")


class LearnedPositionalEncoding:
    """Learned positional embeddings (like BERT)."""
    
    def __init__(self, max_len: int, d_model: int, seed: int = 42):
        """
        Initialize learned positional embeddings.
        
        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
            seed: Random seed
        """
        np.random.seed(seed)
        
        self.max_len = max_len
        self.d_model = d_model
        
        # Initialize position embeddings
        scale = np.sqrt(2.0 / (max_len + d_model))
        self.position_embeddings = np.random.randn(max_len, d_model) * scale
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional encodings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Position embeddings (seq_len, d_model)
        """
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max {self.max_len}"
        return self.position_embeddings[:seq_len, :]


class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes position by rotating query and key vectors.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            base: Base for frequency calculation
        """
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute rotation frequencies
        self.freqs = self._create_frequencies()
    
    def _create_frequencies(self) -> np.ndarray:
        """Create rotation frequencies."""
        # theta_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.d_model, 2) / self.d_model))
        return inv_freq
    
    def _create_rotation_matrix(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rotation matrices for sin and cos.
        
        Returns:
            sin_cached, cos_cached: (seq_len, d_model/2)
        """
        # Position indices
        positions = np.arange(seq_len)
        
        # Compute m * theta for each position and frequency
        # (seq_len, 1) * (d_model/2,) -> (seq_len, d_model/2)
        freqs = positions[:, np.newaxis] * self.freqs[np.newaxis, :]
        
        # Compute sin and cos
        sin_cached = np.sin(freqs)
        cos_cached = np.cos(freqs)
        
        return sin_cached, cos_cached
    
    def apply_rotary_embedding(self, x: np.ndarray) -> np.ndarray:
        """
        Apply rotary embedding to input.
        
        Args:
            x: Input tensor (seq_len, d_model)
            
        Returns:
            Rotated tensor (seq_len, d_model)
        """
        seq_len, d_model = x.shape
        assert d_model == self.d_model
        
        # Get rotation matrices
        sin_cached, cos_cached = self._create_rotation_matrix(seq_len)
        
        # Split into pairs
        x_pairs = x.reshape(seq_len, -1, 2)  # (seq_len, d_model/2, 2)
        
        # Extract x0 and x1 from each pair
        x0 = x_pairs[:, :, 0]  # (seq_len, d_model/2)
        x1 = x_pairs[:, :, 1]  # (seq_len, d_model/2)
        
        # Apply rotation
        # [x0]   [cos  -sin] [x0]   [x0*cos - x1*sin]
        # [x1] = [sin   cos] [x1] = [x0*sin + x1*cos]
        
        rotated_x0 = x0 * cos_cached - x1 * sin_cached
        rotated_x1 = x0 * sin_cached + x1 * cos_cached
        
        # Recombine pairs
        rotated = np.stack([rotated_x0, rotated_x1], axis=-1)
        rotated = rotated.reshape(seq_len, d_model)
        
        return rotated


class ALiBiPositionalBias:
    """
    Attention with Linear Biases (ALiBi).
    
    Adds bias to attention scores based on distance.
    """
    
    def __init__(self, num_heads: int):
        """
        Initialize ALiBi.
        
        Args:
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads
        
        # Compute head-specific slopes
        self.slopes = self._get_slopes()
    
    def _get_slopes(self) -> np.ndarray:
        """
        Get head-specific slopes.
        
        Returns:
            slopes: (num_heads,)
        """
        # m_h = 2^(-8h/H)
        slopes = []
        for h in range(1, self.num_heads + 1):
            slope = 2 ** (-8 * h / self.num_heads)
            slopes.append(slope)
        return np.array(slopes)
    
    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Get ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        # Create distance matrix
        positions = np.arange(seq_len)
        distance = positions[np.newaxis, :] - positions[:, np.newaxis]
        distance = np.abs(distance)  # (seq_len, seq_len)
        
        # Apply head-specific slopes
        # (num_heads, 1, 1) * (1, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
        bias = -self.slopes[:, np.newaxis, np.newaxis] * distance[np.newaxis, :, :]
        
        return bias


# ============================================================================
# PART 4: COMPLETE PIPELINE
# ============================================================================

class TextProcessor:
    """
    Complete text processing pipeline.
    
    Combines tokenization, embedding, and positional encoding.
    """
    
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 512,
        pos_encoding_type: str = "sinusoidal"
    ):
        """
        Initialize text processor.
        
        Args:
            tokenizer: Tokenizer instance
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            pos_encoding_type: Type of positional encoding
                ("sinusoidal", "learned", "rope", or "none")
        """
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Initialize embedding layer
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        
        # Initialize positional encoding
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
        elif pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(max_seq_len, embedding_dim)
        elif pos_encoding_type == "rope":
            self.pos_encoding = RotaryPositionalEmbedding(embedding_dim, max_seq_len)
        else:
            self.pos_encoding = None
    
    def process(self, text: str) -> np.ndarray:
        """
        Process text through complete pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Final embeddings (seq_len, embedding_dim)
        """
        # Step 1: Tokenization
        token_ids = self.tokenizer.encode(text)
        token_ids = np.array(token_ids)
        
        # Step 2: Token embedding
        token_embeds = self.token_embedding.forward(token_ids)
        
        # Step 3: Add positional encoding
        if self.pos_encoding_type == "rope":
            # RoPE applies rotation directly to embeddings
            final_embeds = self.pos_encoding.apply_rotary_embedding(token_embeds)
        elif self.pos_encoding is not None:
            # Sinusoidal or learned: add to token embeddings
            pos_embeds = self.pos_encoding.forward(len(token_ids))
            final_embeds = token_embeds + pos_embeds
        else:
            final_embeds = token_embeds
        
        return final_embeds
    
    def analyze_text(self, text: str):
        """Print detailed analysis of text processing."""
        print("=" * 80)
        print("TEXT PROCESSING ANALYSIS")
        print("=" * 80)
        
        print(f"\nOriginal text: '{text}'")
        
        # Tokenization
        token_ids = self.tokenizer.encode(text)
        print(f"\nStep 1: Tokenization")
        print(f"  Token IDs: {token_ids}")
        print(f"  Number of tokens: {len(token_ids)}")
        
        if hasattr(self.tokenizer, 'decode'):
            decoded = self.tokenizer.decode(token_ids)
            print(f"  Decoded: '{decoded}'")
        
        # Token embeddings
        token_embeds = self.token_embedding.forward(np.array(token_ids))
        print(f"\nStep 2: Token Embeddings")
        print(f"  Shape: {token_embeds.shape}")
        print(f"  First token embedding (first 8 dims): {token_embeds[0, :8]}")
        
        # Positional encoding
        if self.pos_encoding is not None:
            print(f"\nStep 3: Positional Encoding ({self.pos_encoding_type})")
            
            if self.pos_encoding_type != "rope":
                pos_embeds = self.pos_encoding.forward(len(token_ids))
                print(f"  Shape: {pos_embeds.shape}")
                print(f"  Position 0 encoding (first 8 dims): {pos_embeds[0, :8]}")
            else:
                print(f"  RoPE applies rotation directly to embeddings")
        
        # Final result
        final_embeds = self.process(text)
        print(f"\nFinal Output:")
        print(f"  Shape: {final_embeds.shape}")
        print(f"  First token (first 8 dims): {final_embeds[0, :8]}")
        print()


# ============================================================================
# PART 5: EXAMPLES AND DEMONSTRATIONS
# ============================================================================

def demo_tokenization():
    """Demonstrate different tokenization methods."""
    print("\n" + "=" * 80)
    print("TOKENIZATION DEMONSTRATION")
    print("=" * 80)
    
    texts = [
        "Hello world",
        "Natural language processing",
        "Transformer models are powerful"
    ]
    
    # Character tokenizer
    print("\n1. Character-Level Tokenization")
    print("-" * 80)
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.fit(texts)
    
    test_text = "Hello"
    ids = char_tokenizer.encode(test_text)
    decoded = char_tokenizer.decode(ids)
    print(f"Text: '{test_text}'")
    print(f"Token IDs: {ids}")
    print(f"Decoded: '{decoded}'")
    
    # Word tokenizer
    print("\n2. Word-Level Tokenization")
    print("-" * 80)
    word_tokenizer = SimpleWordTokenizer()
    word_tokenizer.fit(texts)
    
    test_text = "hello world"
    ids = word_tokenizer.encode(test_text)
    decoded = word_tokenizer.decode(ids)
    print(f"Text: '{test_text}'")
    print(f"Token IDs: {ids}")
    print(f"Decoded: '{decoded}'")
    
    # BPE tokenizer
    print("\n3. BPE Tokenization")
    print("-" * 80)
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    training_texts = texts * 10  # Repeat for better training
    bpe_tokenizer.fit(training_texts)
    
    test_text = "hello world"
    ids = bpe_tokenizer.encode(test_text)
    decoded = bpe_tokenizer.decode(ids)
    print(f"Text: '{test_text}'")
    print(f"Token IDs: {ids}")
    print(f"Decoded: '{decoded}'")


def demo_embeddings():
    """Demonstrate embedding techniques."""
    print("\n" + "=" * 80)
    print("EMBEDDING DEMONSTRATION")
    print("=" * 80)
    
    vocab_size = 100
    embedding_dim = 16
    
    # Token embedding
    print("\n1. Token Embeddings")
    print("-" * 80)
    embeddings = TokenEmbedding(vocab_size, embedding_dim)
    
    token_ids = np.array([5, 10, 15])
    embeds = embeddings.forward(token_ids)
    
    print(f"Token IDs: {token_ids}")
    print(f"Embeddings shape: {embeds.shape}")
    print(f"Token 5 embedding: {embeds[0]}")
    
    # Similarity
    sim = embeddings.similarity(5, 10)
    print(f"Similarity(5, 10): {sim:.4f}")


def demo_positional_encoding():
    """Demonstrate positional encoding methods."""
    print("\n" + "=" * 80)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 80)
    
    d_model = 64
    seq_len = 10
    
    # Sinusoidal
    print("\n1. Sinusoidal Positional Encoding")
    print("-" * 80)
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model)
    sinusoidal_pe.visualize_frequencies(positions=seq_len)
    
    # Learned
    print("\n2. Learned Positional Encoding")
    print("-" * 80)
    learned_pe = LearnedPositionalEncoding(max_len=100, d_model=d_model)
    pos_emb = learned_pe.forward(seq_len)
    print(f"Learned positional embeddings shape: {pos_emb.shape}")
    print(f"Position 0 (first 8 dims): {pos_emb[0, :8]}")
    
    # RoPE
    print("\n3. Rotary Positional Embedding (RoPE)")
    print("-" * 80)
    rope = RotaryPositionalEmbedding(d_model)
    
    # Sample input
    x = np.random.randn(seq_len, d_model)
    rotated_x = rope.apply_rotary_embedding(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Rotated shape: {rotated_x.shape}")
    print(f"Position 0 before rotation (first 8 dims): {x[0, :8]}")
    print(f"Position 0 after rotation (first 8 dims): {rotated_x[0, :8]}")
    
    # ALiBi
    print("\n4. ALiBi Positional Bias")
    print("-" * 80)
    num_heads = 4
    alibi = ALiBiPositionalBias(num_heads)
    
    bias = alibi.get_bias(seq_len)
    print(f"Bias shape: {bias.shape}")
    print(f"Head slopes: {alibi.slopes}")
    print(f"\nBias matrix for head 0:")
    print(bias[0])


def demo_complete_pipeline():
    """Demonstrate complete text processing pipeline."""
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Prepare data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Natural language processing with deep learning",
        "Transformers revolutionized machine learning"
    ]
    
    # Train tokenizer
    tokenizer = SimpleWordTokenizer()
    tokenizer.fit(texts)
    
    # Create processor
    processor = TextProcessor(
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        max_seq_len=128,
        pos_encoding_type="sinusoidal"
    )
    
    # Process text
    test_text = "The quick brown fox"
    processor.analyze_text(test_text)
    
    # Try different positional encodings
    for pos_type in ["sinusoidal", "learned", "rope", "none"]:
        print(f"\n{'=' * 80}")
        print(f"Positional Encoding: {pos_type.upper()}")
        print('=' * 80)
        
        processor_variant = TextProcessor(
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=64,
            pos_encoding_type=pos_type
        )
        
        result = processor_variant.process(test_text)
        print(f"Output shape: {result.shape}")
        print(f"First token (first 8 dims): {result[0, :8]}")


def visualize_positional_encoding_patterns():
    """Visualize how positional encodings differ across positions."""
    print("\n" + "=" * 80)
    print("POSITIONAL ENCODING PATTERN VISUALIZATION")
    print("=" * 80)
    
    d_model = 8
    seq_len = 10
    
    # Sinusoidal
    pe = SinusoidalPositionalEncoding(d_model)
    encodings = pe.forward(seq_len)
    
    print("\nSinusoidal Positional Encoding Matrix:")
    print("(Rows = positions, Columns = dimensions)")
    print("-" * 80)
    print("Pos |", end="")
    for d in range(d_model):
        print(f"  D{d}  |", end="")
    print()
    print("-" * 80)
    
    for pos in range(seq_len):
        print(f" {pos:2d} |", end="")
        for d in range(d_model):
            print(f"{encodings[pos, d]:6.3f}|", end="")
        print()
    
    print("\nObservations:")
    print("- Even dimensions (0,2,4,6) use sine")
    print("- Odd dimensions (1,3,5,7) use cosine")
    print("- Lower dimensions change faster (high frequency)")
    print("- Higher dimensions change slower (low frequency)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TOKENIZATION, EMBEDDINGS, AND POSITIONAL ENCODING")
    print("Complete Implementation and Demonstration")
    print("=" * 80)
    
    # Run all demonstrations
    demo_tokenization()
    demo_embeddings()
    demo_positional_encoding()
    visualize_positional_encoding_patterns()
    demo_complete_pipeline()
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
