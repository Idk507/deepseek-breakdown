"""
Using KV Cache with HuggingFace Transformers
Demonstrates practical usage with real models like GPT-2
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def compare_generation_with_without_cache():
    """
    Compare text generation speed with and without KV cache using GPT-2.
    """
    print("=" * 80)
    print("HuggingFace KV Cache Comparison")
    print("=" * 80)
    
    # Load model and tokenizer
    model_name = "gpt2"  # Small model for quick demo
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Device: {device}")
    
    # Prepare input
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    max_new_tokens = 50
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_new_tokens} new tokens...\n")
    
    # ============================================
    # Method 1: WITHOUT KV Cache
    # ============================================
    print("-" * 80)
    print("Method 1: WITHOUT KV Cache (use_cache=False)")
    print("-" * 80)
    
    with torch.no_grad():
        start_time = time.time()
        
        generated_ids = input_ids.clone()
        
        for i in range(max_new_tokens):
            # Forward pass WITHOUT cache - must process entire sequence
            outputs = model(generated_ids, use_cache=False)
            logits = outputs.logits
            
            # Get next token (greedy decoding)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1} tokens, processing {generated_ids.shape[1]} tokens each step")
        
        end_time = time.time()
        time_without_cache = end_time - start_time
        
        text_without_cache = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\nTime taken: {time_without_cache:.3f} seconds")
    print(f"Generated text:\n  {text_without_cache}\n")
    
    # ============================================
    # Method 2: WITH KV Cache
    # ============================================
    print("-" * 80)
    print("Method 2: WITH KV Cache (use_cache=True)")
    print("-" * 80)
    
    with torch.no_grad():
        start_time = time.time()
        
        generated_ids = input_ids.clone()
        past_key_values = None  # Initialize cache
        
        for i in range(max_new_tokens):
            # Forward pass WITH cache
            if past_key_values is None:
                # First iteration: process entire prompt
                outputs = model(generated_ids, use_cache=True)
            else:
                # Subsequent iterations: only process last token
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values  # Update cache
            
            # Get next token
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1} tokens, processing 1 token each step (cache size: {i + 1 + input_ids.shape[1]})")
        
        end_time = time.time()
        time_with_cache = end_time - start_time
        
        text_with_cache = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\nTime taken: {time_with_cache:.3f} seconds")
    print(f"Generated text:\n  {text_with_cache}\n")
    
    # ============================================
    # Method 3: Using HuggingFace generate() method (uses cache by default)
    # ============================================
    print("-" * 80)
    print("Method 3: Using .generate() method (automatic cache handling)")
    print("-" * 80)
    
    with torch.no_grad():
        start_time = time.time()
        
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            use_cache=True  # This is default
        )
        
        end_time = time.time()
        time_with_generate = end_time - start_time
        
        text_with_generate = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\nTime taken: {time_with_generate:.3f} seconds")
    print(f"Generated text:\n  {text_with_generate}\n")
    
    # ============================================
    # Summary
    # ============================================
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Without KV Cache: {time_without_cache:.3f}s")
    print(f"With KV Cache:    {time_with_cache:.3f}s")
    print(f"Using .generate():{time_with_generate:.3f}s")
    print(f"\nSpeedup (cache vs no cache): {time_without_cache / time_with_cache:.2f}x")
    print(f"Speedup (cache vs generate): {time_without_cache / time_with_generate:.2f}x")
    print("\nNote: .generate() includes additional optimizations beyond KV cache")
    print("=" * 80)


def inspect_kv_cache_structure():
    """
    Inspect the structure of past_key_values returned by HuggingFace models.
    """
    print("\n" + "=" * 80)
    print("KV Cache Structure in HuggingFace Models")
    print("=" * 80)
    
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Create sample input
    input_text = "Hello, world!"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    print(f"\nModel: {model_name}")
    print(f"Input text: '{input_text}'")
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    print(f"\nCache structure:")
    print(f"  Type: {type(past_key_values)}")
    print(f"  Number of layers: {len(past_key_values)}")
    print(f"  Each layer contains: tuple of (key_cache, value_cache)")
    
    # Inspect first layer
    layer_0_key, layer_0_value = past_key_values[0]
    print(f"\nLayer 0 cache:")
    print(f"  Key shape:   {layer_0_key.shape}   (batch, num_heads, seq_len, head_dim)")
    print(f"  Value shape: {layer_0_value.shape} (batch, num_heads, seq_len, head_dim)")
    
    # Show how cache grows with new tokens
    print("\n" + "-" * 80)
    print("Cache growth demonstration:")
    print("-" * 80)
    
    new_inputs = tokenizer(" How are", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Pass previous cache
        outputs = model(
            new_inputs['input_ids'],
            past_key_values=past_key_values,
            use_cache=True
        )
    
    new_past_key_values = outputs.past_key_values
    new_layer_0_key, new_layer_0_value = new_past_key_values[0]
    
    print(f"After adding {new_inputs['input_ids'].shape[1]} more tokens:")
    print(f"  Key shape:   {new_layer_0_key.shape}")
    print(f"  Value shape: {new_layer_0_value.shape}")
    print(f"  Sequence length increased: {layer_0_key.shape[2]} -> {new_layer_0_key.shape[2]}")
    
    print("\n" + "=" * 80)


def batch_generation_with_cache():
    """
    Demonstrate batch generation with KV cache.
    """
    print("\n" + "=" * 80)
    print("Batch Generation with KV Cache")
    print("=" * 80)
    
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Multiple prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the future, humans will"
    ]
    
    print(f"\nGenerating for {len(prompts)} prompts in parallel...")
    
    # Tokenize with padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        end_time = time.time()
    
    print(f"\nTime taken: {end_time - start_time:.3f}s\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, output_ids)):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Generated: {generated_text}\n")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run all demonstrations
    compare_generation_with_without_cache()
    inspect_kv_cache_structure()
    batch_generation_with_cache()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. KV Cache Structure:
   - past_key_values is a tuple of length num_layers
   - Each element is (key_cache, value_cache) for that layer
   - Shape: [batch_size, num_heads, seq_len, head_dim]

2. Using Cache in HuggingFace:
   - Set use_cache=True in forward pass or .generate()
   - First call: Pass full input, get past_key_values
   - Subsequent calls: Pass only new token(s) + past_key_values
   
3. Performance Benefits:
   - Reduces computation from O(n²) to O(n) per token
   - Typical speedup: 2-10x depending on sequence length
   - Essential for real-time applications

4. Memory Trade-off:
   - Cache uses additional memory: O(layers × heads × seq_len × head_dim)
   - For long sequences, can use techniques like:
     * Sliding window attention
     * Cache compression
     * Offloading to CPU
    """)
    print("=" * 80)
