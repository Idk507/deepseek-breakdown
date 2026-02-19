import torch 
import torch.nn as nn
import math

class RotaryPositionalEncoding(nn.Module):
    """
    Helper module to apply Rotary Positional Encoding (RoPE).
    This is not added to the embeddings but is applied directly to
    the Query and Key vectors.
    """
    def __init__(self, d_head, max_seq_len=2048):
        super().__init__()
        # Precompute the theta values for the rotational matrix
        theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('theta', theta)
        
        # Precompute the frequency terms (m * theta) for all positions
        positions = torch.arange(max_seq_len).unsqueeze(1)
        freqs = positions * self.theta.unsqueeze(0)
        
        # Create the complex number representation for rotation
        # The real part is cos(freqs) and the imaginary part is sin(freqs)
        self.register_buffer('freqs_cis', torch.polar(torch.ones_like(freqs), freqs))

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, d_head)
        seq_len = x.shape[2]
        
        # Reshape x to treat pairs of dimensions as complex numbers
        x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
        # Convert to PyTorch complex type
        x_complex = torch.view_as_complex(x_complex)
        
        # Get the precomputed frequencies for the current sequence length
        freqs_cis = self.freqs_cis[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation by multiplying in the complex domain
        # This rotates each pair of dimensions by the angle m * theta_i
        x_rotated = x_complex * freqs_cis
        
        # Convert back to real number representation
        x_rotated = torch.view_as_real(x_rotated)
        # Reshape back to the original d_head dimension
        x_rotated = x_rotated.flatten(3)
        
        return x_rotated.type_as(x)
