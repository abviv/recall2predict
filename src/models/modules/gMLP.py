import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        # Split channel dimension in half
        self.norm = nn.LayerNorm(d_model // 2)
        # Spatial projection (learnable) along the spatial dimension
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        
        
        # Initialize weights to near-zero and biases to 1
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.ones_(self.spatial_proj.bias)
        
    
    def forward(self, x):
        # x: [B, seq_len, d_model]
        u, v = x.chunk(2, dim=-1)  # Split into two halves
        v = self.norm(v)
        
        # Project along spatial dimension
        v = v.transpose(1, 2) # [B, d_model // 2, seq_len]
        v = self.spatial_proj(v) # [B, d_model // 2, seq_len]
        v = v.transpose(1, 2) # [B, seq_len, d_model // 2]
        
        return u * v  # Gating via element-wise multiplication

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gate = SpatialGatingUnit(d_ffn, seq_len) 
        # Output projection takes the halved dimension from SGU
        self.output_proj = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        shortcut = x # [B, seq_len, d_model]
        x = self.norm(x) 
        x = self.channel_proj(x) # [B, seq_len, d_ffn]
        x = self.activation(x) # [B, seq_len, d_ffn]
        x = self.spatial_gate(x) # [B, seq_len, d_ffn // 2]
        x = self.output_proj(x) # [B, seq_len, d_model]
        return x + shortcut  # Residual connection


if __name__ == '__main__':
    # Usage
    seq_len = 50
    B = 4
    x = torch.randn(B*4, seq_len, 128)  # [Batch, Sequence, Features]
    mlp_block = gMLPBlock(d_model=128, d_ffn=256, seq_len=seq_len)
    output = mlp_block(x)
    print(output.shape)  # torch.Size([4, 50, 128])