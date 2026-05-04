
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

class RotaryEmbedding1D(nn.Module):
    """
    Minimal 1D RoPE that applies rotation along the last feature dimension.
    Expect last dimension (d) to be even; will rotate in 2D pairs.
    seq_len = length along sequence axis where positions are defined.
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10_000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = base
        self.register_buffer(
            "theta",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        # positions: [seq_len, 1]
        positions = torch.arange(max_seq_len).float().unsqueeze(1)  # [S,1]
        # freqs: [S, dim/2]
        freqs = positions * self.theta.unsqueeze(0)                 # [S, D/2]
        # precompute cos/sin
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)  # [S, D/2]
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)  # [S, D/2]

    def forward(self, x, positions=None):
        """
        x: [..., S, D], where D == self.dim and S <= max_seq_len
        positions: optional LongTensor shape [..., S] for custom position ids
        Returns same shape [..., S, D] with rotary applied on last dim.
        """
        *prefix, S, D = x.shape
        assert D == self.dim
        if S > self.max_seq_len:
            self._build_cache(S)

        # reshape features into pairs
        x = x.view(*prefix, S, D // 2, 2)  # [..., S, D/2, 2]
        x1, x2 = x[..., 0], x[..., 1]      # each [..., S, D/2]

        if positions is None:
            cos = self.cos_cached[:S]      # [S, D/2]
            sin = self.sin_cached[:S]      # [S, D/2]
            # broadcast cos/sin to prefix
            for _ in prefix:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
            # gather cos/sin for provided positions
            # positions shape: [..., S]
            cos = torch.take_along_dim(self.cos_cached, positions.unsqueeze(-1), dim=0)  # [..., S, D/2]
            sin = torch.take_along_dim(self.sin_cached, positions.unsqueeze(-1), dim=0)  # [..., S, D/2]

        # rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1r = x1 * cos - x2 * sin
        x2r = x1 * sin + x2 * cos

        out = torch.stack([x1r, x2r], dim=-1).reshape(*prefix, S, D)  # [..., S, D]
        return out

class PositionalEncoding(nn.Module):
    """Module for different positional encoding strategies
    
    Currently supported:
    - learned: learned positional embeddings
    - sinusoidal: fixed sinusoidal embeddings
    - rope: RoPE positional encoding
    
    Returns:
        - x: [batch, seq_len, embed_dim]
    """
    def __init__(self, embed_dim, max_len=5000, encoding_type='none'):
        super().__init__()
        self.encoding_type = encoding_type
        self.embed_dim = embed_dim
        
        if encoding_type == 'learned':
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))
            nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
            
        elif encoding_type == 'sinusoidal':
            # Fixed sinusoidal embeddings like original Transformer
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        
        elif encoding_type == 'rope':
            self.rope = RotaryEmbedding1D(dim=embed_dim, max_seq_len=max_len)
        
        else:
            raise ValueError(f"Invalid encoding type: {encoding_type}")

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape
        
        if self.encoding_type == 'none':
            return x
            
        if self.encoding_type == 'rope':
            # apply RoPE to the input
            return self.rope(x)
        
        if self.encoding_type == 'learned':
            # Add learned positional embeddings
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            return x + pos_emb
            
        elif self.encoding_type == 'sinusoidal':
            # Add fixed sinusoidal positional embeddings
            pos_emb = self.pe[:, :seq_len].expand(batch_size, -1, -1)
            return x + pos_emb

class TrajectoryEncoder(nn.Module):
    """
    Flexible trajectory encoder with options for positional encoding.
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 5,
        cnn_stride: int = 1,
        cnn_padding: int = 2,
        rope_dim: int = 64,
        max_seq_len: int = 128,
        pooling_type: str = "mean",
        use_displacements: bool = False,
        pos_encoding: str = "rope"  # Options: "none", "rope", "learned", "sinusoidal"
    ):
        super().__init__()
        assert rope_dim % 2 == 0 if pos_encoding == "rope" else True, "rope_dim must be even when using RoPE"
        assert pooling_type in ["mean", "max", "attention"], "pooling_type must be 'mean', 'max', or 'attention'"
        assert pos_encoding in ["none", "rope", "learned", "sinusoidal"], "Invalid pos_encoding option"
        
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        self.use_displacements = use_displacements
        self.pos_encoding = pos_encoding

        # 1D CNN over (L=60) with input channels=2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=cnn_channels,
                      kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=rope_dim if pos_encoding == "rope" else embedding_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(rope_dim if pos_encoding == "rope" else embedding_dim),
            nn.GELU()
        )
        
        # Only initialize RoPE if selected
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding1D(dim=rope_dim, max_seq_len=max_seq_len)
        else:
            self.rope = None
            
        # For non-RoPE positional encodings
        if pos_encoding in ["learned", "sinusoidal"]:
            self.pos_encoder = PositionalEncoding(
                embed_dim=embedding_dim, 
                max_len=max_seq_len, 
                encoding_type=pos_encoding
            )
        else:
            self.pos_encoder = None

        # Attention-based pooling components
        if pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(embedding_dim // 2, 1)
            )

        # If not using RoPE, we need a different projection if cnn output ≠ embedding_dim
        cnn_out_dim = rope_dim if pos_encoding == "rope" else embedding_dim
        if pos_encoding != "rope" and cnn_out_dim != embedding_dim:
            self.projection = nn.Linear(cnn_out_dim, embedding_dim)
        else:
            self.projection = None

        # If using RoPE, we still need an MLP after pooling
        if pos_encoding == "rope":
            self.mlp = nn.Sequential(
                nn.Linear(rope_dim, 2 * embedding_dim),
                nn.GELU(),
                nn.Linear(2 * embedding_dim, embedding_dim)
            )
        else:
            self.mlp = None

    def _compute_displacements(self, trajectories):
        """Compute displacement vectors between consecutive trajectory points."""
        displacements = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
        zero_pad = torch.zeros(trajectories.size(0), trajectories.size(1), 1, 2, 
                              device=trajectories.device, dtype=trajectories.dtype)
        return torch.cat([zero_pad, displacements], dim=2)

    def forward(self, x):
        n_target, n_trajectories, L, C = x.shape
        assert L == 60 and C == 2, "Expected last two dims to be [60, 2]"

        # Option to compute displacements instead of using raw positions
        if self.use_displacements:
            x = self._compute_displacements(x)

        # Merge dimensions for CNN processing
        x = x.reshape(n_target * n_trajectories, L, C)
        x = x.permute(0, 2, 1)  # [batch, 2, 60]

        # Apply CNN
        x = self.cnn(x)  # [batch, rope_dim, 60]
        x = x.permute(0, 2, 1)  # [batch, 60, rope_dim]

        # Apply positional encoding based on selected strategy
        if self.pos_encoding == "rope":
            x = self.rope(x)  # Apply RoPE
        elif self.pos_encoding in ["learned", "sinusoidal"]:
            # Project to embedding_dim if needed
            if self.projection is not None:
                x = self.projection(x)
            # Apply positional encoding
            x = self.pos_encoder(x)
        # If "none", we leave x as is

        # Pool over sequence
        if self.pooling_type == "mean":
            x = x.mean(dim=1)
        elif self.pooling_type == "max":
            x, _ = x.max(dim=1)
        elif self.pooling_type == "attention":
            attn_weights = F.softmax(self.attention_pool(x), dim=1)
            x = (x * attn_weights).sum(dim=1)

        # If using RoPE, apply final MLP projection
        if self.pos_encoding == "rope":
            x = self.mlp(x)

        # Reshape to final output
        x = x.view(n_target, n_trajectories, self.embedding_dim)
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Simple learned positional encoding.
    Each position index in the sequence has a trainable vector.
    """
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pos_embedding[:L, :]


class TrajectoryEncoder_DualPath(nn.Module):
    """
    Dual-path Trajectory Encoder:
    - Path 1: Raw positions + learned positional encoding
    - Path 2: Displacements + learned positional encoding
    - Merge via concatenation and linear projection to embedding_dim
    """
    def __init__(
        self,
        seq_len: int = 60,
        embedding_dim: int = 128,
        cnn_channels: int = 64,
        pooling_type: str = "attention"
    ):
        super().__init__()
        assert pooling_type in ["mean", "max", "attention"]

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type

        # CNN for raw positions
        self.cnn_raw = nn.Sequential(
            nn.Conv1d(2, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU()
        )
        self.pos_enc_raw = LearnedPositionalEncoding(seq_len, embedding_dim)

        # CNN for displacements
        self.cnn_disp = nn.Sequential(
            nn.Conv1d(2, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU()
        )
        self.pos_enc_disp = LearnedPositionalEncoding(seq_len, embedding_dim)

        # Merge raw + displacement features
        self.merge_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU()
        )

        # Optional attention pooling
        if pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(embedding_dim // 2, 1)
            )

    def _compute_displacements(self, x):
        """
        Compute displacement vectors between consecutive points.
        Pads with zero at first position for shape consistency.
        x: [B, L, 2]
        """
        disp = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros(x.size(0), 1, 2, device=x.device, dtype=x.dtype)
        return torch.cat([pad, disp], dim=1)

    def forward(self, x):
        """
        x: [n_scene*n_agents, n_anchor_traj, seq_len, 2]
        Returns: [n_scene*n_agents, n_anchor_traj, embedding_dim]
        """
        assert x.shape[2] == 1, f"Expected input shape: [n_scene*n_agents, n_anchor_traj, K=1, seq_len, 2], but got {x.shape}"
        x = einops.rearrange(x, 'n_batch n_proto_traj K seq_len C -> n_batch (n_proto_traj K) seq_len C')
        n_target, n_traj, L, C = x.shape
        assert L == self.seq_len and C == 2, f"Expected input shape: [n_target, n_traj, {self.seq_len}, 2], but got {x.shape}"

        # Flatten anchor trajs for batch processing
        x_flat = x.view(n_target * n_traj, L, C)

        # Path 1: Raw positions
        raw_feat = self.cnn_raw(x_flat.permute(0, 2, 1))  # [B, D, L]
        raw_feat = raw_feat.permute(0, 2, 1)              # [B, L, D]
        raw_feat = self.pos_enc_raw(raw_feat)

        # Path 2: Displacements
        disp_flat = self._compute_displacements(x_flat)
        disp_feat = self.cnn_disp(disp_flat.permute(0, 2, 1))
        disp_feat = disp_feat.permute(0, 2, 1)
        disp_feat = self.pos_enc_disp(disp_feat)

        # Merge features
        merged_feat = torch.cat([raw_feat, disp_feat], dim=-1)
        merged_feat = self.merge_proj(merged_feat)  # [B, L, D]

        # Pooling
        if self.pooling_type == "mean":
            pooled = merged_feat.mean(dim=1)
        elif self.pooling_type == "max":
            pooled, _ = merged_feat.max(dim=1)
        elif self.pooling_type == "attention":
            attn_weights = F.softmax(self.attention_pool(merged_feat), dim=1)
            pooled = (merged_feat * attn_weights).sum(dim=1)

        return pooled.view(n_target, n_traj, self.embedding_dim) # [B, T, 256]


if __name__ == "__main__":
    n_target = 32
    n_trajectories = 16
    embedding_dim = 128

    model = TrajectoryEncoder_DualPath(seq_len=60, 
                                       embedding_dim=128, 
                                       pooling_type="attention")

    x = torch.randn(n_target, n_trajectories, 60, 2)
    print(f"Input shape:{x.shape}")

    y= model(x)
    print(f"Output shape: {y.shape}")