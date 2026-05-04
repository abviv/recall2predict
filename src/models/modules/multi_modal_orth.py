# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor, nn
from HPTR.src.models.modules.mlp import MLP


class MultiModalAnchors(nn.Module):
    def __init__(
        self,
        mode_emb: str,
        mode_init: str,
        hidden_dim: int,
        n_pred: int,
        emb_dim: int,
        use_agent_type: bool,
        scale: float = 1.0,
        enforce_orthogonality: bool = False,
    ) -> None:
        super().__init__()
        self.n_pred = n_pred
        self.use_agent_type = use_agent_type
        self.enforce_orthogonality = enforce_orthogonality

        self.mode_init = mode_init
        n_anchors = 3 if use_agent_type else 1
        
        if self.mode_init == "xavier":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            nn.init.xavier_normal_(self.anchors)
            self.anchors = nn.Parameter(self.anchors * scale, requires_grad=True)
        elif self.mode_init == "uniform":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            self.anchors.uniform_(-scale, scale)
            self.anchors = nn.Parameter(self.anchors, requires_grad=True)
        elif self.mode_init == "randn":
            self.anchors = nn.Parameter(torch.randn([n_anchors, n_pred, hidden_dim]) * scale, requires_grad=True)
        elif self.mode_init == "orthogonal":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            for i in range(n_anchors):
                nn.init.orthogonal_(self.anchors[i])
            self.anchors = nn.Parameter(self.anchors * scale, requires_grad=True)
        elif self.mode_init=="qr_orthogonal":
            self.anchors = self._create_orthogonal_anchors(n_anchors, n_pred, hidden_dim, scale)
            self.anchors = nn.Parameter(self.anchors, requires_grad=True)
        else:
            raise NotImplementedError

        self.mode_emb = mode_emb
        if self.mode_emb == "linear":
            self.mlp_anchor = nn.Linear(self.anchors.shape[-1] + emb_dim, hidden_dim, bias=False)
        elif self.mode_emb == "mlp":
            self.mlp_anchor = MLP([self.anchors.shape[-1] + emb_dim] + [hidden_dim] * 2, end_layer_activation=False)
        elif self.mode_emb == "add" or self.mode_emb == "none":
            assert emb_dim == hidden_dim
            if self.anchors.shape[-1] != hidden_dim:
                self.mlp_anchor = nn.Linear(self.anchors.shape[-1], hidden_dim, bias=False)
            else:
                self.mlp_anchor = None
        else:
            raise NotImplementedError

    def _create_orthogonal_anchors(self, n_anchors: int, n_pred: int, hidden_dim: int, scale: float) -> torch.Tensor:
        """
        Create orthogonal anchor vectors using QR decomposition.
        
        Args:
            n_anchors: Number of anchor groups (usually 1 or 3)
            n_pred: Number of predictions/queries
            hidden_dim: Embedding dimension
            scale: Scaling factor
            
        Returns:
            Orthogonal anchor tensor of shape [n_anchors, n_pred, hidden_dim]
        """
        anchors = torch.empty((n_anchors, n_pred, hidden_dim))
        
        for i in range(n_anchors):
            if n_pred <= hidden_dim:
                # Standard case: more dimensions than vectors
                # Generate random matrix and orthogonalize
                random_matrix = torch.randn(hidden_dim, n_pred)
                q, _ = torch.linalg.qr(random_matrix)
                anchors[i] = q[:, :n_pred].t() * scale  # [n_pred, hidden_dim]
            else:
                # More vectors than dimensions: use multiple orthogonal subspaces
                # Divide n_pred into groups that fit in hidden_dim
                anchors_i = []
                remaining = n_pred
                start_idx = 0
                
                while remaining > 0:
                    current_batch = min(remaining, hidden_dim)
                    random_matrix = torch.randn(hidden_dim, current_batch)
                    q, _ = torch.linalg.qr(random_matrix)
                    anchors_i.append(q[:, :current_batch].t())  # [current_batch, hidden_dim]
                    remaining -= current_batch
                    start_idx += current_batch
                
                anchors[i] = torch.cat(anchors_i, dim=0) * scale  # [n_pred, hidden_dim]
                
        return anchors

    def orthogonalize_anchors(self) -> None:
        """
        Orthogonalize the current anchor parameters in-place using QR decomposition.
        This can be called during training to maintain orthogonality.
        """
        with torch.no_grad():
            n_anchors, n_pred, hidden_dim = self.anchors.shape
            
            for i in range(n_anchors):
                if n_pred <= hidden_dim:
                    # QR decomposition for orthogonalization
                    q, _ = torch.linalg.qr(self.anchors[i].t())  # [hidden_dim, n_pred]
                    self.anchors[i].copy_(q[:, :n_pred].t())  # [n_pred, hidden_dim]
                else:
                    # Handle case where n_pred > hidden_dim
                    # Orthogonalize in chunks
                    start_idx = 0
                    while start_idx < n_pred:
                        end_idx = min(start_idx + hidden_dim, n_pred)
                        chunk = self.anchors[i, start_idx:end_idx]  # [chunk_size, hidden_dim]
                        q, _ = torch.linalg.qr(chunk.t())  # [hidden_dim, chunk_size]
                        self.anchors[i, start_idx:end_idx].copy_(q[:, :end_idx-start_idx].t())
                        start_idx = end_idx

    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality loss to encourage orthogonal anchors during training.
        
        Returns:
            Orthogonality loss scalar
        """
        n_anchors, n_pred, hidden_dim = self.anchors.shape
        total_loss = 0.0
        
        for i in range(n_anchors):
            if n_pred <= hidden_dim:
                # Compute gram matrix A^T A
                gram = torch.mm(self.anchors[i], self.anchors[i].t())  # [n_pred, n_pred]
                # Orthogonality loss: ||A^T A - I||_F^2
                identity = torch.eye(n_pred, device=self.anchors.device)
                loss = torch.norm(gram - identity, p='fro') ** 2
                total_loss += loss
            else:
                # For n_pred > hidden_dim, use different approach
                # Minimize correlation between different anchor vectors
                anchors_normalized = torch.nn.functional.normalize(self.anchors[i], dim=-1)
                correlation_matrix = torch.mm(anchors_normalized, anchors_normalized.t())
                # Remove diagonal (self-correlation)
                correlation_matrix = correlation_matrix - torch.eye(n_pred, device=self.anchors.device)
                loss = torch.norm(correlation_matrix, p='fro') ** 2
                total_loss += loss
                
        return total_loss / n_anchors

    def forward(self, valid: Tensor, emb: Tensor, agent_type: Tensor) -> Tensor:
        """
        This module can either generate latent anchors or use them as is.
        If mode_emb is "linear" or "mlp", the anchors are concatenated with the emb and passed through an MLP to generate the final embeddings.
        If mode_emb is "add", the anchors are added to the emb.
        If mode_emb is "none", the anchors are used to generate latent queries.
        
        Args:
            valid: [n_scene*n_agent]
            emb: [n_scene*n_agent, in_dim]
            agent_type: [n_scene*n_agent, 3]

        Returns:
            mm_emb: [n_scene*n_agent, n_pred, out_dim]
        """
        # Optionally enforce orthogonality during forward pass
        if self.enforce_orthogonality and self.training:
            self.orthogonalize_anchors()
        
        # [n_scene*n_agent, n_pred, emb_dim]
        if self.use_agent_type:
            anchors = (self.anchors.unsqueeze(0) * agent_type[:, :, None, None]).sum(1)
        else:
            anchors = self.anchors.expand(valid.shape[0], -1, -1)

        if self.mode_emb == "linear" or self.mode_emb == "mlp":
            # [n_scene*n_agent, n_pred, hidden_dim + emb_dim]
            mm_emb = torch.cat([emb.unsqueeze(1).expand(-1, self.n_pred, -1), anchors], dim=-1)
            mm_emb = self.mlp_anchor(mm_emb)
        elif self.mode_emb == "add":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)  # [n_scene*n_agent, n_pred, hidden_dim]
            mm_emb = emb.unsqueeze(1) + anchors
        elif self.mode_emb == "none":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)  # [n_scene*n_agent, n_pred, hidden_dim]
            mm_emb = anchors
        
        return mm_emb.masked_fill(~valid[:, None, None], 0)