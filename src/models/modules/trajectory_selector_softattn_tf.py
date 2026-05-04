import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import logging
from torch import Tensor
from typing import Optional, Dict, Tuple
from layers_in_my_way.modules.mlp import MlpS
from HPTR.src.models.modules.transformer import TransformerBlock
from src.models.modules.multi_modal_orth import MultiModalAnchors

log = logging.getLogger(__name__)


class SoftAttentionTrajectorySelector(nn.Module):
    """Selects trajectories from a motion bank with context-conditioned queries."""

    def __init__(self, 
                    loaded_embeddings=None, 
                    traj_tensor=None, 
                    n_latent_anchors=16, 
                    hidden_dim=None,
                    context_cfg: Optional[Dict[str, bool]] = None,
                    endpoint_topM: int = 64,
                    retrieval_mode: str = "endpoint_rerank",
                    selection_mode: str = "full",
                    use_straight_through: bool = True,
                    use_gating: bool = True,
                    ):
        
        super().__init__()
        if loaded_embeddings is None or traj_tensor is None:
            raise ValueError("loaded_embeddings and traj_tensor must be provided.")

        self.register_buffer("loaded_embeddings", F.normalize(loaded_embeddings, dim=-1).detach())
        self.loaded_embeddings.requires_grad_(False)
        self.register_buffer("traj_tensor", traj_tensor.detach())
        self.traj_tensor.requires_grad_(False)
        
        try:
            self.register_buffer("anchor_endpoints", traj_tensor[:, -1, :].clone().detach())
            self.anchor_endpoints.requires_grad_(False)
        except Exception:
            log.warning("Could not register anchor_endpoints; endpoint re-ranking will be disabled.")
            self.anchor_endpoints = None
        
        self.embedding_dim = loaded_embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.retrieval_mode = retrieval_mode
        self.endpoint_topM = endpoint_topM
        self.use_straight_through = use_straight_through
        self.use_gating = use_gating

        valid_modes = {"full", "random", "nearest_neighbor", "no_extras"}
        if selection_mode not in valid_modes:
            raise ValueError(f"selection_mode must be one of {valid_modes}, got '{selection_mode}'")
        self.selection_mode = selection_mode
        self.n_latent_anchors = n_latent_anchors
        
        if self.selection_mode != "full":
            log.info(f"Trajectory selector running in '{self.selection_mode}' mode (baseline/ablation)")
        if not self.use_straight_through:
            log.info("Straight-through estimator DISABLED: using soft attention weights for retrieval")
        if not self.use_gating:
            log.info("Dual-level gating DISABLED: using uniform mean of modality contributions")

        default_context_cfg = {
            "use_target": True,
            "use_other": False,
            "use_map": False,
            "fusion_mode": "attention"
        }
        if context_cfg is not None:
            resolved_cfg = {str(k): bool(v) for k, v in context_cfg.items()}
            default_context_cfg.update(resolved_cfg)
        self.context_cfg = default_context_cfg

        # Only enabled modalities participate in the importance softmax.
        active_modalities = torch.tensor([
            self.context_cfg.get("use_target", True),
            self.context_cfg.get("use_other", False),
            self.context_cfg.get("use_map", False),
        ], dtype=torch.bool)
        self.register_buffer("_active_modalities", active_modalities)
        # The null branch lets the selector skip context updates entirely.
        self.register_buffer("_null_modality", torch.tensor([True], dtype=torch.bool))

        self.latent_orth_query = MultiModalAnchors(
            mode_emb="none",
            mode_init="orthogonal",
            hidden_dim=self.embedding_dim,
            n_pred=n_latent_anchors,
            emb_dim=self.embedding_dim,
            use_agent_type=True,
            scale=5.0,
        )
        # The all-agent path attends in hidden_dim space before projecting back.
        self.latent_orth_query_256 = MultiModalAnchors(
            mode_emb="none",
            mode_init="orthogonal",
            hidden_dim=self.hidden_dim,
            n_pred=n_latent_anchors,
            emb_dim=self.hidden_dim,
            use_agent_type=False,
            scale=1.0,
        )

        self.query_target_ca = TransformerBlock(
            d_model=self.embedding_dim,
            n_head=8,
            d_feedforward=256,
            dropout_p=0.1,
            norm_first=True,
            decoder_self_attn=False
        )
        
        self.query_scene_ca = TransformerBlock(
            d_model=self.embedding_dim,
            n_head=8,
            d_feedforward=256,
            dropout_p=0.1,
            norm_first=True,
            decoder_self_attn=False
        )

        self.all_agent_ca = TransformerBlock(
            d_model=self.hidden_dim,
            n_head=8,
            d_feedforward=256,
            dropout_p=0.1,
            norm_first=True,
            decoder_self_attn=False
        )
        
        self.query_map_ca = TransformerBlock(
            d_model=self.embedding_dim,
            n_head=8,
            d_feedforward=256,
            dropout_p=0.1,
            norm_first=True,
            decoder_self_attn=False
        )

        self.target_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.scene_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.map_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        self.modality_importance = nn.Parameter(torch.ones(3) / 3)
        self.null_importance = nn.Parameter(torch.tensor(0.0))

        self.target_proj = MlpS(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.embedding_dim,
            dropout_p=0.1, 
            use_layernorm=True, 
        )
        
        self.scene_proj = MlpS(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.embedding_dim,
            dropout_p=0.1, 
            use_layernorm=True,
        )
        
        self.map_proj = MlpS(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.embedding_dim,
            dropout_p=0.1, 
            use_layernorm=True,
        )

        self.all_agents_proj = MlpS(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.embedding_dim,
            dropout_p=0.1,
            use_layernorm=True,
        )

        self.context_type_embedding = nn.Embedding(3, self.embedding_dim)

        self.output_projection = MlpS(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout_p=0.2, 
            use_layernorm=True
        )
        
        self.output_projection_adapted_queries = MlpS(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout_p=0.2, 
            use_layernorm=True
        )

        self.coarse_endpoint_head = MlpS(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=2,
            dropout_p=0.1, 
            use_layernorm=True
        )

        self.learnable_temp_factor = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def retrieve_with_queries(
        self,
        query_emb: Tensor,
        temperature: float = 1.0,
        use_rerank: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Query the motion bank with arbitrary embeddings.
        Enables iterative re-retrieval during refinement process.
        You can use this within a forward process.
        
        Args:
            query_emb: [B*T, n_queries, embedding_dim(128)] - retrieval queries
            temperature: Temperature for softmax selection
            use_rerank: Whether to apply endpoint re-ranking
            
        Returns:
            selected_trajectories: [B*T, n_queries, 1, n_steps, 2]
            selected_embeddings: [B*T, n_queries, hidden_dim]
            top_indices: [B*T, n_queries]
        """
        query_emb_norm = F.normalize(query_emb, dim=-1)
        attention_logits = torch.matmul(query_emb_norm, self.loaded_embeddings.t())

        if use_rerank and self.anchor_endpoints is not None:
            coarse_endpoint = self.coarse_endpoint_head(query_emb_norm)
            M = min(int(self.endpoint_topM), attention_logits.shape[-1])
            if M > 1:
                top_vals, top_idx = attention_logits.topk(M, dim=-1)
                endpoints_top = self.anchor_endpoints[top_idx]
                dists = torch.norm(endpoints_top - coarse_endpoint.unsqueeze(-2), dim=-1)
                mu = dists.mean(dim=-1, keepdim=True)
                sigma = dists.std(dim=-1, keepdim=True).clamp_min(1e-6)
                bias = -(dists - mu) / sigma
                attention_logits = attention_logits.scatter_add(-1, top_idx, bias)

        # Straight-through selection keeps the forward pass discrete.
        attention_weights_soft = F.softmax(attention_logits / temperature, dim=-1)
        top_indices = attention_logits.argmax(dim=-1)
        hard_one_hot = F.one_hot(top_indices, num_classes=attention_logits.shape[-1]).float()
        attention_weights_st = hard_one_hot + (attention_weights_soft - attention_weights_soft.detach())

        selected_embeddings = torch.matmul(attention_weights_st, self.loaded_embeddings)
        selected_trajectories = torch.einsum('bqk,kst->bqst', attention_weights_st, self.traj_tensor)

        selected_embeddings = self.output_projection(selected_embeddings.unsqueeze(1)).squeeze(1)
        
        return selected_trajectories.unsqueeze(2), selected_embeddings, top_indices

    def _attend_single_modality(
        self,
        queries: Tensor,           # [batch, n_queries, embed_dim]
        context: Tensor,           # [batch, n_ctx, embed_dim]
        context_mask: Tensor,      # [batch, n_ctx] - True where INVALID
        ca_block: nn.Module,
        gate_net: nn.Module,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply cross-attention for a single modality with optional gating.
        
        When ``self.use_gating`` is False the sigmoid gate is skipped and the
        raw attended output is returned (gate_values are ones for diagnostics).

        Returns:
            attended_delta: [batch, n_queries, embed_dim] - (gated) contribution
            gate_values: [batch, n_queries, 1] - for diagnostics
        """
        context_norm = F.normalize(context, dim=-1)
        
        attended, _ = ca_block(
            src=queries,
            tgt=context_norm,
            tgt_padding_mask=context_mask,
            need_weights=False,
        )
        
        if self.use_gating:
            gate_input = torch.cat([queries, attended], dim=-1)
            gate_values = gate_net(gate_input)
            attended_delta = gate_values * attended
        else:
            gate_values = torch.ones(
                queries.shape[0], queries.shape[1], 1,
                device=queries.device, dtype=queries.dtype,
            )
            attended_delta = attended
        
        return attended_delta, gate_values

    def multimodal_context_fusion(
        self,
        base_queries: Tensor,
        target_emb: Optional[Tensor],
        target_valid: Optional[Tensor],
        others_emb: Optional[Tensor],
        others_valid: Optional[Tensor],
        map_emb: Optional[Tensor],
        map_valid: Optional[Tensor],
        verbose=False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Apply per-modality attention and fuse the resulting query updates."""
        device = base_queries.device
        batch_size, n_queries, embed_dim = base_queries.shape

        modality_contributions = []
        diagnostics = {}

        # Without gating, active modalities contribute uniformly and the null path is disabled.
        if self.use_gating:
            logits = torch.cat(
                [self.modality_importance, self.null_importance.unsqueeze(0)],
                dim=0,
            )
            active_mask = torch.cat([self._active_modalities, self._null_modality], dim=0)
            logits = logits.masked_fill(~active_mask, float("-inf"))
            importance_weights = F.softmax(logits, dim=0)
        else:
            n_active = self._active_modalities.sum().clamp(min=1).float()
            uniform = (self._active_modalities.float() / n_active)
            importance_weights = torch.cat([uniform, torch.zeros(1, device=device)])

        diagnostics['importance_target'] = importance_weights[0].item()
        diagnostics['importance_scene'] = importance_weights[1].item()
        diagnostics['importance_map'] = importance_weights[2].item()
        diagnostics['importance_null'] = importance_weights[3].item()
        if verbose:
            with torch.no_grad():
                Q_base = base_queries[0]
                gram_base = Q_base @ Q_base.T
                off_diag_base = gram_base.fill_diagonal_(0).abs().mean()
                print(f"[BASE] Off-diagonal: {off_diag_base:.3f}, Norm: {base_queries.norm(dim=-1).mean():.3f}")

        if self.context_cfg.get("use_target", True) and target_emb is not None:
            target_ctx = self.target_proj(target_emb)
            target_type_emb = self.context_type_embedding(
                torch.zeros(1, dtype=torch.long, device=device)
            )
            target_ctx = target_ctx + target_type_emb.unsqueeze(0)

            if target_valid is not None:
                target_mask = ~target_valid.bool()
            else:
                target_mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)

            target_delta, target_gates = self._attend_single_modality(
                queries=base_queries,
                context=target_ctx,
                context_mask=target_mask,
                ca_block=self.query_target_ca,
                gate_net=self.target_gate,
            )
            
            modality_contributions.append(importance_weights[0] * target_delta)
            diagnostics['target_gate_mean'] = target_gates.mean()
            diagnostics['target_gate_std'] = target_gates.std()

        if self.context_cfg.get("use_other", False) and others_emb is not None:
            scene_ctx = self.scene_proj(others_emb)
            scene_type_emb = self.context_type_embedding(
                torch.ones(scene_ctx.shape[1], dtype=torch.long, device=device)
            )
            scene_ctx = scene_ctx + scene_type_emb.unsqueeze(0)

            if others_valid is not None:
                scene_mask = ~others_valid.bool()
            else:
                scene_mask = torch.zeros(batch_size, scene_ctx.shape[1], device=device, dtype=torch.bool)

            scene_delta, scene_gates = self._attend_single_modality(
                queries=base_queries,
                context=scene_ctx,
                context_mask=scene_mask,
                ca_block=self.query_scene_ca,
                gate_net=self.scene_gate,
            )
            
            modality_contributions.append(importance_weights[1] * scene_delta)
            diagnostics['scene_gate_mean'] = scene_gates.mean()
            diagnostics['scene_gate_std'] = scene_gates.std()

        if self.context_cfg.get("use_map", False) and map_emb is not None:
            map_ctx = self.map_proj(map_emb)
            map_type_emb = self.context_type_embedding(
                torch.full((map_ctx.shape[1],), 2, dtype=torch.long, device=device)
            )
            map_ctx = map_ctx + map_type_emb.unsqueeze(0)

            if map_valid is not None:
                map_mask = ~map_valid.bool()
            else:
                map_mask = torch.zeros(batch_size, map_ctx.shape[1], device=device, dtype=torch.bool)

            map_delta, map_gates = self._attend_single_modality(
                queries=base_queries,
                context=map_ctx,
                context_mask=map_mask,
                ca_block=self.query_map_ca,
                gate_net=self.map_gate,
            )
            
            modality_contributions.append(importance_weights[2] * map_delta)
            diagnostics['map_gate_mean'] = map_gates.mean()
            diagnostics['map_gate_std'] = map_gates.std()

        if len(modality_contributions) == 0:
            return base_queries, diagnostics

        total_context_delta = torch.stack(modality_contributions, dim=0).sum(dim=0)

        adapted_queries = base_queries + total_context_delta

        if verbose:
            with torch.no_grad():
                Q_adapted_norm = F.normalize(adapted_queries, dim=-1)
                gram_adapted_norm = Q_adapted_norm[0] @ Q_adapted_norm[0].T
                off_diag_adapted_norm = gram_adapted_norm.fill_diagonal_(0).abs().mean()
                print(f"[ADAPTED-NORM] Off-diagonal: {off_diag_adapted_norm:.3f}, Adapted norm: {adapted_queries.norm(dim=-1).mean():.3f}")

        return adapted_queries, diagnostics

    def forward(
        self,
        valid_mask: Tensor,
        target_valid=None,
        target_emb=None,
        agent_type: Optional[Tensor] = None,
        others_emb=None,
        others_valid=None,
        map_emb=None,
        map_valid=None,   
        all_agent_tokens=None,
        all_agent_invalid=None,
    ):
        """Select trajectories from the bank using the configured context inputs."""

        valid_mask_flat = einops.rearrange(valid_mask, 'b t -> (b t)')
        flat_target_emb = einops.rearrange(target_emb, 'b t d -> (b t) d')
        flat_agent_type = einops.rearrange(agent_type, 'b t c -> (b t) c')

        base_queries = self.latent_orth_query(
            valid_mask_flat,
            None,
            agent_type=flat_agent_type,
        )
        base_queries_norm = F.normalize(base_queries, dim=-1) 

        if self.selection_mode == "no_extras":
            adapted_queries, _ = self.multimodal_context_fusion(
                base_queries=base_queries_norm,
                target_emb=target_emb, target_valid=target_valid,
                others_emb=others_emb, others_valid=others_valid,
                map_emb=map_emb, map_valid=map_valid,
            )
            adapted_queries_norm = F.normalize(adapted_queries, dim=-1)
            attention_logits = torch.matmul(adapted_queries_norm, self.loaded_embeddings.t())

        else:
            if all_agent_tokens is None:
                adapted_queries, _ = self.multimodal_context_fusion(
                    base_queries=base_queries_norm,
                    target_emb=target_emb, target_valid=target_valid,
                    others_emb=others_emb, others_valid=others_valid,
                    map_emb=map_emb, map_valid=map_valid, verbose=False,
                )
                adapted_queries_norm = F.normalize(adapted_queries, dim=-1)
            else:
                base_queries = self.latent_orth_query_256(valid_mask_flat, None, agent_type=None)

                adapted_queries_256, _ = self.all_agent_ca(
                    src=base_queries,
                    tgt=all_agent_tokens,
                    tgt_padding_mask=all_agent_invalid,
                )
                # Keep the cross-attention update residual so the query set stays diverse.
                delta = F.normalize(adapted_queries_256 - base_queries, dim=-1)
                adapted_queries_256 = F.normalize(base_queries + 0.3 * delta, dim=-1)
                adapted_queries = self.all_agents_proj(base_queries)
                adapted_queries_norm = F.normalize(adapted_queries, dim=-1)

            attention_logits = torch.matmul(adapted_queries_norm, self.loaded_embeddings.t())

            if (self.retrieval_mode in ("endpoint_rerank")
                and self.anchor_endpoints is not None
                and attention_logits.shape[-1] > 1
            ):
                with torch.no_grad():
                    M = int(self.endpoint_topM)
                    M = max(1, min(M, attention_logits.shape[-1]))
                if M > 1:
                    coarse_endpoint = self.coarse_endpoint_head(adapted_queries_norm)
                    top_vals, top_idx = attention_logits.topk(M, dim=-1)
                    endpoints_top = self.anchor_endpoints[top_idx]
                    dists = torch.norm(endpoints_top - coarse_endpoint.unsqueeze(-2), dim=-1)
                    mu = dists.mean(dim=-1, keepdim=True)
                    sigma = dists.std(dim=-1, keepdim=True).clamp_min(1e-6)
                    z = (dists - mu) / sigma
                    bias = -z
                    attention_logits = attention_logits.scatter_add(-1, top_idx, bias)

        temp = self.learnable_temp_factor.clamp_min(0.05)
        attention_weights_soft = F.softmax(attention_logits / temp, dim=-1)
        top_indices = attention_logits.argmax(dim=-1)

        if self.use_straight_through:
            hard_one_hot = F.one_hot(top_indices, num_classes=attention_logits.shape[-1]).float()
            retrieval_weights = hard_one_hot + (attention_weights_soft - attention_weights_soft.detach())
        else:
            retrieval_weights = attention_weights_soft

        selected_embeddings = torch.matmul(retrieval_weights, self.loaded_embeddings).unsqueeze(1)
        slice_max_indices = top_indices.unsqueeze(-1)
        
        selected_embeddings = self.output_projection(selected_embeddings)
        adapted_queries_out = self.output_projection_adapted_queries(adapted_queries_norm)

        if valid_mask is not None:
            flat_valid = einops.rearrange(valid_mask, 'b t -> (b t)')
            mask = einops.rearrange(flat_valid.float(), 'b -> b 1 1 1')
            selected_embeddings = selected_embeddings * mask

        soft_traj = torch.einsum('bqk,kst->bqst', attention_weights_soft, self.traj_tensor)
        selected_trajectories = torch.einsum('bqk,kst->bqst', retrieval_weights, self.traj_tensor).unsqueeze(2)

        adapted_queries_256 = adapted_queries_out
        adapted_queries_128 = adapted_queries_norm

        return (
            selected_trajectories,
            selected_embeddings,
            slice_max_indices,
            soft_traj,
            adapted_queries_256,
            adapted_queries_128,
        )
