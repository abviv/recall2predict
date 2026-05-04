import torch
import torch.nn.functional as F
import logging
from omegaconf import DictConfig
from typing import Optional, List, Dict
from torch import nn, Tensor
from models.ac_wayformer import InputProjections
from HPTR.src.models.modules.decoder_ensemble import MLPHead
from HPTR.src.models.modules.multi_modal import MultiModalAnchors
from layers_in_my_way.modules.transformer import TransformerBlock as LIMTransformerBlock
from src.models.modules.trajectory_projector import TrajectoryEncoder_DualPath
from src.models.modules.trajectory_selector_softattn_tf import SoftAttentionTrajectorySelector
from src.models.modules.pose_projection import PoseProjection
from layers_in_my_way.modules.mlp import MlpS
from src.models.modules.gMLP import gMLPBlock

log = logging.getLogger(__name__)


class PhysicalAnchorGQA(nn.Module):
    """
    Differentiable Grouped Query Approach for Physical Trajectories.
    Reduces N retrieved anchors to K representative queries while maintaining
    end-to-end gradients to all N original anchors.
    """
    def __init__(self, n_groups: int = 6, temperature: float = 1.0):
        super().__init__()
        self.n_groups = n_groups
        self.temperature = temperature
        
    def _batch_fps(self, endpoints: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Farthest Point Sampling. 
        Note: The argmax here drops gradients for the *indices*, but the 
        subsequent gather operation in forward() preserves gradients for the *tensors*.
        """
        B, K, _ = endpoints.shape
        device = endpoints.device
        
        centroids = torch.zeros((B, num_points), dtype=torch.long, device=device)
        distances = torch.ones((B, K), device=device) * 1e10
        
        # Always seed the first group with the highest-scoring anchor (index 0)
        farthest = torch.zeros(B, dtype=torch.long, device=device)
        batch_indices = torch.arange(B, device=device)
        
        for i in range(num_points):
            centroids[:, i] = farthest
            centroid_points = endpoints[batch_indices, farthest].unsqueeze(1) # [B, 1, 2]
            
            # Squared L2 distance
            dist = torch.sum((endpoints - centroid_points) ** 2, dim=-1) # [B, K]
            
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.argmax(distances, dim=-1)
            
        return centroids

    def forward(self, anchor_tokens: torch.Tensor, anchor_trajectories: torch.Tensor):
        """
        Args:
            anchor_tokens: [B, 32, D] Differentiable latent embeddings
            anchor_trajectories: [B, 32, 1, T, 2] Differentiable physical trajectories (via STE)
            
        Returns:
            grouped_tokens: [B, 6, D] Aggregated queries for the decoder
            medoid_trajectories: [B, 6, 1, T, 2] Exact physical trajectories for the loss function
            attn_weights: [B, 6, 32] The assignment matrix
            seed_indices: [B, 6] The indices of the selected medoids
        """
        B, K, D = anchor_tokens.shape
        
        # 1. Extract physical endpoints [B, 32, 2]
        endpoints = anchor_trajectories[:, :, 0, -1, :] 
        
        # 2. Get Medoid Indices via FPS (Discrete, non-differentiable step)
        seed_indices = self._batch_fps(endpoints, self.n_groups) # [B, 6]
        
        # 3. Gather Medoids (GRADIENT PATH RESTORED HERE via indexing)
        batch_idx = torch.arange(B, device=anchor_tokens.device).unsqueeze(-1)
        medoid_trajectories = anchor_trajectories[batch_idx, seed_indices] # [B, 6, 1, T, 2]
        medoid_endpoints = endpoints[batch_idx, seed_indices] # [B, 6, 2]
        
        # 4. Compute Soft Physical Assignment
        # We compute the squared L2 distance between all 32 endpoints and the 6 medoids.
        # endpoints: [B, 1, 32, 2], medoid_endpoints: [B, 6, 1, 2]
        dist = torch.sum(
            (endpoints.unsqueeze(1) - medoid_endpoints.unsqueeze(2)) ** 2, 
            dim=-1
        ) # [B, 6, 32]
        
        # Convert physical distance to attention weights.
        # We softmax over the 32 anchors (dim=-1) so each of the 6 groups 
        # is a weighted sum of the anchors, heavily biased to physically close ones.
        attn_weights = F.softmax(-dist / self.temperature, dim=-1) # [B, 6, 32]
        
        # 5. Aggregate Latent Tokens
        # [B, 6, 32] @ [B, 32, D] -> [B, 6, D]
        grouped_tokens = torch.bmm(attn_weights, anchor_tokens)

        # 6. Assignment entropy: scalar measuring how soft/hard the grouping is
        entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(-1).mean()

        return grouped_tokens, medoid_trajectories, attn_weights, seed_indices, entropy

class R2PDecoder(nn.Module):

    """
    Recall2Predict refinement decoder with alternating cross-attention per layer.

    Two operating modes controlled by ``use_anchors_as_queries``:

    **Anchors-as-queries (default, use_anchors_as_queries=True):**
        Retrieved anchor embeddings ARE the decoder queries.  Each query maps
        1-to-1 to a specific retrieved trajectory from the motion bank.
        Refinement only conditions them on the scene (agent + lane CA) without
        ever mixing anchors together, preserving full explainability.

    **Learnable-queries (use_anchors_as_queries=False):**
        Standard DETR-style learnable mode embeddings attend to anchors as KV
        memory via an additional anchor_ca layer.  This is the ablation
        baseline where 1-to-1 correspondence is NOT guaranteed.

    Refinement order per layer:
        1) CA(queries, agent_tokens)   - focal agent context
        2) CA(queries, lane_tokens)    - lane/map context
        3) CA(queries, anchor_tokens)  - only when use_anchors_as_queries=False

    Inputs:
    - agent_tokens: [B*T, 1, D] focal agent embedding only (ViT style)
    - lane_tokens: [B*T, n_lane, D] lane/map embeddings
    - anchor_tokens: [B*T, K_anch, D] from trajectory selector (+ projection)
    """
    def __init__(
        self,
        embed_dim: int,
        future_steps: int,
        n_pred: int = 6,
        attn_depth: int = 3,
        num_heads: int = 8,
        dropout_p: float = 0.1,
        d_feedforward: Optional[int] = None,
        norm_first: bool = True,
        bias: bool = True,
        use_anchors_as_queries: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.n_pred = n_pred
        self.attn_depth = attn_depth
        self.use_anchors_as_queries = use_anchors_as_queries
        
        if not self.use_anchors_as_queries:
            self.latent_query = MultiModalAnchors(
                hidden_dim=embed_dim,
                emb_dim=embed_dim,
                n_pred=n_pred,
                mode_emb="none",
                mode_init="uniform",
                use_agent_type=True
            )

        if d_feedforward is None:
            d_feedforward = embed_dim * 2

        _ca_kwargs = dict(emb_dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=d_feedforward, dropout_p=dropout_p,
                          is_crossattention=True)

        self.refinement_layers = nn.ModuleList()
        for _ in range(attn_depth):
            layer = nn.ModuleDict({
                "agent_ca": LIMTransformerBlock(**_ca_kwargs),
                "lane_ca":  LIMTransformerBlock(**_ca_kwargs),
            })
            if not self.use_anchors_as_queries:
                layer["anchor_ca"] = LIMTransformerBlock(**_ca_kwargs)
            self.refinement_layers.append(layer)
    
    def forward(
        self,
        agent_tokens: Tensor,        # [B, 1, D] focal agent embedding only
        lane_tokens: Tensor,         # [B, n_lane, D] lane/map embeddings
        anchor_tokens: Tensor,       # [B, K_anch, D] anchor embeddings from trajectory selector
        valid_mask: Tensor,          # [B] valid mask for latent query
        agent_type: Optional[Tensor] = None,   # [B, 3] one-hot agent type
        agent_mask: Optional[Tensor] = None,   # [B, 1] True where padded
        lane_mask: Optional[Tensor] = None,    # [B, n_lane] True where padded
        anchor_mask: Optional[Tensor] = None,  # [B, K_anch] True where padded
    ) -> Tensor:
        """        
        Args:
            agent_tokens: [B, 1, D] focal agent embedding only
            lane_tokens: [B, n_lane, D] lane/map embeddings
            anchor_tokens: [B, K_anch, D] anchor embeddings from trajectory selector
            valid_mask: [B] valid mask for initializing latent queries
            agent_type: [B, 3] one-hot agent type (required when use_agent_type=True)
            agent_mask: [B, 1] padding mask for focal agent (True where invalid)
            lane_mask: [B, n_lane] padding mask for lanes (True where invalid)
            anchor_mask: [B, K_anch] padding mask for anchors (True where invalid)
            
        Returns:
            mode_queries: [B, n_pred, D] mode embeddings for shared MLP head
        """
        if self.use_anchors_as_queries:
            # Each query IS a retrieved anchor: 1-to-1 correspondence is
            # preserved throughout the decoder.  Agent/lane CA conditions
            # the anchors on scene context without ever mixing them.
            mode_queries = anchor_tokens  # [B, K_anch, D]
        else:
            mode_queries = self.latent_query(valid_mask, None, agent_type=agent_type)
        
        # Flip masks once: HPTR (True=invalid) -> LIM (True=valid)
        agent_valid = ~agent_mask if agent_mask is not None else None
        lane_valid = ~lane_mask if lane_mask is not None else None
        anchor_valid = ~anchor_mask if anchor_mask is not None else None

        for layer in self.refinement_layers:
            mode_queries = layer["agent_ca"](
                mode_queries, k=agent_tokens, v=agent_tokens, mask=agent_valid,
            )
            mode_queries = layer["lane_ca"](
                mode_queries, k=lane_tokens, v=lane_tokens, mask=lane_valid,
            )
            if not self.use_anchors_as_queries:
                mode_queries = layer["anchor_ca"](
                    mode_queries, k=anchor_tokens, v=anchor_tokens, mask=anchor_valid,
                )
        
        return mode_queries


class BoundaryAware(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            agent_attr_dim: int,
            map_attr_dim: int,
            tl_attr_dim: int,
            n_pl_node: int,
            use_current_tl: bool,
            pl_aggr: bool,
            n_step_hist: int,
            n_decoders: int,
            use_encoder: bool,
            tf_cfg: DictConfig,
            local_encoder: DictConfig,
            motion_decoder: DictConfig,
            early_fusion_encoder: DictConfig,
            trajectory_selector: DictConfig,
            gt_in_local: bool = True,
            agent_centric: bool = True,
            loaded_embeddings: Optional[torch.Tensor] = None,
            traj_tensor: Optional[torch.Tensor] = None,
            encoder_init_from: Optional[Dict] = None,
            freeze_modules: Optional[List[str]] = None,
            **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_decoders = n_decoders
        self.tf_cfg = tf_cfg
        self.pl_aggr = pl_aggr
        self.n_step_hist = n_step_hist
        self.n_pl_node = n_pl_node
        self.n_pred = motion_decoder.n_pred
        self.use_encoder = use_encoder
        self.gt_in_local = gt_in_local
        self.agent_centric = agent_centric
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        self.n_step_future = (motion_decoder["mlp_head_cfg"]["n_step_future"] //
                             self.pred_subsampling_rate)
        self.use_traj_projection = trajectory_selector.get("use_traj_projection", True)
        self.n_anchors = trajectory_selector.get('n_latent_anchors', 32)
        
        # Decoder config
        self.decoder_depth = motion_decoder.get("n_refinement_layer", 3)
        self.use_offset_prediction = motion_decoder.get("use_offset_prediction", True)
        self.use_anchors_as_queries = motion_decoder.get("use_anchors_as_queries", True)
        self.use_gqa = trajectory_selector.get("use_gqa", False)
        self.lambda_gqa_entropy = float(trajectory_selector.get("lambda_gqa_entropy", 0.0))
        if self.use_anchors_as_queries:
            if self.use_gqa:
                if self.n_pred > self.n_anchors:
                    raise ValueError(
                        f"use_gqa requires n_pred ({self.n_pred}) <= "
                        f"n_anchors ({self.n_anchors})."
                    )
                self.anchor_gqa = PhysicalAnchorGQA(n_groups=self.n_pred, temperature=1.0)
            else:
                if self.n_pred != self.n_anchors:
                    raise ValueError(
                        f"use_anchors_as_queries requires n_pred ({self.n_pred}) == "
                        f"n_anchors ({self.n_anchors}), since each anchor becomes a "
                        f"decoder query with 1-to-1 correspondence."
                    )
        
        # === Input Projections ===
        self.local_projections = InputProjections(
            hidden_dim=hidden_dim, agent_attr_dim=agent_attr_dim, map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim, pl_aggr=pl_aggr, use_current_tl=use_current_tl,
            n_step_hist=n_step_hist, n_pl_node=n_pl_node, **local_encoder
        )

        self.target_gmlp_proj = MlpS(
            input_dim=agent_attr_dim,
            hidden_dim=agent_attr_dim * 2,
            output_dim=hidden_dim,
            dropout_p=0.1,
            use_layernorm=True,
        )
        self.target_gmlp = gMLPBlock(
            d_model=hidden_dim,
            d_ffn=hidden_dim * 2,
            seq_len=n_step_hist,
        )

        self.pose_projection = PoseProjection(
            hidden_dim=hidden_dim,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            pl_aggr=pl_aggr,
            use_point_net=local_encoder.get("use_point_net", False),
        )

        self.agent_attn_depth = early_fusion_encoder.get("agent_attn_depth", 1)
        self.lane_attn_depth = early_fusion_encoder.get("lane_attn_depth", 1)
        self.r2p_attn_depth = early_fusion_encoder.get("r2p_attn_depth", 1)
        self.anchor_attn_depth = early_fusion_encoder.get(
            "anchor_attn_depth", trajectory_selector.get("anchor_attn_depth", 1)
        )
        self.encoder_ffn_dim = early_fusion_encoder.get("d_feedforward", hidden_dim)
        if self.encoder_ffn_dim is None:
            self.encoder_ffn_dim = hidden_dim
        decoder_tf_cfg = motion_decoder.get("tf_cfg", {})
        self.decoder_ffn_dim = decoder_tf_cfg.get(
            "d_feedforward", motion_decoder.get("d_feedforward", hidden_dim)
        )
        if self.decoder_ffn_dim is None:
            self.decoder_ffn_dim = hidden_dim

        _lim_kwargs = dict(
            emb_dim=hidden_dim,
            num_heads=tf_cfg.n_head,
            mlp_hidden_dim=self.encoder_ffn_dim,
            dropout_p=tf_cfg.get("dropout_p", 0.1),
        )
        self.agent_self_attn = nn.ModuleList([
            LIMTransformerBlock(**_lim_kwargs) for _ in range(self.agent_attn_depth)
        ])
        self.lane_self_attn = nn.ModuleList([
            LIMTransformerBlock(**_lim_kwargs) for _ in range(self.lane_attn_depth)
        ])
        self.r2p_fusion_attn = nn.ModuleList([
            LIMTransformerBlock(**_lim_kwargs) for _ in range(self.r2p_attn_depth)
        ])

        # Self-attn over selected anchor tokens before decoding
        self.anchor_self_attn = nn.ModuleList([
            LIMTransformerBlock(**_lim_kwargs) for _ in range(self.anchor_attn_depth)
        ])

        # === Trajectory Selector ===
        if trajectory_selector.use_layer == "softattn":
            if loaded_embeddings is None or traj_tensor is None:
                raise ValueError("loaded_embeddings and traj_tensor must be provided")

            self.trajectory_selector = SoftAttentionTrajectorySelector(
                loaded_embeddings=loaded_embeddings, traj_tensor=traj_tensor,
                hidden_dim=hidden_dim,
                n_latent_anchors=trajectory_selector.get('n_latent_anchors', 32),
                context_cfg=trajectory_selector.get('contexts', None),
                retrieval_mode=trajectory_selector.get("retrieval_mode", None),
                endpoint_topM=trajectory_selector.get('endpoint_topM', 32),
                selection_mode=trajectory_selector.get('selection_mode', 'full'),
                use_straight_through = trajectory_selector.get('use_straight_through', True),
                use_gating = trajectory_selector.get('use_gating', True),
            )
        else:
            raise ValueError(f"Trajectory selector {trajectory_selector.use_layer} not supported.")

        # === Trajectory Projection ===
        if self.use_traj_projection:
            self.anchor_projection = TrajectoryEncoder_DualPath(
                seq_len=self.n_step_future, embedding_dim=hidden_dim, cnn_channels=128, pooling_type="max"
            )

        # === DETR-Style Refinement Decoder ===
        self.decoder = R2PDecoder(
            embed_dim=hidden_dim,
            future_steps=self.n_step_future,
            n_pred=self.n_pred,
            attn_depth=self.decoder_depth,
            num_heads=tf_cfg.n_head,
            dropout_p=tf_cfg.get("dropout_p", 0.1),
            d_feedforward=self.decoder_ffn_dim,
            norm_first=tf_cfg.get("norm_first", True),
            bias=tf_cfg.get("bias", True),
            use_anchors_as_queries=self.use_anchors_as_queries,
        )

        # === MLP Head Configuration ===
        # See cfg for more control defn. about the prediction head:
        #   n_pred: total mode queries the decoder produces
        #   mlp_head_num_heads: how many non-parameter-shared heads to split them across
        self.mlp_head_num_heads = motion_decoder.get("mlp_head_num_heads", 1)
        
        if self.mlp_head_num_heads > 1:
            if self.n_pred % self.mlp_head_num_heads != 0:
                raise ValueError(
                    f"n_pred ({self.n_pred}) must be divisible by "
                    f"mlp_head_num_heads ({self.mlp_head_num_heads})."
                )
            self.mlp_head_max_modes = self.n_pred // self.mlp_head_num_heads
        else:
            self.mlp_head_max_modes = self.n_pred

        mlp_head_cfg = motion_decoder["mlp_head_cfg"]
        self.mlp_heads = nn.ModuleList([
            MLPHead(
                hidden_dim=hidden_dim,
                use_vmap=mlp_head_cfg.get("use_vmap", False),
                n_step_future=mlp_head_cfg.get("n_step_future", 60),
                out_mlp_layernorm=mlp_head_cfg.get("out_mlp_layernorm", True),
                out_mlp_batchnorm=mlp_head_cfg.get("out_mlp_batchnorm", False),
                use_agent_type=mlp_head_cfg.get("use_agent_type", False),
                predictions=mlp_head_cfg.get("predictions", ["pos"]),
                n_classes=mlp_head_cfg.get("n_classes", 0),
            )
            for _ in range(self.mlp_head_num_heads)
        ])
        
        self.dense_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, 256), # [256 -> 256]
                nn.ReLU(inplace=True), 
                nn.Linear(256,  self.n_step_future * 2) # [256 -> (60*2)] 
                )

        # === Offset Head ===
        if self.use_offset_prediction:
            offset_dropout_p = motion_decoder.get("offset_dropout_p", 0.2)
            self.offset_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.ReLU(),
                nn.Dropout(offset_dropout_p),
                nn.Linear(hidden_dim*2, 2),
            )
            nn.init.normal_(self.offset_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.offset_head[-1].bias)
        
        if freeze_modules:
            self._freeze_modules(freeze_modules)

        self._print_param_counts()

    def _print_param_counts(self):
        def _trainable_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("Parameter Counts (DETR-Refine Architecture)")
        print("="*60)
        
        for name, module in [
            ('Projections', self.local_projections),
            # ('Encoder', self.encoder),
            ('PoseProjection', self.pose_projection),
            ('Selector', self.trajectory_selector),
        ]:
            params = _trainable_params(module)
            print(f"  {name}: {params/1e6:.2f}M")

        if self.use_traj_projection:
            params = _trainable_params(self.anchor_projection)
            print(f"  TrajectoryProjection: {params/1e6:.2f}M")

        agent_attn_params = sum(_trainable_params(b) for b in self.agent_self_attn)
        lane_attn_params = sum(_trainable_params(b) for b in self.lane_self_attn)
        r2p_attn_params = sum(_trainable_params(b) for b in self.r2p_fusion_attn)
        print(f"  AgentSelfAttn: {agent_attn_params/1e6:.2f}M")
        print(f"  LaneSelfAttn: {lane_attn_params/1e6:.2f}M")
        print(f"  R2PFusionAttn: {r2p_attn_params/1e6:.2f}M")
        anchor_attn_params = sum(_trainable_params(b) for b in self.anchor_self_attn)
        print(f"  AnchorSelfAttn: {anchor_attn_params/1e6:.2f}M")
        
        decoder_params = _trainable_params(self.decoder)
        print(f"  Decoder (DETR Style): {decoder_params/1e6:.2f}M")
        agent_params = sum(_trainable_params(l["agent_ca"]) for l in self.decoder.refinement_layers)
        lane_params = sum(_trainable_params(l["lane_ca"]) for l in self.decoder.refinement_layers)
        print(f"  -> agent_ca: {agent_params/1e6:.2f}M")
        print(f"  -> lane_ca: {lane_params/1e6:.2f}M")
        if not self.use_anchors_as_queries:
            anchor_params = sum(_trainable_params(l["anchor_ca"]) for l in self.decoder.refinement_layers)
            print(f"  -> anchor_ca: {anchor_params/1e6:.2f}M")
            print(f"  -> latent_query: {_trainable_params(self.decoder.latent_query)/1e6:.4f}M")
        else:
            print(f"  -> (anchors-as-queries: no anchor_ca or latent_query)")
        mlp_head_params = sum(_trainable_params(head) for head in self.mlp_heads)
        print(f"  MLPHead (x{self.mlp_head_num_heads}): {mlp_head_params/1e6:.2f}M")
        
        if self.use_offset_prediction:
            offset_params = _trainable_params(self.offset_head)
            print(f"  OffsetHead: {offset_params/1e6:.2f}M")
        
        total_params = _trainable_params(self)
        print("-"*60)
        print(f"  TOTAL: {total_params/1e6:.2f}M")
        print("="*60 + "\n")

    def forward(self,
        target_valid: Tensor,
        target_type: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Forward pass with DETR-style refinement decoder.
        
        Returns: A dictionary containing:
            "valid_mask": [B, T]
            "conf": [1, B, T, n_pred] logits
            "pred": [1, B, T, n_pred, n_step_future, pred_dim]
            "pred_pos_logits": None
            "offset_pred": [1, B, T, n_anchors, 2] if use_offset_prediction else None
            "anchor_container": dict with anchor information
        """
        for _ in range(inference_repeat_n):
            
            raw_target_attr = target_attr
            raw_other_attr = other_attr
            raw_map_attr = map_attr
            raw_target_valid = target_valid
            raw_other_valid = other_valid
            raw_map_valid = map_valid

            valid_mask = target_valid if self.pl_aggr else target_valid.any(-1)  # [B, T]
            batch_size, T = valid_mask.shape

            # ---- Input Projections
            target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid = self.local_projections(
                target_valid=target_valid, target_attr=target_attr,
                other_valid=other_valid, other_attr=other_attr,
                map_valid=map_valid, map_attr=map_attr,
                tl_valid=tl_valid, tl_attr=tl_attr
            )

            # ---- Encoder Phase (agent/lane self-attn + R2P fusion)
            # Combine target and other agents for scene-level self-attention
            # but keep track of focal agent position for decoder
            all_agent_tokens = torch.cat([target_emb, other_emb], dim=1)
            all_agent_invalid = ~torch.cat([target_valid, other_valid], dim=1)
            focal_agent_len = target_emb.shape[1]  # 1 or n_step_hist

            # Start the environment stream from map tokens only; TL tokens are
            # fused in after map positional embeddings are restored.
            lane_tokens = map_emb
            lane_invalid = ~map_valid
            
            # ---- Adding the positional info into the all_agents_token and map_tokens
            # for this to work properly: model.pre_processing.ac_global.pose_pe.map=xy_dir
            pos_embed_agent, pos_embed_lane = self.pose_projection(
                target_attr=raw_target_attr,
                other_attr=raw_other_attr,
                map_attr=raw_map_attr,
                target_valid=raw_target_valid,
                other_valid=raw_other_valid,
                map_valid=raw_map_valid,
                agent_token_count=all_agent_tokens.shape[1],
                lane_token_count=lane_tokens.shape[1],
            )

            # ---- gMLP Target History Encoding
            # Process raw target_attr through gMLP for temporal mixing across n_step_hist,
            # producing [B*T, n_step_hist, hidden_dim] context for the trajectory selector.
            flat_raw_target = raw_target_attr.flatten(0, 1)          # [B*T, n_step_hist, attr_dim]
            flat_raw_target_valid = raw_target_valid.flatten(0, 1)   # [B*T, n_step_hist]
            gmlp_target_emb = self.target_gmlp_proj(flat_raw_target)
            gmlp_target_emb = gmlp_target_emb * flat_raw_target_valid.unsqueeze(-1).float()
            gmlp_target_emb = self.target_gmlp(gmlp_target_emb)     # [B*T, n_step_hist, hidden_dim]

            # ---- Trajectory Selector
            selector_outputs = self.trajectory_selector(
                valid_mask=valid_mask,
                target_emb=gmlp_target_emb,
                target_valid=flat_raw_target_valid,
                agent_type=target_type,
                others_emb=other_emb,
                others_valid=other_valid,
                map_emb=map_emb,
                map_valid=map_valid,
            )

            ( selected_anchor_trajectories, selected_embeddings, selected_anchor_indices, 
              _, adapted_queries, adapted_queries_128 ) = selector_outputs

            if self.use_traj_projection:
                proto_embeddings = self.anchor_projection(selected_anchor_trajectories)
            else:
                raise ValueError("Trajectory projection must be used")

            # ---- Build Anchor Tokens
            n_anchors = selected_embeddings.shape[2]
            selected_embeddings_flat = selected_embeddings.flatten(0, 1)  # [B*T, K, D]
            anchor_tokens = proto_embeddings + selected_embeddings_flat + adapted_queries  # [B*T, K, D]

            # ---- GQA INJECTION POINT
            gqa_attn_weights = None
            gqa_entropy_loss = torch.tensor(0.0, device=anchor_tokens.device)
            if getattr(self, 'use_gqa', False) and self.use_anchors_as_queries:
                anchor_tokens, selected_anchor_trajectories, gqa_attn_weights, seed_indices, gqa_entropy = self.anchor_gqa(
                    anchor_tokens, selected_anchor_trajectories
                )
                n_anchors = anchor_tokens.shape[1]

                # Entropy regularization: maximize entropy -> softer assignments
                if self.lambda_gqa_entropy > 0:
                    gqa_entropy_loss = -self.lambda_gqa_entropy * gqa_entropy

                # Update selected_anchor_indices to reflect the medoids' original bank indices
                batch_idx = torch.arange(anchor_tokens.shape[0], device=anchor_tokens.device).unsqueeze(-1)
                selected_anchor_indices = selected_anchor_indices.squeeze(-1) # [B*T, K]
                selected_anchor_indices = selected_anchor_indices[batch_idx, seed_indices].unsqueeze(-1) # [B*T, 6, 1]

                adapted_queries = adapted_queries[batch_idx, seed_indices]
                adapted_queries_128 = adapted_queries_128[batch_idx, seed_indices]

            # Re-add positional information to map tokens before extending the
            # environment stream with traffic-light tokens.
            all_agent_tokens = all_agent_tokens + pos_embed_agent
            lane_tokens = lane_tokens + pos_embed_lane

            # Traffic lights act as dynamic environment context, so append them
            # to the lane/environment stream rather than the agent or anchor streams.
            lane_tokens = torch.cat([lane_tokens, tl_emb], dim=1)
            lane_invalid = torch.cat([lane_invalid, ~tl_valid], dim=1)
            assert lane_tokens.shape[1] == lane_invalid.shape[1]

            # ---- Encoder
            # Masks are flipped from HPTR convention (True=invalid) -> LIM (True=valid).

            # Agent self-attention (all agents for scene understanding)
            all_agent_valid = ~all_agent_invalid
            for blk in self.agent_self_attn:
                all_agent_tokens = blk(all_agent_tokens, mask=all_agent_valid)

            # Environment self-attention over map + traffic-light tokens.
            lane_valid = ~lane_invalid
            for blk in self.lane_self_attn:
                lane_tokens = blk(lane_tokens, mask=lane_valid)

            # R2P fusion: joint attention over agents + lanes
            r2p_tokens = torch.cat([all_agent_tokens, lane_tokens], dim=1)
            r2p_valid = torch.cat([all_agent_valid, lane_valid], dim=1)
            for blk in self.r2p_fusion_attn:
                r2p_tokens = blk(r2p_tokens, mask=r2p_valid)

            # Split back after R2P fusion
            all_agent_len = all_agent_tokens.shape[1]
            all_agent_tokens = r2p_tokens[:, :all_agent_len, :]
            lane_tokens = r2p_tokens[:, all_agent_len:, :]

            # Extract only focal agent for decoder cross-attention
            focal_agent_tokens = all_agent_tokens[:, :focal_agent_len, :]
            focal_agent_invalid = all_agent_invalid[:, :focal_agent_len]
            other_agent_tokens = all_agent_tokens[:, focal_agent_len:, :] # [B*T, n_others, hidden_dim]
            other_agent_invalid = all_agent_invalid[:, focal_agent_len:]

            valid_mask_flat = valid_mask.flatten(0, 1)  # [B*T]

            # Anchor self-attn (no padding mask - all anchor slots are valid)
            for blk in self.anchor_self_attn:
                anchor_tokens = blk(anchor_tokens)
            anchor_mask = None  # still None for the decoder below

            # ---- DETR Stlye Refinement Decoder
            # Pass only focal agent (not other agents)
            mode_queries = self.decoder(
                agent_tokens=focal_agent_tokens,
                agent_mask=focal_agent_invalid,
                lane_tokens=lane_tokens,
                lane_mask=lane_invalid,
                anchor_tokens=anchor_tokens,
                anchor_mask=anchor_mask,
                valid_mask=valid_mask_flat,
                agent_type=target_type.flatten(0, 1),
            )

            # Reshape back to [B, T, ...] for MLP head(s)
            mode_queries = mode_queries.view(batch_size, T, self.n_pred, self.hidden_dim)
            
            # Ensemble mode (experimental): randomly permute, split into chunks, route to separate heads
            if self.mlp_head_num_heads > 1:
                perm = torch.randperm(self.n_pred, device=mode_queries.device)
                mode_queries = mode_queries.index_select(2, perm)
                mode_chunks = mode_queries.split(self.mlp_head_max_modes, dim=2)

                conf_chunks = []
                pred_chunks = []
                for head_idx, chunk in enumerate(mode_chunks):
                    _conf, _pred = self.mlp_heads[head_idx](
                        valid=valid_mask, emb=chunk, agent_type=target_type
                    )
                    conf_chunks.append(_conf)
                    pred_chunks.append(_pred)
                conf = torch.cat(conf_chunks, dim=2)
                pred = torch.cat(pred_chunks, dim=2)
            else:
                # Single head: pass all n_pred mode queries through one head
                conf, pred = self.mlp_heads[0](valid=valid_mask, emb=mode_queries, agent_type=target_type)

            # Optional: Offset Prediction
            if self.use_offset_prediction:
                offset_pred = self.offset_head(anchor_tokens)  # [B*T, K, 2]
                offset_pred = offset_pred.view(batch_size, T, n_anchors, 2)
            else:
                offset_pred = None
        
        # ---- Dense prediction for non-focal agents (no decoder, single-shot regression)
        y_hat_others = self.dense_predictor(other_agent_tokens)  # [B*T, n_others, n_step_fut*2]
        n_others = y_hat_others.shape[1]
        y_hat_others = y_hat_others.view(batch_size, T, n_others, self.n_step_future, 2)

        # Build output container
        anchor_container = {
            "selected_anchors": selected_anchor_trajectories,
            "selected_embeddings": anchor_tokens,
            "selected_anchor_indices": selected_anchor_indices.squeeze(-1),
            "adapted_queries": adapted_queries,
            "adapted_queries_128": adapted_queries_128,
        }
        if gqa_attn_weights is not None:
            anchor_container["gqa_attn_weights"] = gqa_attn_weights

        return {
            "valid_mask": valid_mask,
            "conf": conf.unsqueeze(0),        # [1, B, T, n_pred_head]
            "pred": pred.unsqueeze(0),        # [1, B, T, n_pred_head, n_step_future, pred_dim]
            "y_pred_others": y_hat_others,    # [B, T, n_others, n_step_future, 2]
            "pred_pos_logits": None,
            "offset_pred": offset_pred.unsqueeze(0) if offset_pred is not None else None,
            "gqa_entropy_loss": gqa_entropy_loss,
            "anchor_container": anchor_container,
        }
