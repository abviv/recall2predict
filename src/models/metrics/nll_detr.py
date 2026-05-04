# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import logging
from typing import Dict, Optional, Tuple

import einops
import torch
from omegaconf import ListConfig
from src.HPTR.src.utils.transform_utils import torch_pos2global, torch_rad2rot
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torchmetrics.metric import Metric
from src.models.data_alignment.sdf_utils import sample_sdf_at_world_points


log = logging.getLogger(__name__)

def compute_nll_mtr(dmean: Tensor, cov: Tensor) -> Tensor:
    dx = dmean[..., 0]
    dy = dmean[..., 1]
    sx = cov[..., 0, 0]
    sy = cov[..., 1, 1]
    rho = torch.tanh(cov[..., 1, 0])  # mtr uses clamp to [-0.5, 0.5]
    one_minus_rho2 = 1 - rho ** 2
    log_prob = (
        torch.log(sx)
        + torch.log(sy)
        + 0.5 * torch.log(one_minus_rho2)
        + 0.5 / one_minus_rho2 * ((dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * dx * dy / (sx * sy))
    )
    return log_prob


class NllMetrics(Metric):
    """TorchMetrics wrapper for training loss + logging summaries.

    Important conventions used throughout this class:
    - `error_*` and `counter_*` are running sums (detached) used for epoch-level logging.
    - `_last_total_loss` stores the *current batch* loss with gradients, so the training
      loop can backprop. It is intentionally not a TorchMetrics state.
    - `compute()` normalizes running sums to produce logging-friendly metrics and
      returns `_last_total_loss` as the loss when available.
    """
    full_state_update = False

    def __init__(
        self,
        prefix: str,
        winner_takes_all: str,
        p_rand_train_agent: float,
        n_decoders: int, # inactive and set to 1
        n_pred: int,
        l_pos: str,
        n_step_add_train_agent: ListConfig,
        focal_gamma_conf: ListConfig,
        w_conf: ListConfig,
        w_pos: ListConfig,
        w_yaw: ListConfig,  # cos
        w_spd: ListConfig,  # huber
        w_vel: ListConfig,  # huber
        w_raster: ListConfig,
        w_anchor_diversity: ListConfig, # For orthogonal property
        w_anchor_selection: ListConfig,  # Offset regression loss weights
        visualize_probmap: bool,
        visualize_anchor_selection: bool = False,
        lambda_motion: float = 1.0,
        lambda_raster: float = 1.0,
        lambda_anchor: float = 1.0,
        lambda_endpoint: float = 1.0,
        lambda_anchor_endpoint: Optional[float] = None,
        lambda_anchor_as_traj: Optional[float] = None,
        n_anchor_proposals: int = 8,  # Number of latent queries (Q) from trajectory selector
        anchor_softmin_tau: float = 1.0,  # Temperature for soft-min. Set to -1 for winner-takes-all.
        offset_huber_delta: float = 1.0,  # Huber loss delta parameter for offset regression
        offset_regularization_lambda: float = 0.1,
        raster_beta_m: float = 0.5,
        endpoint_relative_to: str = "anchor",  # 'anchor' (anchor-relative) or 'none' (direct absolute)
        conf_label_smoothing: float = 0.0,
        lambda_others: float = 1.0,
        loaded_embeddings: Optional[torch.Tensor] = None,
        traj_tensor: Optional[torch.Tensor] = None,
        cfg: Dict = None, # Added cfg parameter
    ) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.winner_takes_all = winner_takes_all
        self.p_rand_train_agent = p_rand_train_agent
        self.n_decoders = n_decoders
        self.n_pred = n_pred
        self.l_pos = l_pos
        self.n_step_add_train_agent = n_step_add_train_agent
        self.focal_gamma_conf = list(focal_gamma_conf)
        self.w_conf = list(w_conf)
        self.w_pos = list(w_pos)
        self.w_yaw = list(w_yaw)
        self.w_spd = list(w_spd)
        self.w_vel = list(w_vel)
        self.w_raster = list(w_raster)
        self.w_anchor_diversity = list(w_anchor_diversity)
        self.w_anchor_selection = list(w_anchor_selection)
        self.visualize_probmap = visualize_probmap
        self.visualize_anchor_selection = visualize_anchor_selection
        self.lambda_motion = lambda_motion
        self.lambda_raster = lambda_raster
        # Backward-compatible aliases:
        # - lambda_endpoint -> lambda_anchor_endpoint (old anchor endpoint loss)
        # - lambda_anchor -> lambda_anchor_as_traj (full-trajectory anchor loss)
        self.lambda_anchor_endpoint = (
            lambda_anchor_endpoint if lambda_anchor_endpoint is not None else lambda_endpoint
        )
        self.lambda_anchor_as_traj = (
            lambda_anchor_as_traj if lambda_anchor_as_traj is not None else lambda_anchor
        )
        self.offset_huber_delta = offset_huber_delta  
        self.offset_regularization_lambda = offset_regularization_lambda
        self.raster_beta_m = raster_beta_m
        self.n_anchor_proposals = n_anchor_proposals
        self.anchor_softmin_tau = anchor_softmin_tau
        self.endpoint_relative_to = endpoint_relative_to
        self.conf_label_smoothing = conf_label_smoothing
        self.lambda_others = lambda_others

        # Running sums for epoch-level logging (detached in `update`).
        self.add_state("counter_traj", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter_conf", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_pos", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_conf", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_yaw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_spd", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_vel", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_raster", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_anchor_diversity", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("winner_l1_distances", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("winner_l1_count", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_anchor_quality", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_offset_regression", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_anchor_endpoint", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_anchor_as_traj", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_others", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter_others", default=torch.tensor(0.0), dist_reduce_fx="sum")

        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                self.add_state(f"counter_d{i}_p{j}", default=tensor(0.0), dist_reduce_fx="sum")
                self.add_state(f"conf_d{i}_p{j}", default=tensor(0.0), dist_reduce_fx="sum")

        # Store trajectory embeddings if provided
        if loaded_embeddings is not None:
            self.register_buffer("loaded_embeddings", loaded_embeddings)
        if traj_tensor is not None:
            self.register_buffer("traj_tensor", traj_tensor)
            log.info(f"Loaded trajectory tensor with shape: {traj_tensor.shape}")
        # Max magnitude clamp for endpoint offsets in meters (loss-level). None or <=0 disables.
        self.endpoint_max_offset_m = getattr(cfg, 'endpoint_max_offset_m', 15.0)

        # Keep the last per-batch loss tensor (with graph) so the training
        # loop can backpropagate through it. This is intentionally not a
        # TorchMetrics state to avoid detaching from autograd.
        self._last_total_loss = None

    def _compute_endpoint_direct_loss(
        self,
        endpoint_pred: Tensor,         # [B, T, n_anchors, 2]
        gt_pos: Tensor,                # [B, T, n_steps, 2]
        valid_mask: Tensor,            # [B, T]
        agent_type_weights: Tensor,    # [B, T]
        temperature: float = 1.0,
        delta: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Direct endpoint loss without anchor dependency.

        Mirrors the anchor-relative formulation but operates directly on predicted endpoints.
        Applies smooth L1 on the endpoint deltas and aggregates proposals via soft-min weights.
        Returns (total_loss, final_endpoint_loss_for_logging, offset_regularization_loss=0).
        """
        beta = max(delta, 1e-6)
        temperature = max(temperature, 1e-6)

        gt_endpoints = gt_pos[..., -1, :].unsqueeze(2).to(endpoint_pred)  # [B, T, 1, 2]
        proposal_deltas = endpoint_pred - gt_endpoints  # [B, T, n_anchors, 2]
        proposal_distances = torch.norm(proposal_deltas, dim=-1)  # [B, T, n_anchors]

        valid_mask_bool = valid_mask.bool()
        valid_mask_expanded = valid_mask_bool.unsqueeze(-1).expand_as(proposal_distances)
        valid_mask_float = valid_mask_expanded.to(endpoint_pred.dtype)
        valid_mask_scalar = valid_mask_bool.to(endpoint_pred.dtype)

        with torch.no_grad():
            raw_weights = F.softmax(-proposal_distances / temperature, dim=-1)
            weights = raw_weights * valid_mask_float
            weights_sum = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(
                weights_sum > 0,
                weights / weights_sum.clamp_min(1e-9),
                torch.zeros_like(weights),
            )

        gt_endpoints_expanded = gt_endpoints.expand_as(endpoint_pred)
        proposal_loss = F.smooth_l1_loss(
            endpoint_pred,
            gt_endpoints_expanded,
            reduction='none',
            beta=beta,
        ).sum(dim=-1)  # [B, T, n_anchors]
        proposal_loss = proposal_loss * valid_mask_float

        final_endpoint_loss_per_agent = (weights * proposal_loss).sum(dim=-1)  # [B, T]

        agent_type_weights = agent_type_weights.to(endpoint_pred.dtype)
        weighted = final_endpoint_loss_per_agent * agent_type_weights * valid_mask_scalar
        total = weighted.sum()

        with torch.no_grad():
            detached_loss = F.smooth_l1_loss(
                endpoint_pred.detach(),
                gt_endpoints_expanded.detach(),
                reduction='none',
                beta=beta,
            ).sum(dim=-1)  # [B, T, n_anchors]
            large_value = torch.finfo(detached_loss.dtype).max
            best_endpoint_loss_unweighted, _ = torch.min(
                detached_loss.masked_fill(~valid_mask_expanded, large_value), dim=-1
            )
            best_endpoint_loss_unweighted = torch.where(
                valid_mask_bool, best_endpoint_loss_unweighted, torch.zeros_like(best_endpoint_loss_unweighted)
            )
            final_endpoint_loss_for_logging = (
                best_endpoint_loss_unweighted * agent_type_weights * valid_mask_scalar
            ).sum()

        zero_reg = torch.tensor(0.0, device=endpoint_pred.device)
        return total, final_endpoint_loss_for_logging, zero_reg

    def _compute_endpoint_regression_loss(
        self,
        selected_anchors: Tensor,      # [B, T, n_anchors, n_steps, 2]
        offset_pred: Tensor,           # [B, T, n_anchors, 2]
        gt_pos: Tensor,                # [B, T, n_steps, 2]
        valid_mask: Tensor,            # [B, T]
        agent_type_weights: Tensor,    # [B, T]
        regularization_weight: float = 0.01, #Controls penalty on large offsets
        temperature: float = 1.0,      # softmin temperature
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes a dual-objective endpoint loss using a soft-min approach for stability.
        """
        anchor_endpoints = selected_anchors[..., -1, :] # [B, T, n_anchors, 2]
        gt_endpoints = gt_pos[..., -1, :].unsqueeze(2) # [B, T, 1, 2] for broadcasting
        
        # --- Calculate distances for ALL anchors ---
        anchor_distances = torch.norm(anchor_endpoints - gt_endpoints, dim=-1) # [B, T, n_anchors]
        
        # Use negative distances so that smaller distances get higher weights after softmax.
        # The temperature 'tau' controls the softness of the minimum.
        # Lower temp -> harder selection (closer to argmin). Higher temp -> softer selection.
        with torch.no_grad(): # Don't backprop through the weight calculation itself
            weights = F.softmax(-anchor_distances / temperature, dim=-1)

        # Clamp offsets for endpoint accuracy loss, but preserve raw offsets for regularization
        # so oversized predictions remain penalized.
        offset_pred_for_reg = offset_pred
        if self.endpoint_max_offset_m is not None and self.endpoint_max_offset_m > 0:
            _norm = torch.norm(offset_pred, dim=-1, keepdim=True).clamp(min=1e-8)
            _scale = torch.clamp(self.endpoint_max_offset_m / _norm, max=1.0)
            offset_pred_clamped = offset_pred * _scale
        else:
            offset_pred_clamped = offset_pred

        corrected_endpoints = anchor_endpoints + offset_pred_clamped
        corrected_distances = torch.norm(corrected_endpoints - gt_endpoints, dim=-1) # [B, T, n_anchors]
        # === COMPONENT 1: Weighted Accuracy Loss (L_accuracy) ===
        # Calculate Huber loss for ALL anchors
        final_endpoint_loss_all_anchors = F.huber_loss(
            corrected_distances,
            torch.zeros_like(corrected_distances),
            reduction='none',
            delta=self.offset_huber_delta
        )
        # Apply the soft-min weights
        final_endpoint_loss_per_agent = (weights * final_endpoint_loss_all_anchors).sum(dim=-1) # [B, T]

        # === COMPONENT 2: Weighted Offset Regularization Loss (L_regularization) ===
        # This part should allow us to maintain the offset_preds to be controllable. Thus
        # ensuring that we are not taking the offset_preds for granted and allowing it to 
        # carry the burden of the anchor selection. So we use this component to ensure that
        # small corrections are encouraged. 
         
        # Calculate L2 norm (magnitude) for ALL predicted offsets (pre-clamp to penalize large predictions)
        offset_magnitudes = torch.norm(offset_pred_for_reg, dim=-1)  # [B, T, n_anchors]
        
        # Apply Huber loss to all magnitudes
        offset_reg_loss_all_anchors = F.huber_loss(
            offset_magnitudes,
            torch.zeros_like(offset_magnitudes),
            reduction='none',
            delta=self.offset_huber_delta
        )
        # Apply the same soft-min weights
        offset_regularization_loss_per_agent = (weights * offset_reg_loss_all_anchors).sum(dim=-1) # [B, T]

        # === APPLY MASKS and WEIGHTS, THEN SUM ===
        weighted_endpoint_loss = (final_endpoint_loss_per_agent * agent_type_weights * valid_mask.float())
        weighted_regularization_loss = (offset_regularization_loss_per_agent * agent_type_weights * valid_mask.float())

        final_endpoint_loss = weighted_endpoint_loss.sum()
        offset_regularization_loss = weighted_regularization_loss.sum()

        total_loss = final_endpoint_loss + regularization_weight * offset_regularization_loss
        
        # For logging, find the best mode's error without gradients
        with torch.no_grad():
            best_distance, best_idx = torch.min(corrected_distances, dim=-1)
            # Use the same masking and weighting for a comparable metric
            best_endpoint_loss_unweighted = F.huber_loss(
                best_distance, torch.zeros_like(best_distance), reduction='none', delta=self.offset_huber_delta
            )
            final_endpoint_loss_for_logging = (best_endpoint_loss_unweighted * agent_type_weights * valid_mask.float()).sum()

        # The loss returned for backprop is total_loss. The others are for logging.
        # Returning `final_endpoint_loss_for_logging` in the second position to keep metric tracking consistent with the old logic.
        return total_loss, final_endpoint_loss_for_logging, offset_regularization_loss

    def _compute_anchor_trajectory_loss(
        self,
        selected_anchors: Tensor,      # [B, T, n_anchors, n_steps, 2]
        gt_pos: Tensor,                # [B, T, n_steps, 2]
        valid_mask_steps: Tensor,      # [B, T, n_steps]
        agent_type_weights: Tensor,    # [B, T]
        temperature: float = 1.0,
        delta: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full-trajectory anchor loss.
        Compares each anchor trajectory to GT across all steps, masks invalid steps,
        then aggregates anchors with a soft-min weighting.
        Returns (total_loss, best_anchor_loss_for_logging).
        """
        beta = max(delta, 1e-6)
        temperature = max(temperature, 1e-6)

        gt_traj = gt_pos.unsqueeze(2)  # [B, T, 1, n_steps, 2]
        per_step_loss = F.smooth_l1_loss(
            selected_anchors,
            gt_traj,
            reduction='none',
            beta=beta,
        ).sum(dim=-1)  # [B, T, n_anchors, n_steps]

        valid_mask_steps = valid_mask_steps.bool()
        valid_steps_float = valid_mask_steps.unsqueeze(2).to(per_step_loss.dtype)
        per_step_loss = per_step_loss * valid_steps_float

        step_counts = valid_steps_float.sum(dim=-1).clamp_min(1.0)  # [B, T, n_anchors]
        per_anchor_loss = per_step_loss.sum(dim=-1) / step_counts  # [B, T, n_anchors]

        valid_agent = valid_mask_steps.any(-1)  # [B, T]
        valid_agent_float = valid_agent.to(per_anchor_loss.dtype)
        valid_anchor_mask = valid_agent_float.unsqueeze(-1).expand_as(per_anchor_loss)

        with torch.no_grad():
            raw_weights = F.softmax(-per_anchor_loss / temperature, dim=-1)
            weights = raw_weights * valid_anchor_mask
            weights_sum = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(
                weights_sum > 0,
                weights / weights_sum.clamp_min(1e-9),
                torch.zeros_like(weights),
            )

        final_loss_per_agent = (weights * per_anchor_loss).sum(dim=-1)  # [B, T]

        agent_type_weights = agent_type_weights.to(per_anchor_loss.dtype)
        weighted = final_loss_per_agent * agent_type_weights * valid_agent_float
        total_loss = weighted.sum()

        with torch.no_grad():
            large_value = torch.finfo(per_anchor_loss.dtype).max
            best_anchor_loss_unweighted, _ = torch.min(
                per_anchor_loss.masked_fill(~valid_anchor_mask.bool(), large_value), dim=-1
            )
            best_anchor_loss_unweighted = torch.where(
                valid_agent, best_anchor_loss_unweighted, torch.zeros_like(best_anchor_loss_unweighted)
            )
            best_anchor_loss_for_logging = (
                best_anchor_loss_unweighted * agent_type_weights * valid_agent_float
            ).sum()

        return total_loss, best_anchor_loss_for_logging

    def update(
        self,
        pred_valid: Tensor,
        pred_conf: Tensor,
        pred_pos: Tensor,          # Agent-Local [n_decoder, B, T, n_pred, n_step_future, 2]
        pred_spd: Optional[Tensor],
        pred_vel: Optional[Tensor],
        pred_yaw_bbox: Optional[Tensor],
        pred_cov: Optional[Tensor],
        ref_role: Tensor,
        ref_type: Tensor,
        ref_pos: Tensor,          # Agent pose relative to SDC [B, T, 1, 2]
        ref_rot: Tensor,          # Agent rotation relative to SDC [B, T, 2, 2]
        gt_valid: Tensor,
        gt_pos: Tensor,           # Agent-Local [B, T, n_step_future, 2]
        gt_spd: Tensor,
        gt_vel: Tensor,
        gt_yaw_bbox: Tensor,
        gt_cmd: Tensor,
        gt_sdf_map: Optional[Tensor] = None,  # Signed distance field [B, H, W] in meters
        gt_sim2_R: Optional[Tensor] = None,
        gt_sim2_t: Optional[Tensor] = None,
        gt_sim2_s: Optional[Tensor] = None,
        gt_scenario_center: Optional[Tensor] = None, # SDC center in World [B, 2]
        gt_scenario_yaw: Optional[Tensor] = None,    # SDC yaw in World [B]
        gt_sdf_map_orig_dims: Optional[Tensor] = None,
        selected_anchors: Optional[Tensor] = None,
        selected_embeddings: Optional[Tensor] = None,
        adapted_queries_128: Optional[Tensor] = None,
        offset_pred: Optional[Tensor] = None,  # [B, T, 2] - Predicted endpoint offsets
        y_pred_others: Optional[Tensor] = None,      # [B, T, n_others, n_step_future, 2]
        gt_y_disp_others: Optional[Tensor] = None,  # [B, T, n_others, n_step_future, 2] (displacement)
        gt_other_valid: Optional[Tensor] = None,     # [B, T, n_others, n_step_future]
        **kwargs,
    ) -> None:
        """
        Args:
            B: batch size, T: n_target
            pred_valid: [B, T], bool
            pred_conf: [n_decoder, B, T, n_pred], not normalized!
            pred_pos: [n_decoder, B, T, n_pred, n_step_future, 2]
            pred_spd: [n_decoder, B, T, n_pred, n_step_future, 1]
            pred_vel: [n_decoder, B, T, n_pred, n_step_future, 2]
            pred_yaw_bbox: [n_decoder, B, T, n_pred, n_step_future, 1]
            pred_cov: [n_decoder, B, T, n_pred, n_step_future, 2, 2]
            gt_valid: [B, T, n_step_future], bool
            gt_pos: [B, T, n_step_future, 2]
            gt_spd: [B, T, n_step_future, 1]
            gt_vel: [B, T, n_step_future, 2]
            gt_yaw_bbox: [B, T, n_step_future, 1]
            gt_sdf_map: [B, H, W] signed distance field in meters
            gt_sdf_map_orig_dims: [B, 2] Original dimensions (H, W) of each SDF map before padding
            ref_role: [B, T, 3], one hot bool [sdc=0, interest=1, predict=2]
            ref_type: [B, T, 3], one hot bool [veh=0, ped=1, cyc=2]
            agent_cmd: [B, T, 8], one hot bool
            selected_anchors: [B*T, n_anchor_proposals, k(1), n_step_future, 2]
            selected_embeddings: [B*T, n_anchor_proposals, hidden_dim]
            offset_pred: [B, T, 2], predicted endpoint offsets (delta_x, delta_y)
        """
        n_agent_type = ref_type.shape[-1]
        n_decoder, B, T, n_pred = pred_conf.shape
        assert (ref_role.any(-1) & pred_valid == ref_role.any(-1)).all(), "All relevant agents shall be predicted!"

        # --- 1) Build availability masks (valid steps / agents)
        avails = ref_role.any(-1)  # [B, T]
        # add rand agents for training
        if self.p_rand_train_agent > 0:
            avails = avails | (torch.bernoulli(self.p_rand_train_agent * torch.ones_like(avails)).bool())
        # add long tracked agents for training
        _track_len = gt_valid.sum(-1)  # [B, T]
        for i in range(n_agent_type):
            if self.n_step_add_train_agent[i] > 0:
                avails = avails | (ref_type[:, :, i] & (_track_len > self.n_step_add_train_agent[i]))

        avails = gt_valid & avails.unsqueeze(-1)  # [B, T, n_step_future]
        avails = avails.unsqueeze(0).expand(n_decoder, -1, -1, -1)  # [n_decoder, B, T, n_step_future]
        if n_decoder > 1:
            # [n_decoder], randomly train ensembles with 50% of chance
            mask_ensemble = torch.bernoulli(0.5 * torch.ones_like(pred_conf[:, 0, 0, 0])).bool()
            # make sure at least one ensemble is trained
            if not mask_ensemble.any():
                mask_ensemble[torch.randint(0, n_decoder, (1,))] |= True
            avails = avails & mask_ensemble[:, None, None, None]
        # [n_decoder, B, T, n_pred, n_step_future]
        avails_full_pred_shape = avails.unsqueeze(3).expand(-1, -1, -1, n_pred, -1)

        # [n_decoder, B, T, n_pred], per ensemble
        pred_conf_normalized = torch.softmax(pred_conf, dim=-1)

        # --- Track confidence histogram (per mode)
        _prob = pred_conf_normalized.masked_fill(~(pred_valid[None, :, :, None]), 0.0)
        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                x = getattr(self, f"conf_d{i}_p{j}")
                # Implicity assignment of conf_d0_p0, conf_d0_p1, conf_d0_p2, etc
                # due to add_state of torchmetrics
                x += (_prob[i, :, :, j] * (avails_full_pred_shape[i, :, :, j].any(-1))).sum()

        # --- Winner-takes-all mode selection
        with torch.no_grad():
            decoder_idx = torch.arange(n_decoder)[:, None, None, None]  # [n_decoder, 1, 1, 1]
            scene_idx = torch.arange(B)[None, :, None, None]  # [1, B, 1, 1]
            agent_idx = torch.arange(T)[None, None, :, None]  # [1, 1, T, 1]

            if "hard1" in self.winner_takes_all:
                # [n_decoder, B, T, n_pred, n_step_future, 2]
                dist = torch.norm(pred_pos - gt_pos[None, :, :, None, :, :], dim=-1)
                dist = dist.masked_fill(~avails_full_pred_shape, 0.0).sum(-1)  # [n_decoder, B, T, n_pred]
                if "joint" in self.winner_takes_all:
                    dist = dist.sum(2, keepdim=True)  # [n_decoder, B, 1, n_pred]
                k_top = int(self.winner_takes_all[-1])
                i = torch.randint(high=k_top, size=())
                # [n_decoder, B, T, 1]
                mode_idx = dist.topk(k_top, dim=-1, largest=False, sorted=False)[1][..., [i]]
            elif self.winner_takes_all == "cmd":
                assert n_pred == gt_cmd.shape[-1]
                mode_idx = (gt_cmd + 0.0).argmax(-1, keepdim=True)  # [B, T, 1]
                mode_idx = mode_idx.unsqueeze(0).expand(n_decoder, -1, -1, -1)  # [n_decoder, B, T, 1]
            else: # Default to choosing the first mode if not specified or recognized
                mode_idx = torch.zeros((n_decoder, B, T, 1), dtype=torch.long, device=pred_pos.device)

            # Track hard assignment histogram: [n_decoder, B, T, n_pred]
            counter_modes = torch.nn.functional.one_hot(mode_idx.squeeze(-1), self.n_pred)
            for i in range(self.n_decoders):
                for j in range(self.n_pred):
                    x = getattr(self, f"counter_d{i}_p{j}")
                    x += (counter_modes[i, :, :, j] * (avails_full_pred_shape[i, :, :, j].any(-1))).sum()

        # --- Update counters for logging
        # avails_full_pred_shape: [n_decoder, B, T, n_pred, n_step_future]
        avails_win_mode_steps = avails_full_pred_shape[decoder_idx, scene_idx, agent_idx, mode_idx] # [n_d, n_s, n_a, 1, n_step_future]
        # Update counters for logging (detach to avoid growing the graph).
        self.counter_traj += avails_win_mode_steps.sum().detach()
        
        avails_agent_win_mode = avails_win_mode_steps[:,:,:,0,:].any(-1) # [n_d, n_s, n_a] (True if agent is valid for any step in win mode)
        self.counter_conf += avails_agent_win_mode.sum().detach()

        # --- Per-agent loss weights
        focal_gamma_conf_w = 0
        w_conf_w, w_pos_w, w_yaw_w, w_spd_w, w_vel_w = 0, 0, 0, 0, 0
        w_raster_w, w_anchor_diversity_w, w_anchor_selection_w = 0, 0, 0
        
        for i in range(n_agent_type):  # [B, T]
            focal_gamma_conf_w += ref_type[:, :, i] * self.focal_gamma_conf[i]
            w_conf_w += ref_type[:, :, i] * self.w_conf[i]
            w_pos_w += ref_type[:, :, i] * self.w_pos[i]
            w_yaw_w += ref_type[:, :, i] * self.w_yaw[i]
            w_spd_w += ref_type[:, :, i] * self.w_spd[i]
            w_vel_w += ref_type[:, :, i] * self.w_vel[i]
            w_raster_w += ref_type[:, :, i] * self.w_raster[i] # raster loss weight
            w_anchor_diversity_w += ref_type[:, :, i] * self.w_anchor_diversity[i] # anchor diversity loss weight
            w_anchor_selection_w += ref_type[:, :, i] * self.w_anchor_selection[i] # offset regression loss weight

        # --- Confidence loss (per-agent)
        # pred_conf_normalized: [n_decoder, B, T, n_pred]
        pred_conf_win_mode = pred_conf_normalized[decoder_idx, scene_idx, agent_idx, mode_idx] # [n_decoder, B, T, 1]
        if self.conf_label_smoothing > 0:
            with torch.no_grad():
                # Use endpoint distance for more stable soft targets (correlates with anchor selection)
                gt_endpoint = gt_pos[..., -1, :]  # [B, T, 2]
                pred_endpoints = pred_pos[..., -1, :]  # [n_decoder, B, T, n_modes, 2]
                gt_endpoint_exp = gt_endpoint.unsqueeze(0).unsqueeze(3)  # [1, B, T, 1, 2]
                endpoint_dist = torch.norm(pred_endpoints - gt_endpoint_exp, dim=-1)  # [n_decoder, B, T, n_modes]
                soft_targets = F.softmax(-endpoint_dist / self.conf_label_smoothing, dim=-1)

            log_probs = torch.log(pred_conf_normalized.clamp(min=1e-9))
            soft_ce_loss = -(soft_targets * log_probs).sum(dim=-1, keepdim=True)
            conf_error_per_agent = soft_ce_loss * w_conf_w[None, :, :, None]
        else:
            focal_gamma_term = torch.pow(1 - pred_conf_win_mode, focal_gamma_conf_w[None, :, :, None])
            conf_error_per_agent = (-torch.log(pred_conf_win_mode.clamp(min=1e-9)) * w_conf_w[None, :, :, None] * focal_gamma_term)
        # Accumulate running sum for logging (detach to avoid growing the graph).
        self.error_conf += conf_error_per_agent.masked_fill(~avails_agent_win_mode.unsqueeze(-1), 0.0).sum().detach()
        
        # NOTE: Select the winning mode's prediction *before* transforming
        pred_pos_win_mode = pred_pos[decoder_idx, scene_idx, agent_idx, mode_idx] # [n_d, n_s, n_a, 1, n_step_future, 2] Agent-Local

        # --- Position loss (agent-local)
        if self.l_pos == "huber":
            errors_pos = F.huber_loss(pred_pos_win_mode, gt_pos[None, :, :, None, :, :], reduction="none").sum(-1)
        elif self.l_pos == "l2":
            errors_pos = torch.norm(pred_pos_win_mode - gt_pos[None, :, :, None, :, :], p=2, dim=-1)
        elif self.l_pos == "nll_mtr":
            pred_cov_win_mode = pred_cov[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_pos = compute_nll_mtr(pred_pos_win_mode - gt_pos[None, :, :, None, :, :], pred_cov_win_mode)
        elif self.l_pos == "nll_torch":
            pred_cov_win_mode = pred_cov[decoder_idx, scene_idx, agent_idx, mode_idx]
            gmm = MultivariateNormal(pred_pos_win_mode, scale_tril=pred_cov_win_mode)
            errors_pos = -gmm.log_prob(gt_pos[None, :, :, None, :, :])
        # Batch error sums (keep graph for per-batch loss).
        batch_error_pos = (errors_pos * w_pos_w[None, :, :, None, None]).masked_fill(~avails_win_mode_steps, 0.0).sum()
        self.error_pos += batch_error_pos.detach()

        # error_spd
        if sum(self.w_spd) > 0 and pred_spd is not None:
            pred_spd_win_mode = pred_spd[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_spd = F.huber_loss(pred_spd_win_mode, gt_spd[None, :, :, None, :, :], reduction="none").squeeze(-1)
            batch_error_spd = (errors_spd * w_spd_w[None, :, :, None, None]).masked_fill(~avails_win_mode_steps, 0.0).sum()
            self.error_spd += batch_error_spd.detach()
        else:
            batch_error_spd = torch.tensor(0.0, device=pred_pos.device)

        # error_vel
        if sum(self.w_vel) > 0 and pred_vel is not None:
            pred_vel_win_mode = pred_vel[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_vel = F.huber_loss(pred_vel_win_mode, gt_vel[None, :, :, None, :, :], reduction="none").sum(-1)
            batch_error_vel = (errors_vel * w_vel_w[None, :, :, None, None]).masked_fill(~avails_win_mode_steps, 0.0).sum()
            self.error_vel += batch_error_vel.detach()
        else:
            batch_error_vel = torch.tensor(0.0, device=pred_pos.device)

        # error_yaw
        if sum(self.w_yaw) > 0 and pred_yaw_bbox is not None:
            pred_yaw_bbox_win_mode = pred_yaw_bbox[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_yaw = 1.0 - torch.cos(pred_yaw_bbox_win_mode - gt_yaw_bbox[None, :, :, None, :, :]).squeeze(-1)
            batch_error_yaw = (errors_yaw * w_yaw_w[None, :, :, None, None]).masked_fill(~avails_win_mode_steps, 0.0).sum()
            self.error_yaw += batch_error_yaw.detach()
        else:
            batch_error_yaw = torch.tensor(0.0, device=pred_pos.device)

        # error_raster (winner-only, per-agent normalized)
        if sum(self.w_raster) > 0 and gt_sdf_map is not None:
            required = (
                gt_sim2_R is not None
                and gt_sim2_t is not None
                and gt_sim2_s is not None
                and gt_scenario_center is not None
                and gt_scenario_yaw is not None
            )
            if not required:
                log.warning("Missing Sim2 or scenario transform components. Skipping SDF raster loss calculation.")
                batch_error_raster = torch.tensor(0.0, device=pred_pos.device)
            else:
                pred_pos_win_mode_local = pred_pos_win_mode.squeeze(3)
                ref_pos_metric = ref_pos if ref_pos.dim() == 4 else ref_pos.unsqueeze(2)
                pred_pos_sdc = torch_pos2global(
                    pred_pos_win_mode_local,
                    ref_pos_metric.unsqueeze(0).expand(n_decoder, -1, -1, -1, -1),
                    ref_rot.unsqueeze(0).expand(n_decoder, -1, -1, -1, -1),
                ).to(device=pred_pos.device, dtype=pred_pos.dtype)

                scenario_center = gt_scenario_center.to(device=pred_pos.device, dtype=pred_pos.dtype)
                scenario_rot = torch_rad2rot(gt_scenario_yaw.to(device=pred_pos.device, dtype=pred_pos.dtype))
                pred_pos_world = torch_pos2global(
                    pred_pos_sdc,
                    scenario_center.view(1, B, 1, 1, 2).expand(n_decoder, -1, T, -1, -1),
                    scenario_rot.view(1, B, 1, 2, 2).expand(n_decoder, -1, T, -1, -1),
                ).to(device=pred_pos.device, dtype=pred_pos.dtype)

                n_step_future = pred_pos_world.shape[-2]
                sdf_map = gt_sdf_map.to(device=pred_pos.device, dtype=pred_pos.dtype)
                sdf_map = sdf_map.unsqueeze(0).expand(n_decoder, -1, -1, -1).reshape(n_decoder * B, *sdf_map.shape[-2:])
                points_world = pred_pos_world.reshape(n_decoder * B, T * n_step_future, 2)

                sim2_R = gt_sim2_R.to(device=pred_pos.device, dtype=pred_pos.dtype)
                sim2_t = gt_sim2_t.to(device=pred_pos.device, dtype=pred_pos.dtype)
                sim2_s = gt_sim2_s.to(device=pred_pos.device, dtype=pred_pos.dtype).reshape(B, -1)
                sim2_R = sim2_R.unsqueeze(0).expand(n_decoder, -1, -1, -1).reshape(n_decoder * B, 2, 2)
                sim2_t = sim2_t.unsqueeze(0).expand(n_decoder, -1, -1).reshape(n_decoder * B, 2)
                sim2_s = sim2_s.unsqueeze(0).expand(n_decoder, -1, -1).reshape(n_decoder * B, -1)[:, 0]
                sdf_orig_dims = None
                if gt_sdf_map_orig_dims is not None:
                    sdf_orig_dims = gt_sdf_map_orig_dims.to(device=pred_pos.device, dtype=torch.long)
                    sdf_orig_dims = sdf_orig_dims.unsqueeze(0).expand(n_decoder, -1, -1).reshape(n_decoder * B, 2)

                _, offroad_distance_flat, _ = sample_sdf_at_world_points(
                    sdf_map=sdf_map,
                    points_world=points_world,
                    sim2_R=sim2_R,
                    sim2_t=sim2_t,
                    sim2_s=sim2_s,
                    orig_dims=sdf_orig_dims,
                )
                offroad_distance = offroad_distance_flat.reshape(n_decoder, B, T, n_step_future)

                per_step_raster = F.smooth_l1_loss(
                    offroad_distance,
                    torch.zeros_like(offroad_distance),
                    reduction="none",
                    beta=self.raster_beta_m,
                )
                valid_raster_steps = avails_win_mode_steps.squeeze(3)
                per_step_raster = per_step_raster.masked_fill(~valid_raster_steps, 0.0)

                valid_steps_per_agent = valid_raster_steps.sum(dim=-1).clamp(min=1.0)
                per_agent_raster = per_step_raster.sum(dim=-1) / valid_steps_per_agent

                valid_agents = avails_agent_win_mode.to(per_agent_raster.dtype)
                weighted_agent_raster = per_agent_raster * w_raster_w[None, :, :] * valid_agents
                batch_error_raster = weighted_agent_raster.sum()
                self.error_raster += batch_error_raster.detach()
            # else branch already sets batch_error_raster
        else:
            batch_error_raster = torch.tensor(0.0, device=pred_pos.device)
    
        total_endpoint_loss = torch.tensor(0.0, device=pred_pos.device)
        anchor_endpoint_loss = torch.tensor(0.0, device=pred_pos.device)
        anchor_as_traj_loss = torch.tensor(0.0, device=pred_pos.device)
        
        selected_anchors1 = None
        if selected_anchors is not None:
            selected_anchors1 = selected_anchors.squeeze(2)  # remove k(1)
            selected_anchors1 = einops.rearrange(
                selected_anchors1,
                '(B T) n_ach n_step_fut xy -> B T n_ach n_step_fut xy',
                B=B, T=T,
            )

        if sum(self.w_anchor_selection) > 0 and offset_pred is not None:
            # Calculate the anchor based endpoint loss. This punishes the anchors only for where its endpoint is.
            # This also uses offsets to do so
            if offset_pred.dim() == 5:  # [n_decoder, B, T, n_anchors, 2]
                offset_pred_single = offset_pred[0]
            else:
                offset_pred_single = offset_pred
            if self.endpoint_relative_to == 'anchor' and selected_anchors1 is not None:
                anchor_endpoint_loss, final_endpoint_loss, offset_regression_loss = self._compute_endpoint_regression_loss(
                    selected_anchors=selected_anchors1,
                    offset_pred=offset_pred_single,
                    gt_pos=gt_pos,
                    valid_mask=avails_agent_win_mode[0],
                    agent_type_weights=w_anchor_selection_w,
                    regularization_weight=self.offset_regularization_lambda,
                    temperature=self.anchor_softmin_tau,
                )
            elif self.endpoint_relative_to == 'none':
                # Direct absolute endpoint prediction per proposal
                # anchor_endpoint_loss, final_endpoint_loss, offset_regression_loss = self._compute_endpoint_direct_loss(
                #     endpoint_pred=offset_pred_single,
                #     gt_pos=gt_pos,
                #     valid_mask=avails_agent_win_mode[0],
                #     agent_type_weights=w_anchor_selection_w,
                #     temperature=max(self.anchor_softmin_tau, 1e-6),
                #     delta=self.offset_huber_delta,
                # )
                raise NotImplementedError
            else:
                anchor_endpoint_loss = torch.tensor(0.0, device=pred_pos.device)

            # Track components for logging (detached or no-grad).
            self.error_anchor_endpoint += anchor_endpoint_loss.detach()
            if 'final_endpoint_loss' in locals():
                self.error_anchor_quality += final_endpoint_loss
            if 'offset_regression_loss' in locals():
                self.error_offset_regression += offset_regression_loss

        # --- Full-trajectory anchor loss (encourage anchors to match GT path) ---
        if sum(self.w_anchor_selection) > 0 and selected_anchors1 is not None:
            anchor_as_traj_loss, anchor_traj_loss_for_logging = self._compute_anchor_trajectory_loss(
                selected_anchors=selected_anchors1,
                gt_pos=gt_pos,
                valid_mask_steps=avails[0],
                agent_type_weights=w_anchor_selection_w,
                temperature=self.anchor_softmin_tau,
                delta=self.offset_huber_delta,
            )
            # Running sum for logging.
            self.error_anchor_as_traj += anchor_as_traj_loss.detach()

        # Combine anchor endpoint + full-trajectory anchor loss with separate lambdas
        total_endpoint_loss = (
            self.lambda_anchor_endpoint * anchor_endpoint_loss
            + self.lambda_anchor_as_traj * anchor_as_traj_loss
        )
        # --- Anchor diversity loss (optional)
        if sum(self.w_anchor_diversity) > 0:
            batch_error_anchor_diversity = torch.tensor(0.0, device=pred_pos.device)
            if adapted_queries_128 is not None:
                # Query diversity (directly penalize query collapse)
                query_emb_norm = F.normalize(adapted_queries_128, dim=-1)
                query_sim = torch.matmul(query_emb_norm, query_emb_norm.transpose(-1, -2))
                n_batch, n_queries, _ = query_sim.shape
                identity = torch.eye(n_queries, device=query_sim.device).expand(n_batch, -1, -1)
                query_diversity_error = (query_sim - identity).pow(2)
                query_diversity_loss_per_agent = query_diversity_error.mean(dim=(-1, -2))
                valid_mask_flat = pred_valid.flatten()
                w_coeff_flat = w_anchor_diversity_w.flatten()
                query_diversity_loss = (query_diversity_loss_per_agent * w_coeff_flat).masked_fill(~valid_mask_flat, 0.0).sum()
                batch_error_anchor_diversity = batch_error_anchor_diversity + query_diversity_loss
                self.error_anchor_diversity += query_diversity_loss.detach()

            # Track anchor winner endpoint error (hard-min) for logging only.
            if selected_anchors1 is not None:
                with torch.no_grad():
                    endpoints_pred = selected_anchors1[..., -1, :]  # [B, T, n_anchors, 2]
                    gt_endpoints = gt_pos[..., -1, :].unsqueeze(2)  # [B, T, 1, 2]
                    dist_l1 = F.smooth_l1_loss(
                        endpoints_pred,
                        gt_endpoints,
                        reduction='none',
                        beta=self.offset_huber_delta,
                    ).sum(dim=-1)  # [B, T, n_anchors]
                    winner_error_per_agent, _ = dist_l1.min(dim=-1)  # [B, T]

                    valid_mask_agents = avails_agent_win_mode[0]
                    if valid_mask_agents.any():
                        valid_winner_distances = winner_error_per_agent.masked_select(valid_mask_agents)
                        if valid_winner_distances.numel() > 0:
                            self.winner_l1_distances += valid_winner_distances.sum()
                            self.winner_l1_count += valid_winner_distances.numel()

        # ---- Other-agent dense prediction loss (single-shot smooth L1 on displacements) ----
        batch_error_others = torch.tensor(0.0, device=pred_pos.device)
        if (
            self.lambda_others > 0
            and y_pred_others is not None
            and gt_y_disp_others is not None
            and gt_other_valid is not None
        ):
            # y_pred_others:   [B, T, n_others, n_step_future, 2]  (predicted displacement)
            # gt_y_disp_others:[B, T, n_others, n_step_future, 2]  (GT displacement)
            # gt_other_valid:  [B, T, n_others, n_step_future]
            n_others_pred = y_pred_others.shape[2]
            n_others_gt = gt_y_disp_others.shape[2]
            n_others = min(n_others_pred, n_others_gt)
            y_pred_others = y_pred_others[:, :, :n_others]
            gt_y_disp_others = gt_y_disp_others[:, :, :n_others]
            gt_other_valid = gt_other_valid[:, :, :n_others]

            others_avail = gt_other_valid.bool()  # [B, T, n_others, n_step_future]
            per_step_error = F.smooth_l1_loss(
                y_pred_others, gt_y_disp_others, reduction="none", beta=1.0
            ).sum(dim=-1)  # [B, T, n_others, n_step_future]
            per_step_error = per_step_error.masked_fill(~others_avail, 0.0)

            others_error_sum = per_step_error.sum()
            n_valid_steps = others_avail.sum().clamp(min=1.0)
            batch_error_others = others_error_sum / n_valid_steps

            self.error_others += others_error_sum.detach()
            self.counter_others += n_valid_steps.detach()

        # ---- Build per-batch loss used for backpropagation (retain graph) ----
        # This uses *batch-normalized* errors, not the running sums, so the loss
        # carries gradients for the current batch only.
        # Safeguard optional terms that might not be set in some configs/branches
        if 'batch_error_pos' not in locals():
            batch_error_pos = torch.tensor(0.0, device=pred_pos.device)
        if 'batch_error_yaw' not in locals():
            batch_error_yaw = torch.tensor(0.0, device=pred_pos.device)
        if 'batch_error_vel' not in locals():
            batch_error_vel = torch.tensor(0.0, device=pred_pos.device)
        if 'batch_error_spd' not in locals():
            batch_error_spd = torch.tensor(0.0, device=pred_pos.device)
        if 'batch_error_raster' not in locals():
            batch_error_raster = torch.tensor(0.0, device=pred_pos.device)
        if 'total_endpoint_loss' not in locals():
            total_endpoint_loss = torch.tensor(0.0, device=pred_pos.device)
        if 'batch_error_anchor_diversity' not in locals():
            batch_error_anchor_diversity = torch.tensor(0.0, device=pred_pos.device)

        counter_traj_batch = avails_win_mode_steps.sum().clamp(min=1.0)
        counter_conf_batch = avails_agent_win_mode.sum().clamp(min=1.0)

        # conf_error_per_agent is [n_d, n_s, n_a, 1]
        error_conf_sum_b = conf_error_per_agent.masked_fill(~avails_agent_win_mode.unsqueeze(-1), 0.0).sum()
        
        error_pos_norm_b = batch_error_pos / counter_traj_batch
        error_conf_norm_b = error_conf_sum_b / counter_conf_batch
        error_yaw_norm_b = batch_error_yaw / counter_traj_batch
        error_vel_norm_b = batch_error_vel / counter_traj_batch
        error_spd_norm_b = batch_error_spd / counter_traj_batch
        error_raster_norm_b = batch_error_raster / counter_conf_batch
        error_endpoint_norm_b = total_endpoint_loss / counter_conf_batch
        error_anchor_diversity_norm_b = batch_error_anchor_diversity / counter_conf_batch

        loss_motion_b = (error_pos_norm_b + error_conf_norm_b + error_yaw_norm_b + error_vel_norm_b + error_spd_norm_b) * self.lambda_motion
        loss_raster_b = error_raster_norm_b * self.lambda_raster
        loss_endpoint_b = error_endpoint_norm_b
        loss_others_b = batch_error_others * self.lambda_others
        # GQA entropy regularization (pre-scaled by lambda in model forward; 0.0 when disabled)
        gqa_entropy_loss = kwargs.get("gqa_entropy_loss", torch.tensor(0.0, device=pred_pos.device))
        # Keep a gradient-carrying tensor for the training step to backprop.
        self._last_total_loss = (
            loss_motion_b
            + loss_raster_b
            + loss_endpoint_b
            + error_anchor_diversity_norm_b
            + loss_others_b
            + gqa_entropy_loss
        )

    def compute(self) -> Dict[str, Tensor]:
        """
        Counter_traj: Number of valid trajectory points depends on n_step_future
        Counter_conf: Number of valid agents
        Error_pos: Position error (normalized)
        Error_conf: Confidence error (normalized)
        Error_yaw: Yaw error (normalized)
        Error_vel: Velocity error (normalized)
        Error_spd: Speed error (normalized)
        Error_raster: Raster error (normalized)
        Error_anchor_diversity: Anchor diversity error (normalized)
        Error_endpoint_regression: Endpoint regression error (normalized)
        Error_anchor_quality: Anchor quality error (normalized)
        Error_anchor_as_traj: Anchor full-trajectory error (normalized)
        Error_offset_correction: Offset correction error (normalized)

        Notes:
        - `error_*` are running sums accumulated in `update()` (detached).
        - `loss_*` are derived from normalized errors for logging.
        - `{prefix}/loss` returns the last per-batch loss when available.
        """
        # Normalize running sums (avoid division by zero).
        counter_traj = self.counter_traj.clamp_min(1.0)
        counter_conf = self.counter_conf.clamp_min(1.0)

        # Normalized errors (epoch-level logging)
        error_pos_normalized = self.error_pos / counter_traj
        error_conf_normalized = self.error_conf / counter_conf
        error_yaw_normalized = self.error_yaw / counter_traj
        error_vel_normalized = self.error_vel / counter_traj
        error_spd_normalized = self.error_spd / counter_traj
        error_raster_normalized = self.error_raster / counter_conf

        error_anchor_endpoint_normalized = self.error_anchor_endpoint / counter_conf
        error_anchor_as_traj_normalized = self.error_anchor_as_traj / counter_conf
        error_offset_regression_normalized = self.error_offset_regression / counter_conf 
        
        error_anchor_diversity_normalized = self.error_anchor_diversity / counter_conf
        error_anchor_quality_normalized = self.error_anchor_quality / counter_conf

        counter_others = self.counter_others.clamp_min(1.0)
        error_others_normalized = self.error_others / counter_others
        
        # --- Categorize, scale, and sum losses (logging only) ---
        # Category 1: Motion prediction
        loss_motion_prediction_unscaled = (
            error_pos_normalized
            + error_conf_normalized
            + error_yaw_normalized
            + error_vel_normalized
            + error_spd_normalized
        )
        loss_motion_prediction_scaled = loss_motion_prediction_unscaled * self.lambda_motion

        # Category 2: Endpoint regression (anchor-based)
        loss_endpoint_scaled = (
            error_anchor_endpoint_normalized * self.lambda_anchor_endpoint
            + error_anchor_as_traj_normalized * self.lambda_anchor_as_traj
        )

        # Category 3: SDF raster penalty
        loss_raster_scaled = error_raster_normalized * self.lambda_raster

        # Category 4: Other-agent dense prediction
        loss_others_scaled = error_others_normalized * self.lambda_others

        # Calculate the total loss from running averages (for logging)
        total_loss_logged = (
            loss_motion_prediction_scaled
            + loss_raster_scaled
            + loss_endpoint_scaled
            + error_anchor_diversity_normalized
            + loss_others_scaled
        )
        
        # Calculate average winner L1 distance
        avg_winner_l1_distance = self.winner_l1_distances / self.winner_l1_count.clamp_min(1.0)
        
        # Construct the output dictionary with various metrics.
        # Use the last per-batch loss for backprop if available (keeps gradients).
        loss_for_backprop = self._last_total_loss if self._last_total_loss is not None else total_loss_logged

        out_dict = {
            f"{self.prefix}/loss": loss_for_backprop,
            f"{self.prefix}/loss_logged": total_loss_logged,
            f"{self.prefix}/loss_motion_scaled": loss_motion_prediction_scaled,
            f"{self.prefix}/loss_raster_scaled": loss_raster_scaled,
            f"{self.prefix}/loss_motion_unscaled": loss_motion_prediction_unscaled, # Motion
            f"{self.prefix}/loss_endpoint_scaled": loss_endpoint_scaled,  # Anchor
            f"{self.prefix}/error_pos": error_pos_normalized, 
            f"{self.prefix}/error_conf": error_conf_normalized,
            f"{self.prefix}/error_yaw": error_yaw_normalized,
            f"{self.prefix}/error_vel": error_vel_normalized,
            f"{self.prefix}/error_spd": error_spd_normalized,
            f"{self.prefix}/error_raster": error_raster_normalized,
            f"{self.prefix}/error_anchor_endpoint": error_anchor_endpoint_normalized, # Anchor
            f"{self.prefix}/error_offset_regression": error_offset_regression_normalized,  # For tracking
            f"{self.prefix}/error_anchor_diversity": error_anchor_diversity_normalized,
            f"{self.prefix}/error_anchor_quality": error_anchor_quality_normalized,  # For tracking
            f"{self.prefix}/error_anchor_as_traj": error_anchor_as_traj_normalized,
            f"{self.prefix}/error_others": error_others_normalized,
            f"{self.prefix}/loss_others_scaled": loss_others_scaled,
            f"{self.prefix}/stats/avg_winner_l1_distance": avg_winner_l1_distance,
            f"{self.prefix}/stats/total_winner_count": self.winner_l1_count,
        }
        
        # Add per-mode confidence metrics if available
        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                out_dict[f"{self.prefix}/counter_d{i}_p{j}"] = getattr(self, f"counter_d{i}_p{j}")
                out_dict[f"{self.prefix}/conf_d{i}_p{j}"] = getattr(self, f"conf_d{i}_p{j}")

        return out_dict
