import hydra
import wandb
import numpy as np
import matplotlib.pyplot as plt
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from src.models.viz_components.plot_3d import (
    plot_motion_forecasts,
    mplfig_to_npimage,
    tensor_dict_to_cpu,
)
from src.models.viz_components.plot_motion import (
    plot_motion_2d,
    plot_motion_focal_track_multi_modality,
)
from src.models.viz_components.visualization_hooks import PredDictVisualizer
from src.models.viz_components.plot_anchor_selection import plot_anchor_selection
from src.models.viz_components.plot_endpoints import plot_endpoint_predictions
from src.models.data_alignment.load_embeddings import load_embeddings
from src.models.data_alignment.post_process_logits import PostProcessLogits

# Get a logger instance for this module
log = logging.getLogger(__name__)

class FutureMotion(LightningModule):
    def __init__(
        self,
        time_step_current: int,
        time_step_end: int,
        data_size: DictConfig,
        train_metric: DictConfig,
        waymo_metric: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        pre_processing: DictConfig,
        post_processing: DictConfig,
        n_video_batch: int,
        inference_repeat_n: int,
        inference_cache_map: bool,
        sub_womd: DictConfig,
        sub_av2: DictConfig,
        interactive_challenge: bool = False,
        wb_artifact: Optional[str] = None,
        plot_motion: bool = False,
        plot_probmap: bool = False,  # Added parameter to enable/disable probability map visualization
        plot_anchor_selection: bool = False,
        plot_endpoints: bool = False,
        plot_motion_focal_track: bool = False,
        control_temperatures: list = [-20, -10, 0, 10, 20],
        pretrained_emb_path: Optional[str] = None,
        w_scheduler: Optional[DictConfig] = None,  # None for backwards compatibility
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # accessible through self.hparams
        # --------------------- Load trajectory embeddings for consistent usage ---------------------
        self.loaded_embeddings = None
        self.traj_tensor = None
        self.tgt_w_conf = train_metric.w_conf
        # Load embeddings if trajectory selector uses softattn
        if (
            hasattr(model, "trajectory_selector")
            and model.trajectory_selector.get("use_layer") == "softattn"
        ):
            if pretrained_emb_path is None:
                raise ValueError(
                    "pretrained_emb_path must be provided for softattn trajectory selector"
                )

            subset_percentage = model.trajectory_selector.get("subset_percentage", 0.10)

            if pretrained_emb_path:
                log.info(f"Loading trajectory embeddings from {pretrained_emb_path}")
                try:
                    weights, _, traj_data = load_embeddings(
                        pretrained_emb_path=pretrained_emb_path,
                        subset_percentage=subset_percentage,
                        device=str(self.device),
                    )
                    # Store as buffers so they follow the module device automatically
                    self.register_buffer("loaded_embeddings_bank", weights)
                    self.register_buffer("traj_tensor_bank", traj_data)
                    self.loaded_embeddings = self.loaded_embeddings_bank
                    self.traj_tensor = self.traj_tensor_bank
                    log.info(
                        f"Successfully loaded {weights.shape[0]} trajectory embeddings with {weights.shape[1]} dimensions"
                    )
                except Exception as e:
                    log.error(f"Failed to load trajectory embeddings: {e}")
                    raise e
            else:
                log.warning(
                    "No pretrained_emb_path specified for softattn trajectory selector"
                )

        # Cache anchor endpoints for quick coverage checks
        if self.traj_tensor is not None:
            self.register_buffer(
                "traj_bank_endpoints", self.traj_tensor[:, -1, :].clone()
            )
        else:
            self.traj_bank_endpoints = None

        # --------------------- pre_processing ---------------------
        pre_proc_kwargs = {}
        pre_proc_modules = []
        for _, v in pre_processing.items():
            _pre_proc = hydra.utils.instantiate(
                v, time_step_current=time_step_current, data_size=data_size
            )
            pre_proc_modules.append(_pre_proc)
            pre_proc_kwargs.update(_pre_proc.model_kwargs)
        self.pre_processing = nn.ModuleList(pre_proc_modules)

        # --------------------- model ---------------------
        # Pass the loaded embeddings to the model if available
        model_kwargs = {**pre_proc_kwargs}
        if self.loaded_embeddings is not None:
            model_kwargs["loaded_embeddings"] = self.loaded_embeddings
            model_kwargs["traj_tensor"] = self.traj_tensor

        self.model = hydra.utils.instantiate(model, **model_kwargs, _recursive_=False)

        # Optionally attach FAISS full-memory retrieval to the trajectory selector
        try:
            ts_cfg = getattr(model, "trajectory_selector", None)
            if ts_cfg is not None and hasattr(self.model, "trajectory_selector"):
                faiss_index_path = ts_cfg.get("faiss_index_path", None)
                mem2static_path = ts_cfg.get("mem2static_path", None)
                # knobs (optional overrides)
                knn_topM = ts_cfg.get("knn_topM", None)
                knn_lambda = ts_cfg.get("knn_lambda", None)
                knn_metric = ts_cfg.get("knn_metric", None)
                if faiss_index_path and mem2static_path:
                    self.model.trajectory_selector.attach_faiss_from_files(
                        index_path=faiss_index_path,
                        mem2static_path=mem2static_path,
                        knn_topM=knn_topM,
                        knn_lambda=knn_lambda,
                        knn_metric=knn_metric,
                    )
                    log.info(
                        "Attached FAISS full-memory fusion to trajectory selector."
                    )
                elif faiss_index_path or mem2static_path:
                    log.warning(
                        "FAISS fusion requested but one of the paths is missing. Provide both faiss_index_path and mem2static_path."
                    )
        except Exception as e:
            log.warning(f"Could not attach FAISS memory: {e}")

        # --------------------- post_processing ---------------------
        self.post_processing = nn.Sequential(
            *[hydra.utils.instantiate(v) for _, v in post_processing.items()]
        )

        # --------------------- save submission files ---------------------
        self.sub_womd = hydra.utils.instantiate(
            sub_womd,
            k_futures=post_processing.waymo.k_pred,
            wb_artifact=wb_artifact,
            interactive_challenge=interactive_challenge,
        )
        self.sub_av2 = hydra.utils.instantiate(
            sub_av2, k_futures=post_processing.waymo.k_pred
        )

        # --------------------- metrics ---------------------
        # Pass trajectory embeddings to train_metric for loss calculation
        train_metric_kwargs = {}
        if self.loaded_embeddings is not None:
            train_metric_kwargs["loaded_embeddings"] = self.loaded_embeddings
            train_metric_kwargs["traj_tensor"] = self.traj_tensor

        self.train_metric = hydra.utils.instantiate(
            train_metric,  # computing NLL loss
            prefix="train",
            n_decoders=self.model.n_decoders,
            n_pred=self.model.n_pred,
            **train_metric_kwargs,
            # K_anchors will be passed from train_metric config, should match selector's k
        )
        self.waymo_metric = hydra.utils.instantiate(
            waymo_metric,  # computing waymo metrics
            prefix="waymo_pred",
            step_gt=time_step_end,
            step_current=time_step_current,
            interactive_challenge=interactive_challenge,
            n_agent=data_size["agent/valid"][-1],
        )
        self.pred_dict_visualizer = PredDictVisualizer()

        # Instantiate the logit post-processor if needed for classification models
        self.post_process_logits = None
        if self.hparams.model.get("motion_decoder", {}).get(
            "use_classification_head", False
        ):
            # The traj_tensor is loaded at the beginning of this __init__
            if self.traj_tensor is not None:
                self.post_process_logits = PostProcessLogits(
                    n_pred=self.model.n_pred, traj_tensor=self.traj_tensor
                )
            else:
                log.warning(
                    "Cannot instantiate PostProcessLogits: self.traj_tensor is None. "
                    "Visualization of classification logits will not work."
                )

        if self.hparams.model.get("motion_decoder", {}).get(
            "use_classification_tower_head", False
        ):
            # The traj_tensor is loaded at the beginning of this __init__
            if self.traj_tensor is not None:
                self.post_process_logits = PostProcessLogits(
                    n_pred=self.model.n_pred,
                    traj_tensor=self.traj_tensor,
                    use_classification_tower_head=True,
                )
            else:
                log.warning(
                    "Cannot instantiate PostProcessLogits: self.traj_tensor is None. "
                    "Visualization of classification logits will not work."
                )

    def _log_offset_stats(
        self,
        offset_pred: Optional[Tensor],
        prefix: str = "train",
        on_step: bool = True,
        on_epoch: bool = False,
    ) -> None:
        """Log offset prediction statistics to monitor magnitude and distribution.

        Args:
            offset_pred: Offset predictions tensor, shape varies:
                - [n_decoder, B, n_agents, n_anchors, 2] or
                - [B, n_agents, n_anchors, 2]
            prefix: Logging prefix ('train' or 'val')
            on_step: Whether to log on each step
            on_epoch: Whether to log on epoch end
        """
        if offset_pred is None:
            return

        try:
            with torch.no_grad():
                # Handle different tensor shapes
                if offset_pred.dim() == 5:  # [n_decoder, B, n_agents, n_anchors, 2]
                    offset = offset_pred[0]  # Take first decoder
                elif offset_pred.dim() == 4:  # [B, n_agents, n_anchors, 2]
                    offset = offset_pred
                else:
                    log.debug(f"Unexpected offset_pred shape: {offset_pred.shape}")
                    return

                # Compute magnitude per anchor: [B, n_agents, n_anchors]
                offset_magnitudes = torch.norm(offset, dim=-1)

                # Flatten for statistics (ignore batch structure)
                flat_magnitudes = offset_magnitudes.flatten()

                # Filter out any invalid values
                valid_mask = torch.isfinite(flat_magnitudes)
                if not valid_mask.any():
                    return
                valid_magnitudes = flat_magnitudes[valid_mask]

                # Core statistics
                self.log(
                    f"{prefix}/offset_mean_magnitude",
                    valid_magnitudes.mean(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/offset_max_magnitude",
                    valid_magnitudes.max(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/offset_std_magnitude",
                    valid_magnitudes.std(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/offset_median_magnitude",
                    valid_magnitudes.median(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )

                # Distribution tracking - percentage above thresholds
                n_valid = valid_magnitudes.numel()
                self.log(
                    f"{prefix}/offset_pct_above_5m",
                    (valid_magnitudes > 5.0).float().sum() / n_valid * 100,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/offset_pct_above_10m",
                    (valid_magnitudes > 10.0).float().sum() / n_valid * 100,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/offset_pct_above_15m",
                    (valid_magnitudes > 15.0).float().sum() / n_valid * 100,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                )

                # Percentiles for distribution shape
                if n_valid >= 10:  # Need enough samples for meaningful percentiles
                    self.log(
                        f"{prefix}/offset_p50",
                        torch.quantile(valid_magnitudes, 0.50),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                    self.log(
                        f"{prefix}/offset_p90",
                        torch.quantile(valid_magnitudes, 0.90),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                    self.log(
                        f"{prefix}/offset_p95",
                        torch.quantile(valid_magnitudes, 0.95),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                    self.log(
                        f"{prefix}/offset_p99",
                        torch.quantile(valid_magnitudes, 0.99),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )

                # X/Y component analysis (detect directional bias)
                offset_x = offset[..., 0].flatten()
                offset_y = offset[..., 1].flatten()
                valid_x = offset_x[torch.isfinite(offset_x)]
                valid_y = offset_y[torch.isfinite(offset_y)]

                if valid_x.numel() > 0:
                    self.log(
                        f"{prefix}/offset_x_mean",
                        valid_x.mean(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                    self.log(
                        f"{prefix}/offset_x_std",
                        valid_x.std(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                if valid_y.numel() > 0:
                    self.log(
                        f"{prefix}/offset_y_mean",
                        valid_y.mean(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )
                    self.log(
                        f"{prefix}/offset_y_std",
                        valid_y.std(),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )

        except Exception as e:
            # Don't let logging failures break training
            log.debug(f"Could not log offset stats: {e}")

    def _log_faiss_stats(
        self, prefix: str = "train", on_step: bool = True, on_epoch: bool = False
    ) -> None:
        """Log FAISS fusion statistics from trajectory selector if available.

        Args:
            prefix: Logging prefix ('train' or 'val')
            on_step: Whether to log on each step
            on_epoch: Whether to log on epoch end
        """
        try:
            if not hasattr(self.model, "trajectory_selector"):
                return

            selector = self.model.trajectory_selector
            if not hasattr(selector, "_last_faiss"):
                return

            faiss_stats = selector._last_faiss
            if faiss_stats is None:
                return

            # Log whether FAISS fusion was used
            faiss_used = faiss_stats.get("used", False)
            self.log(
                f"{prefix}/faiss_used",
                float(faiss_used),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
            )

            if faiss_used:
                # Log detailed FAISS statistics
                if faiss_stats.get("neighbors") is not None:
                    self.log(
                        f"{prefix}/faiss_neighbors",
                        float(faiss_stats["neighbors"]),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )

                if faiss_stats.get("mean_sim") is not None:
                    self.log(
                        f"{prefix}/faiss_mean_sim",
                        faiss_stats["mean_sim"],
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )

                if faiss_stats.get("mean_bias") is not None:
                    self.log(
                        f"{prefix}/faiss_mean_bias",
                        faiss_stats["mean_bias"],
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                    )

            # Log retrieval mode for reference (as a sanity check)
            # Only log once per epoch to avoid clutter
            if on_epoch and self.global_rank == 0:
                mode = faiss_stats.get("mode", "unknown")
                log.debug(f"FAISS retrieval mode: {mode}, used: {faiss_used}")

        except Exception as e:
            # Don't let logging failures break training
            log.debug(f"Could not log FAISS stats: {e}")

    def _schedule_conf_weight(self) -> None:
        """
        Schedule confidence loss weight with optional linear ramp.

        Control options via w_scheduler config:
            start_epoch=None or -1 -> Use tgt_w_conf (w_conf) throughout (no scheduling)
            start_epoch=N (N >= 0) -> Use base_w_conf until epoch N, then linear ramp to tgt_w_conf
            base_w_conf            -> Floor weight before ramp starts (default: 10% of tgt_w_conf)

        Example with start_epoch=20, max_epochs=30, tgt_w_conf=[10,10,10], base_w_conf=[1,1,1]:
            Epochs 0-20:  w_conf = [1.0, 1.0, 1.0]  (constant base)
            Epochs 20-30: w_conf linearly interpolates from base to target
            Epoch 30:     w_conf = [10.0, 10.0, 10.0]
        """
        if self.hparams.w_scheduler is None:
            return

        tm = getattr(self, "train_metric", None)
        trainer = getattr(self, "trainer", None)

        if tm is None or not hasattr(tm, "w_conf"):
            return
        if (
            trainer is None
            or not hasattr(trainer, "max_epochs")
            or trainer.max_epochs is None
            or trainer.max_epochs <= 0
        ):
            return

        # Get start_epoch - handle OmegaConf null properly
        start_epoch_raw = self.hparams.w_scheduler.get("start_epoch", None)

        max_epochs = trainer.max_epochs
        current_epoch = trainer.current_epoch

        # Debug: log once at start to see actual parsed value
        if (
            current_epoch == 0
            and getattr(self, "_w_scheduler_debug_logged", False) is False
        ):
            self._w_scheduler_debug_logged = True
            log.info(
                f"w_scheduler debug: start_epoch_raw={start_epoch_raw!r} (type={type(start_epoch_raw).__name__}), "
                f"tgt_w_conf={self.tgt_w_conf}"
            )

        # No scheduling if start_epoch is None, null, or negative - use target weights directly
        if start_epoch_raw is None or (
            isinstance(start_epoch_raw, (int, float)) and start_epoch_raw < 0
        ):
            tm.w_conf = list(self.tgt_w_conf)
            return

        # Convert to int in case it's read as float/string
        start_epoch = int(start_epoch_raw)

        # Base weight: configurable floor, defaults to 10% of target
        base_w_conf = self.hparams.w_scheduler.get("base_w_conf", None)
        if base_w_conf is None:
            base_w_conf = [0.1 * w for w in self.tgt_w_conf]
        else:
            base_w_conf = list(base_w_conf)

        # Phase 1: Before start_epoch - use base weights (constant floor)
        if current_epoch <= start_epoch:
            target = list(base_w_conf)
        # Phase 2: After start_epoch - linear ramp from base to target
        else:
            ramp_duration = max_epochs - start_epoch
            if ramp_duration <= 0:
                target = list(self.tgt_w_conf)
            else:
                # progress: 0.0 at start_epoch+1, 1.0 at max_epochs
                progress = min((current_epoch - start_epoch) / ramp_duration, 1.0)
                target = [
                    base + progress * (tgt - base)
                    for base, tgt in zip(base_w_conf, self.tgt_w_conf)
                ]

        target = [round(val, 4) for val in target]
        tm.w_conf = target

        # Log once per epoch at epoch boundary for debugging
        if current_epoch != getattr(self, "_last_logged_conf_epoch", -1):
            self._last_logged_conf_epoch = current_epoch
            log.info(
                f"w_conf schedule: epoch={current_epoch}, w_conf={target} "
                f"(start_epoch={start_epoch}, base={base_w_conf}, tgt={self.tgt_w_conf})"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, or predict begins."""
        for module in self.pre_processing:
            if hasattr(module, "set_stage"):
                module.set_stage(stage)

    def _apply_preprocessing(self, batch):
        for module in self.pre_processing:
            batch = module(batch)
        return batch

    def _apply_post_processing(self, pred_dict):
        for module in self.post_processing:
            pred_dict = module(pred_dict)
        return pred_dict

    @staticmethod
    def _get_selector_temperature(model: nn.Module) -> Optional[Tensor]:
        trajectory_selector = getattr(model, "trajectory_selector", None)
        temp = getattr(trajectory_selector, "learnable_temp_factor", None)
        if not torch.is_tensor(temp):
            return None
        return temp.clamp_min(1e-4)

    def _log_selector_temperature(
        self,
        prefix: str,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        temp = FutureMotion._get_selector_temperature(self.model)
        if temp is None:
            return

        self.log(
            f"{prefix}/selector_temperature",
            temp.detach(),
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            logger=True,
        )

    def _compute_bank_coverage(
        self,
        gt_pos: Tensor,
        valid_mask: Tensor,
        gt_valid: Optional[Tensor] = None,
    ) -> Optional[Dict[str, Tensor]]:
        """Compute nearest trajectory-bank coverage statistics for each valid agent.

        Args:
            gt_pos: [B, T, n_step_future, 2] ground-truth trajectories in agent frame.
            valid_mask: [B, T] boolean mask indicating agents to keep.

        Returns:
            dict with keys:
                "nearest_distance": [B, T] tensor (NaN for invalid agents)
                "nearest_index": [B, T] long tensor (-1 for invalid agents)
                "nearest_traj": [B, T, n_step_future, 2] tensor containing the
                    closest bank trajectory (zero for invalid agents)
                "stats": Dict[str, Tensor] summarizing coverage for valid agents
            or ``None`` if bank trajectories are not available.
        """

        if gt_pos is None or valid_mask is None:
            return None

        if self.traj_tensor is None or self.traj_bank_endpoints is None:
            return None

        with torch.no_grad():
            # explictly convert to the same device and dtype to avoid potential type mismatch
            traj_endpoints = self.traj_bank_endpoints.to(
                device=gt_pos.device, dtype=gt_pos.dtype
            )  # [K, 2]
            traj_bank = self.traj_tensor.to(device=gt_pos.device, dtype=gt_pos.dtype)
            B, T, _, _ = gt_pos.shape
            # Only consider agents with a valid endpoint in GT
            endpoint_valid = (
                gt_valid[..., -1]
                if gt_valid is not None and gt_valid.ndim == 3
                else valid_mask
            )  # [B, T]
            flat_valid = (valid_mask & endpoint_valid).view(-1)

            gt_endpoints = gt_pos[..., -1, :].view(-1, 2)  # [B*T, 2]
            nearest_distance = torch.full((B * T,), float("nan"), device=gt_pos.device)
            nearest_index = torch.full((B * T,), -1, device=gt_pos.device)
            nearest_traj = torch.zeros(
                (B * T, gt_pos.shape[-2], gt_pos.shape[-1]),
                device=gt_pos.device,
                dtype=gt_pos.dtype,
            )

            stats = None
            if flat_valid.any():
                gt_valid = gt_endpoints[flat_valid]  # [N_valid, 2]
                # [N_valid, K]
                dists = torch.cdist(gt_valid, traj_endpoints)
                min_dist, min_idx = dists.min(dim=1)
                nearest_distance[flat_valid] = min_dist
                nearest_index[flat_valid] = min_idx
                nearest_traj_src = traj_bank[min_idx.to(traj_bank.device)]
                nearest_traj[flat_valid] = nearest_traj_src.to(nearest_traj.dtype)

                # Additional robust statistics
                p90 = torch.quantile(min_dist, 0.90)
                p95 = torch.quantile(min_dist, 0.95)
                p99 = torch.quantile(min_dist, 0.99)
                stats = {
                    "min": min_dist.min(),
                    "max": min_dist.max(),
                    "mean": min_dist.mean(),
                    "median": min_dist.median(),
                    "p90": p90,
                    "p95": p95,
                    "p99": p99,
                }

        return {
            "nearest_distance": nearest_distance.view(B, T),
            "nearest_index": nearest_index.view(B, T),
            "nearest_traj": nearest_traj.view(B, T, gt_pos.shape[-2], gt_pos.shape[-1]),
            "stats": stats,
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # --------------------- Apply pre-processing ---------------------
        with torch.no_grad():
            batch = self._apply_preprocessing(batch)
            input_dict = {
                k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
            }
            gt_dict = {k.replace("/", "_"): v for k, v in batch.items() if "gt/" in k}
            pred_dict = {
                k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k
            }

        # --------------------- Forward pass through the model ---------------------
        # Schedule confidence weights per epoch
        self._schedule_conf_weight()
        model_outputs = self.model(**input_dict)
        self._log_selector_temperature(
            prefix="train",
            on_step=True,
            on_epoch=False,
        )

        pred_pos_logits = model_outputs.get("pred_pos_logits")
        if pred_pos_logits is not None and not torch.all(
            torch.isfinite(pred_pos_logits)
        ):
            log.error("!!! NaN or Inf detected in model's pred_pos_logits output !!!")

        if not torch.all(torch.isfinite(model_outputs["pred"])):
            log.error("!!! NaN or Inf detected in model's pred output !!!")

        pred_dict["pred_valid"] = model_outputs["valid_mask"]
        pred_dict["pred_conf"] = model_outputs["conf"]
        pred_dict["pred"] = model_outputs["pred"]
        pred_dict["pred_pos_logits"] = pred_pos_logits
        pred_dict["offset_pred"] = model_outputs.get("offset_pred")
        pred_dict["y_pred_others"] = model_outputs.get("y_pred_others")
        pred_dict["gqa_entropy_loss"] = model_outputs.get("gqa_entropy_loss")
        anchor_container = model_outputs.get("anchor_container")
        if anchor_container is not None:
            # Expose selected anchors for visualization modules
            if anchor_container.get("selected_anchors") is not None:
                pred_dict["selected_anchors"] = anchor_container["selected_anchors"]
            if anchor_container.get("selected_embeddings") is not None:
                pred_dict["selected_embeddings"] = anchor_container[
                    "selected_embeddings"
                ]
        # --------------------- Apply post-processing ---------------------
        # during training, waymo_post_processing is skipped
        pred_dict = self.post_processing(pred_dict)

        # --------------------- Compute Loss ---------------------
        metrics_dict_input = {**pred_dict, **gt_dict}

        # Pass anchor_scores from model output (used by scoring loss)
        anchor_scores = model_outputs.get("anchor_scores")
        if anchor_scores is not None:
            metrics_dict_input["anchor_scores"] = anchor_scores.to(self.device)

        if anchor_container:
            selected_anchors = anchor_container.get("selected_anchors")
            selected_embeddings = anchor_container.get("selected_embeddings")
            selected_anchor_indices = anchor_container.get("selected_anchor_indices")
            adapted_queries_128 = anchor_container.get("adapted_queries_128")

            if selected_anchors is not None:
                metrics_dict_input["selected_anchors"] = selected_anchors.to(
                    self.device
                )
            if selected_embeddings is not None:
                metrics_dict_input["selected_embeddings"] = selected_embeddings.to(
                    self.device
                )
            if adapted_queries_128 is not None:
                metrics_dict_input["adapted_queries_128"] = adapted_queries_128.to(
                    self.device
                )

            if selected_anchors is not None:
                coverage_info = self._compute_bank_coverage(
                    gt_pos=gt_dict.get("gt_pos"),
                    valid_mask=pred_dict["pred_valid"],
                    gt_valid=gt_dict.get("gt_valid"),
                )
                if coverage_info is not None:
                    anchor_container["nearest_bank_distance"] = coverage_info[
                        "nearest_distance"
                    ]
                    anchor_container["nearest_bank_index"] = coverage_info[
                        "nearest_index"
                    ]
                    anchor_container["nearest_bank_traj"] = coverage_info[
                        "nearest_traj"
                    ]

                    stats = coverage_info.get("stats")
                    if stats is not None:
                        self.log(
                            "train/coverage_min",
                            stats["min"],
                            on_step=True,
                            prog_bar=False,
                        )
                        self.log(
                            "train/coverage_max",
                            stats["max"],
                            on_step=True,
                            prog_bar=False,
                        )
                        self.log(
                            "train/coverage_mean",
                            stats["mean"],
                            on_step=True,
                            prog_bar=False,
                        )
                        self.log(
                            "train/coverage_median",
                            stats["median"],
                            on_step=True,
                            prog_bar=False,
                        )

                        # Log tail metrics to catch outliers driving the mean/max
                        self.log(
                            "train/coverage_p90",
                            stats["p90"],
                            on_step=True,
                            prog_bar=False,
                        )
                        self.log(
                            "train/coverage_p95",
                            stats["p95"],
                            on_step=True,
                            prog_bar=False,
                        )
                        self.log(
                            "train/coverage_p99",
                            stats["p99"],
                            on_step=True,
                            prog_bar=False,
                        )

        # --------------------- Track Mode Collapse
        allow_mode_entropy_tracking = True  # Keep it local dont pollute cfgs.

        if allow_mode_entropy_tracking:
            with torch.no_grad():
                pred_conf_probs = F.softmax(model_outputs["conf"], dim=-1)
                valid_mask = model_outputs["valid_mask"]
                n_pred = pred_conf_probs.shape[-1]

                if valid_mask.any():
                    # Entropy
                    log_probs = pred_conf_probs.log().clamp(min=-100)
                    per_agent_entropy = -(pred_conf_probs * log_probs).sum(dim=-1)
                    mean_entropy = per_agent_entropy[0].masked_select(valid_mask).mean()
                    max_entropy = torch.log(
                        torch.tensor(float(n_pred), device=mean_entropy.device)
                    )

                    self.log("train/conf_entropy", mean_entropy, on_step=True)
                    self.log(
                        "train/conf_entropy_norm",
                        mean_entropy / max_entropy,
                        on_step=True,
                        prog_bar=True,
                    )

                    # Max probability (overconfidence indicator)
                    max_prob = (
                        pred_conf_probs.max(dim=-1)[0][0]
                        .masked_select(valid_mask)
                        .mean()
                    )
                    self.log("train/conf_max_prob", max_prob, on_step=True)

                    # Track current w_conf for reference
                    if hasattr(self.train_metric, "w_conf"):
                        self.log(
                            "train/current_w_conf",
                            sum(self.train_metric.w_conf),
                            on_step=True,
                        )

        # --------------------- Log FAISS fusion statistics ---------------------
        self._log_faiss_stats(prefix="train", on_step=True, on_epoch=False)

        # --------------------- Log offset prediction statistics ---------------------
        self._log_offset_stats(
            offset_pred=pred_dict.get("offset_pred"),
            prefix="train",
            on_step=True,
            on_epoch=False,
        )

        metrics_dict = self.train_metric(**metrics_dict_input)

        for k in metrics_dict.keys():
            if (
                ("error_" in k)
                or ("loss" in k)
                or ("counter_traj" in k)
                or ("counter_conf" in k)
            ):
                self.log(k, metrics_dict[k], on_step=True)

        # Get total loss from the metrics
        total_loss = metrics_dict[f"{self.train_metric.prefix}/loss"]

        if self.global_rank == 0 and self.logger is not None:
            n_d = self.train_metric.n_decoders
            n_p = self.train_metric.n_pred
            for k in ["conf", "counter"]:
                for i in range(n_d):
                    w = []
                    for j in range(n_p):
                        k_str = f"{self.train_metric.prefix}/{k}_d{i}_p{j}"
                        if k_str in metrics_dict:
                            w.append(metrics_dict[k_str].item())
                    if w:  # Only create histogram if we have data
                        h = np.histogram(
                            range(n_p),
                            weights=w,
                            density=True,
                            bins=n_p,
                            range=(0, n_p - 1),
                        )
                        self.logger.experiment.log(
                            {
                                f"{self.train_metric.prefix}/{k}_d{i}": wandb.Histogram(
                                    np_histogram=h
                                )
                            }
                        )
            # --------------------- Log anchor distribution metrics during training ---------------------
            try:
                # Log scalar anchor metrics for training tracking
                anchor_scalar_keys = [
                    f"{self.train_metric.prefix}/stats/avg_winner_l1_distance",
                    f"{self.train_metric.prefix}/stats/total_winner_count",
                ]

                anchor_scalars = {}
                for key in anchor_scalar_keys:
                    if key in metrics_dict:
                        # Convert to scalar if tensor
                        value = metrics_dict[key]
                        if isinstance(value, torch.Tensor):
                            value = (
                                value.item()
                                if value.numel() == 1
                                else value.cpu().numpy()
                            )
                        anchor_scalars[key] = value

                if anchor_scalars:
                    # Remove explicit step parameter - let Lightning handle it
                    self.logger.experiment.log(anchor_scalars)
                    log.debug(
                        f"Logged {len(anchor_scalars)} anchor scalar metrics during training"
                    )

            except Exception as e:
                log.warning(f"Error logging anchor scalar metrics during training: {e}")

        return total_loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # --------------------- Pre-processing ---------------------
        batch = self._apply_preprocessing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        gt_dict = {k.replace("/", "_"): v for k, v in batch.items() if "gt/" in k}
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}

        # Schedule confidence weights per epoch (mirror train)
        self._schedule_conf_weight()
        model_outputs = self.model(
            inference_repeat_n=self.hparams.get("inference_repeat_n", 1),
            inference_cache_map=self.hparams.get("inference_cache_map", False),
            **input_dict,
        )
        pred_dict["pred_valid"] = model_outputs["valid_mask"]
        pred_dict["pred_conf"] = model_outputs["conf"]
        pred_dict["pred"] = model_outputs["pred"]
        pred_dict["pred_pos_logits"] = model_outputs.get("pred_pos_logits")
        pred_dict["offset_pred"] = model_outputs.get("offset_pred")
        pred_dict["y_pred_others"] = model_outputs.get("y_pred_others")
        anchor_container = model_outputs.get("anchor_container")

        # --------------------- Apply post-processing --------------------
        if self.hparams.model.motion_decoder.use_classification_head:
            # During training_step we do this within the nll_classification.py
            pred_dict = self.post_process_logits(pred_dict)
        if self.hparams.model.motion_decoder.use_classification_tower_head:
            pred_dict = self.post_process_logits(pred_dict)

        pred_dict = self.post_processing(pred_dict)

        # --------------------- Pack additional inputs for validation metrics ---------------------
        val_metrics_dict_input = {**pred_dict, **gt_dict}

        # Pass anchor_scores from model output (used by scoring loss)
        anchor_scores = model_outputs.get("anchor_scores")
        if anchor_scores is not None:
            val_metrics_dict_input["anchor_scores"] = anchor_scores.to(self.device)

        # Simplified trajectory preparation for visualization and metrics
        if anchor_container:
            selected_anchors = anchor_container.get("selected_anchors")
            selected_embeddings = anchor_container.get("selected_embeddings")
            adapted_queries_128 = anchor_container.get("adapted_queries_128")

            if selected_anchors is not None:
                val_metrics_dict_input["selected_anchors"] = selected_anchors.to(
                    self.device
                )
            if selected_embeddings is not None:
                val_metrics_dict_input["selected_embeddings"] = selected_embeddings.to(
                    self.device
                )
            if adapted_queries_128 is not None:
                val_metrics_dict_input["adapted_queries_128"] = adapted_queries_128.to(
                    self.device
                )

            if selected_anchors is not None:
                coverage_info = self._compute_bank_coverage(
                    gt_pos=gt_dict.get("gt_pos"),
                    valid_mask=pred_dict["pred_valid"],
                    gt_valid=gt_dict.get("gt_valid"),
                )
                if coverage_info is not None:
                    anchor_container["nearest_bank_distance"] = coverage_info[
                        "nearest_distance"
                    ]
                    anchor_container["nearest_bank_index"] = coverage_info[
                        "nearest_index"
                    ]
                    anchor_container["nearest_bank_traj"] = coverage_info[
                        "nearest_traj"
                    ]

                    stats = coverage_info.get("stats")
                    if stats is not None:
                        self.log(
                            "val/coverage_min",
                            stats["min"],
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            "val/coverage_max",
                            stats["max"],
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            "val/coverage_mean",
                            stats["mean"],
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            "val/coverage_median",
                            stats["median"],
                            on_step=False,
                            on_epoch=True,
                        )

                        self.log(
                            "val/coverage_p90",
                            stats["p90"],
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            "val/coverage_p95",
                            stats["p95"],
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            "val/coverage_p99",
                            stats["p99"],
                            on_step=False,
                            on_epoch=True,
                        )

        # --------------------- Log FAISS fusion statistics ---------------------
        self._log_faiss_stats(prefix="val", on_step=False, on_epoch=True)

        # --------------------- Log offset prediction statistics ---------------------
        self._log_offset_stats(
            offset_pred=pred_dict.get("offset_pred"),
            prefix="val",
            on_step=False,
            on_epoch=True,
        )

        # --------------------- Calculate validation metrics ---------------------
        val_metrics_dict = self.train_metric(**val_metrics_dict_input)

        # Log validation metrics
        for k in val_metrics_dict.keys():
            if (
                ("error_" in k)
                or ("loss" in k)
                or ("counter_traj" in k)
                or ("counter_conf" in k)
            ):
                self.log(f"val_{k}", val_metrics_dict[k], on_step=False, on_epoch=True)

        # Log anchor-quality scalar for validation (selection quality signal)
        val_anchor_key = f"{self.train_metric.prefix}/stats/avg_winner_l1_distance"
        if val_anchor_key in val_metrics_dict:
            self.log(
                f"val_{val_anchor_key}",
                val_metrics_dict[val_anchor_key],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # --------------------- Probability map visualization ---------------------
        if (
            self.hparams.get("plot_probmap", False)
            and batch_idx < 3
            and not self.trainer.sanity_checking
        ):
            # Prob map gets plotted and ontop we plot the motion forecasts
            try:
                self.pred_dict_visualizer.plot_probmap_visualization(
                    tensor_dict_to_cpu(batch),
                    tensor_dict_to_cpu(pred_dict),
                    self.current_epoch,
                    logger=self.logger,
                )
                log.info(
                    f"Successfully generated probability map visualizations for epoch {self.current_epoch}"
                )
            except Exception as e:
                raise ValueError(
                    f"Error plotting probability maps for epoch {self.current_epoch}: {str(e)}"
                )

        # --------------------- Plot motion forecasts ---------------------
        if (
            self.global_rank == 0
            and self.hparams.get("plot_motion", False)
            and batch_idx < 15
            and not self.trainer.sanity_checking
        ):
            # Prepare only the required batch index to reduce memory footprint
            idx_to_plot = 0
            batch_slice = {
                k: (
                    v[idx_to_plot : idx_to_plot + 1]
                    if isinstance(v, torch.Tensor)
                    and v.ndim > 0
                    and v.size(0) > idx_to_plot
                    else v
                )
                for k, v in batch.items()
            }
            # Only pass prediction keys used by the plot function
            pred_needed = {}
            for k in ("waymo_trajs", "waymo_scores"):
                if k in pred_dict and isinstance(pred_dict[k], torch.Tensor):
                    pred_needed[k] = pred_dict[k][idx_to_plot : idx_to_plot + 1]
                elif k in pred_dict:
                    pred_needed[k] = pred_dict[k]

            wandb_imgs = []
            fig = plot_motion_2d(
                tensor_dict_to_cpu(batch_slice),
                pred_dict=tensor_dict_to_cpu(pred_needed),
            )
            np_img = mplfig_to_npimage(fig)

            caption = f"motion_forecasts"
            if self.current_epoch is not None:
                caption += f" epoch_{self.current_epoch}_batch_{batch_idx}"
            wandb_imgs.append(wandb.Image(np_img, caption=caption))

            self.logger.experiment.log(
                {
                    f"epoch_{self.current_epoch}/motion_forecasts_batch_{batch_idx}": wandb_imgs
                },
                commit=False,
            )

        # --------------------- Plot motion Focal-track multi-modality ---------------------
        if (
            self.global_rank == 0
            and self.hparams.get("plot_motion_focal_track", False)
            and batch_idx < 15
            and not self.trainer.sanity_checking
        ):
            # Plot the gt of all the vehicles but visualize all the predictions for the focal track.
            idx_to_plot = 0
            batch_slice = {
                k: (
                    v[idx_to_plot : idx_to_plot + 1]
                    if isinstance(v, torch.Tensor)
                    and v.ndim > 0
                    and v.size(0) > idx_to_plot
                    else v
                )
                for k, v in batch.items()
            }
            pred_needed = {}
            for k in ("waymo_trajs", "waymo_scores"):
                if k in pred_dict and isinstance(pred_dict[k], torch.Tensor):
                    pred_needed[k] = pred_dict[k][idx_to_plot : idx_to_plot + 1]
                elif k in pred_dict:
                    pred_needed[k] = pred_dict[k]

            wandb_imgs = []
            fig = plot_motion_focal_track_multi_modality(
                tensor_dict_to_cpu(batch_slice),
                pred_dict=tensor_dict_to_cpu(pred_needed),
            )
            np_img = mplfig_to_npimage(fig)
            wandb_imgs.append(
                wandb.Image(np_img, caption=f"focal-track multi-modality")
            )

            self.logger.experiment.log(
                {
                    f"epoch_{self.current_epoch}/motion_focal_track_multi_modality_batch_{batch_idx}": wandb_imgs
                },
                commit=False,
            )
        # --------------------- Plot anchor selection ---------------------
        if (
            self.global_rank == 0
            and self.hparams.get("plot_anchor_selection", False)
            and batch_idx < 15
            and not self.trainer.sanity_checking
        ):
            if "selected_anchors" in val_metrics_dict_input:
                idx_to_plot = 0
                plot_keys = {
                    "ref_pos",
                    "ref_rot",
                    "ref_role",
                    "ref_type",
                    "gt_pos",
                    "gt_valid",
                    "selected_anchors",
                    "offset_pred",
                }
                plot_dict = {}

                for k in plot_keys:
                    if k not in val_metrics_dict_input:
                        continue
                    v = val_metrics_dict_input[k]
                    if not isinstance(v, torch.Tensor) or v.ndim == 0:
                        plot_dict[k] = v
                        continue
                    if k == "selected_anchors" and v.ndim == 6:
                        plot_dict[k] = v[:, idx_to_plot : idx_to_plot + 1]
                    elif k == "offset_pred" and v.ndim == 5:
                        plot_dict[k] = v[:, idx_to_plot : idx_to_plot + 1]
                    else:
                        plot_dict[k] = v[idx_to_plot : idx_to_plot + 1]

                sa = val_metrics_dict_input.get("selected_anchors")
                if isinstance(sa, torch.Tensor) and sa.ndim == 5:
                    gt_pos = val_metrics_dict_input.get("gt_pos")
                    if isinstance(gt_pos, torch.Tensor) and gt_pos.ndim >= 2:
                        n_agent = int(gt_pos.shape[1])
                        start = idx_to_plot * n_agent
                        end = start + n_agent
                        plot_dict["selected_anchors"] = (
                            sa[start:end] if sa.size(0) >= end else sa
                        )

                for k in ("ac/map_pos", "ac/map_valid", "ac/map_type"):
                    if (
                        k in batch
                        and isinstance(batch[k], torch.Tensor)
                        and batch[k].ndim > 0
                    ):
                        plot_dict[k.replace("/", "_")] = batch[k][
                            idx_to_plot : idx_to_plot + 1
                        ]

                scenario_id = None
                if "scenario_id" in batch:
                    sid = batch["scenario_id"]
                    try:
                        scenario_id = str(sid[idx_to_plot])
                    except Exception:
                        scenario_id = str(sid)

                anchor_plots = plot_anchor_selection(
                    tensor_dict_to_cpu(plot_dict),
                    plot_individual_anchors=False,
                    current_epoch=self.current_epoch,
                    batch_idx=batch_idx,
                    scenario_id=scenario_id,
                )
                if anchor_plots:
                    wandb_imgs = []
                    for fig, caption in anchor_plots:
                        np_img = mplfig_to_npimage(fig)
                        wandb_imgs.append(
                            wandb.Image(np_img, caption=f"{caption}_batch_{batch_idx}")
                        )
                        plt.close(fig)
                    log_key = f"anchor_selection_batch_{batch_idx}"
                    if self.current_epoch is not None:
                        log_key = f"epoch_{self.current_epoch}/" + log_key
                    self.logger.experiment.log({log_key: wandb_imgs}, commit=False)
            else:
                log.warning(
                    "Skipping anchor selection plot: 'selected_anchors' not in val_metrics_dict_input."
                )

        # --------------------- Plot endpoint predictions ---------------------
        if (
            self.global_rank == 0
            and self.hparams.get("plot_endpoints", False)
            and batch_idx < 15
            and not self.trainer.sanity_checking
        ):
            endpoint_plots = plot_endpoint_predictions(
                tensor_dict_to_cpu(batch),
                tensor_dict_to_cpu(pred_dict),
                current_epoch=self.current_epoch,
            )
            if endpoint_plots:
                wandb_imgs = []
                for fig, caption in endpoint_plots:
                    np_img = mplfig_to_npimage(fig)
                    wandb_imgs.append(
                        wandb.Image(np_img, caption=f"{caption}_batch_{batch_idx}")
                    )
                    plt.close(fig)
                log_key = f"endpoint_predictions_batch_{batch_idx}"
                if self.current_epoch is not None:
                    log_key = f"epoch_{self.current_epoch}/" + log_key
                self.logger.experiment.log({log_key: wandb_imgs}, commit=False)

        # --------------------- Update waymo metrics ---------------------
        waymo_ops_inputs = self.waymo_metric(
            batch, pred_dict["waymo_trajs"], pred_dict["waymo_scores"]
        )
        self.waymo_metric.aggregate_on_cpu(waymo_ops_inputs)
        self.waymo_metric.reset()

        self._save_to_submission_files(pred_dict, batch)

    def on_validation_epoch_end(self):
        # --------------------- Realize the accumulated metrics ---------------------
        # during the validation step, the metrics are accumulated and we realize them here
        epoch_waymo_metrics = self.waymo_metric.compute_waymo_motion_metrics()
        epoch_waymo_metrics["epoch"] = self.current_epoch

        for k, v in epoch_waymo_metrics.items():
            # sync_dist =/= True because we are using CPU aggregate method in val_step
            self.log(k, v, on_epoch=True)

        # negative since we want to checkpoint based on the best mean average precision
        self.log(
            "val/neg_mean_average_precision",
            -epoch_waymo_metrics[f"{self.waymo_metric.prefix}/mean_average_precision"],
            sync_dist=True,
        )

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        ## TODO: Not properly implemented, use the following things with caution
        # ! map can be empty for some scenes, check batch["map/valid"]
        batch = self._apply_preprocessing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}
        model_outputs = self.model(**input_dict)
        pred_dict["pred_valid"] = model_outputs["valid_mask"]
        pred_dict["pred_conf"] = model_outputs["conf"]
        pred_dict["pred"] = model_outputs["pred"]
        pred_dict = self.post_processing(pred_dict)
        self._save_to_submission_files(pred_dict, batch)

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.sub_womd.save_sub_files(self.logger)
            self.sub_av2.save_sub_files(self.logger)

    def configure_optimizers(self):
        weight_decay = self.hparams.optimizer.get("weight_decay", 0.0)
        if weight_decay is None:
            weight_decay = 0.0

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_param_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                    else:
                        # Fallback for custom modules with direct "weight" params.
                        if param.ndim <= 1:
                            no_decay.add(full_param_name)
                        else:
                            decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)

        param_dict = {
            param_name: param
            for param_name, param in self.named_parameters()
            if param.requires_grad
        }
        inter_params = decay & no_decay
        if inter_params:
            raise RuntimeError(f"Parameters in both decay/no_decay sets: {sorted(inter_params)}")
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            raise RuntimeError(
                f"Parameters not assigned to decay/no_decay sets: {sorted(missing_params)}"
            )

        optim_groups = []
        if decay:
            optim_groups.append(
                {
                    "params": [param_dict[param_name] for param_name in sorted(decay)],
                    "weight_decay": weight_decay,
                }
            )
        if no_decay:
            optim_groups.append(
                {
                    "params": [param_dict[param_name] for param_name in sorted(no_decay)],
                    "weight_decay": 0.0,
                }
            )

        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=optim_groups, _convert_="all"
        )

        # Calculate total steps and warmup steps
        if self.trainer is None:
            raise RuntimeError(
                "Trainer is not initialized. This method should be called after trainer is set."
            )

        total_steps = self.trainer.estimated_stepping_batches
        log.info(f"------------------Total steps------------------: {total_steps}")

        scheduler = {
            "scheduler": hydra.utils.instantiate(
                self.hparams.lr_scheduler,
                optimizer=optimizer,
                total_steps=int(total_steps),
            ),
            "monitor": "val_train/loss",
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

    def _save_to_submission_files(self, pred_dict: Dict, batch: Dict) -> None:
        submission_kargs_dict = {
            "waymo_trajs": pred_dict["waymo_trajs"],  # after nms
            "waymo_scores": pred_dict["waymo_scores"],  # after nms
            "mask_pred": batch["history/agent/role"][..., 2],
            "object_id": batch["history/agent/object_id"],
            "scenario_center": batch["scenario_center"],
            "scenario_yaw": batch["scenario_yaw"],
            "scenario_id": batch["scenario_id"],
        }
        self.sub_av2.add_to_submissions(**submission_kargs_dict)
        self.sub_womd.add_to_submissions(**submission_kargs_dict)

    def forward(
        self,
        batch: Dict[str, Tensor],
        adversarial_perturbation: Optional[Dict] = None,
        return_decoder_attention: bool = False,
    ) -> Dict:
        """
        Forward pass with optional adversarial perturbation.
        
        Args:
            batch: Input batch dictionary
            adversarial_perturbation: Optional dict with keys:
                - "type": perturbation type ("random", "shuffle", "worst_k", "uniform")
                - "level": perturbation level [0.0, 1.0]
                - "rng": optional torch.Generator for reproducibility
            return_decoder_attention: If True, collect cross-attention weights
                from the R2P decoder. Only supported by the HPTR-backend model
                (ac_model_R2P.py); ignored silently by other model variants.
        """
        batch = self._apply_preprocessing(batch)
        input_dict = {
            k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k
        }
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}

        # Extract all outputs from the model
        model_kwargs = {
            "adversarial_perturbation": adversarial_perturbation,
            **input_dict,
        }
        if return_decoder_attention:
            model_kwargs["return_decoder_attention"] = True
        model_outputs = self.model(**model_kwargs)

        pred_dict["pred_valid"] = model_outputs["valid_mask"]
        pred_dict["pred_conf"] = model_outputs["conf"]
        pred_dict["pred"] = model_outputs["pred"]
        pred_dict["offset_pred"] = model_outputs.get("offset_pred")

        # Extract and store ancillary outputs if available
        anchor_container = model_outputs.get("anchor_container")

        if anchor_container:
            pred_dict["selected_anchor_indices"] = anchor_container.get(
                "selected_anchor_indices"
            )
            if anchor_container.get("adapted_queries") is not None:
                pred_dict["adapted_queries"] = anchor_container["adapted_queries"]
            if anchor_container.get("adapted_queries_128") is not None:
                pred_dict["adapted_queries_128"] = anchor_container[
                    "adapted_queries_128"
                ]
            if anchor_container.get("selected_anchors") is not None:
                pred_dict["selected_anchors"] = anchor_container["selected_anchors"]

        # Apply post-processing
        pred_dict = self.post_processing(pred_dict)

        return pred_dict
