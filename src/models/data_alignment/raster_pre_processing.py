import os
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, Optional
from src.HPTR.src.utils.transform_utils import torch_pos2local, torch_pos2global
import torch.nn.functional as F
import numpy as np
import logging


logger = logging.getLogger(__name__)

class RasterPreProcessing(nn.Module):
    def __init__(self, n_target: int, n_other: int, n_map: int, **kwargs) -> None:
        super().__init__()
        self.n_target = n_target
        self.n_other = n_other
        self.n_map = n_map
        self._model_kwargs = kwargs
        self.stage: Optional[str] = None

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    @property
    def model_kwargs(self):
        return self._model_kwargs

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.stage not in ("fit", "validate"):
            return batch

        sdf_map = batch.get("gt/sdf_map")
        if sdf_map is None:
            return batch

        if "gt/sdf_map_orig_dims" not in batch:
            if sdf_map.ndim == 2:
                dims = torch.tensor([[sdf_map.shape[0], sdf_map.shape[1]]], device=sdf_map.device, dtype=torch.long)
            else:
                dims = torch.tensor(
                    [[sdf_map.shape[-2], sdf_map.shape[-1]]] * int(sdf_map.shape[0]),
                    device=sdf_map.device,
                    dtype=torch.long,
                )
            batch["gt/sdf_map_orig_dims"] = dims

        return batch


class RasterPreProcessingWthRaster(nn.Module):
    def __init__(self, n_target: int, n_other: int, n_map: int, **kwargs) -> None:
        super().__init__()
        self.n_target = n_target
        self.n_other = n_other
        self.n_map = n_map
        self._model_kwargs = kwargs
        self.stage: Optional[str] = None
        # Visualization controls (optional)
        self.visualize_packed_map: bool = bool(kwargs.get("visualize_packed_map", False))
        self.viz_max_scenes: int = int(kwargs.get("viz_max_scenes", 1))
        self.viz_max_targets_per_scene: int = int(kwargs.get("viz_max_targets_per_scene", 1))
        self.viz_max_batches: int = int(kwargs.get("viz_max_batches", 1))
        self.viz_save_dir: str = str(kwargs.get("viz_save_dir", "outputs/preproc_viz"))
        self.viz_plot_other_agents: bool = bool(kwargs.get("viz_plot_other_agents", True))
        self.viz_max_other_agents: int = int(kwargs.get("viz_max_other_agents", -1))  # -1 means no limit
        # capture current-step index if provided by hydra (from model cfg)
        self.step_current: int = int(kwargs.get("time_step_current", 0))
        self._viz_batches_done: int = 0

    def set_stage(self, stage: str) -> None:
        """Set the current stage of the trainer."""
        self.stage = stage

    @property
    def model_kwargs(self):
        """Return model kwargs that were passed during initialization"""
        return self._model_kwargs

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: Dictionary containing raster data, sim2 transform, scenario center and scenario yaw
            
        returns:
            Adds the following to the batch:
            batch['gt/sdf_map']: Signed distance field of the raster data in true world coordinates
            batch['gt/sim2_R']: Sim2 transform rotation matrix
            batch['gt/sim2_t']: Sim2 transform translation vector
            batch['gt/sim2_s']: Sim2 transform scaling factor
            batch['gt/scenario_center']: Scenario center
            batch['gt/scenario_yaw']: Scenario yaw
            batch['gt/sdf_map_orig_dims']: Original dimensions of the SDF map
        """
        if self.stage not in ['fit', 'validate']:
            return batch

        # Check for required keys and provide detailed error messages
        required_keys = ['sim2_R', 'sim2_t', 'sim2_s', 'scenario_center', 'scenario_yaw']
        missing_keys = [key for key in required_keys if key not in batch]
        
        if missing_keys:
            logger.warning(f"RasterPreProcessing: Missing required keys: {missing_keys}")
            # Continue processing but log the issue
        
        # Check for probability map dimensions
        if 'gt/sdf_map_orig_dims' not in batch:
            sdf_map = batch.get('gt/sdf_map')
            if sdf_map is not None:
                batch['gt/sdf_map_orig_dims'] = torch.tensor(
                    [[sdf_map[i].shape[0], sdf_map[i].shape[1]] for i in range(sdf_map.shape[0])],
                    device=sdf_map.device, dtype=torch.long
                )
        
        # Ensure Sim2 transform is properly copied to gt namespace
        if all(key in batch for key in ['sim2_R', 'sim2_t', 'sim2_s']):
            batch['gt/sim2_R'] = batch['sim2_R']
            batch['gt/sim2_t'] = batch['sim2_t'] 
            batch['gt/sim2_s'] = batch['sim2_s']
        
        # Ensure scenario center and yaw are properly copied to gt namespace
        if 'scenario_center' in batch:
            batch['gt/scenario_center'] = batch['scenario_center']
        if 'scenario_yaw' in batch:
            batch['gt/scenario_yaw'] = batch['scenario_yaw']
            
        # Optional: visualize packed map elements around each target in agent-local frame
        try:
            if (
                self.visualize_packed_map 
                and self._viz_batches_done < self.viz_max_batches
                and 'ac/map_pos' in batch and 'ac/map_valid' in batch
                and 'ac/target_pos' in batch and 'ac/target_valid' in batch
            ):
                self._visualize_packed_map(batch)
                self._viz_batches_done += 1
        except Exception as e:
            logger.warning(f"RasterPreProcessing visualization failed: {e}")
            # Do not interrupt training/validation due to viz issues

        return batch

    @torch.no_grad()
    def _visualize_packed_map(self, batch: Dict[str, Tensor]) -> None:
        """Save simple 2D plots of packed map polylines per target (agent-local frame).

        Expects keys:
            - ac/map_pos: [n_scene, n_target, n_map, n_pl_node, 2]
            - ac/map_valid: [n_scene, n_target, n_map, n_pl_node]
            - ac/target_pos: [n_scene, n_target, n_step_hist, 2]
            - ac/target_valid: [n_scene, n_target, n_step_hist]
        """
        os.makedirs(self.viz_save_dir, exist_ok=True)

        map_pos = batch['ac/map_pos']
        map_valid = batch['ac/map_valid']
        tgt_pos = batch['ac/target_pos']
        tgt_valid = batch['ac/target_valid']
        other_pos = batch.get('ac/other_pos', None)
        other_valid = batch.get('ac/other_valid', None)
        other_type = batch.get('ac/other_type', None)

        n_scene = map_pos.shape[0]
        n_target = map_pos.shape[1]
        n_scene_to_plot = min(self.viz_max_scenes, n_scene)
        n_target_to_plot = min(self.viz_max_targets_per_scene, n_target)

        # Use CPU tensors for matplotlib
        map_pos_cpu = map_pos.detach().cpu()
        map_valid_cpu = map_valid.detach().cpu()
        tgt_pos_cpu = tgt_pos.detach().cpu()
        tgt_valid_cpu = tgt_valid.detach().cpu()

        for i_scene in range(n_scene_to_plot):
            for i_tgt in range(n_target_to_plot):
                fig, ax = plt.subplots(figsize=(8, 8))
                # Plot packed polylines with type-aware coloring (AV2 semantics)
                pos_st = map_pos_cpu[i_scene, i_tgt]        # [n_map, n_pl_node, 2]
                val_st = map_valid_cpu[i_scene, i_tgt]      # [n_map, n_pl_node]
                type_st = None
                if 'ac/map_type' in batch:
                    type_st = batch['ac/map_type'][i_scene, i_tgt].detach().cpu()  # [n_map, 11] bool
                n_map_elems = pos_st.shape[0]
                plotted_any = False

                # Color scheme: centerlines (0,1,2) -> light gray; lane boundaries (4..9) -> black
                for i_map in range(n_map_elems):
                    mask = val_st[i_map]
                    if not mask.any():
                        continue
                    pts = pos_st[i_map][mask]
                    if pts.shape[0] < 2:
                        continue

                    color = 'gray'
                    alpha = 0.5
                    lw = 1.0
                    label = None

                    if type_st is not None:
                        oh = type_st[i_map]
                        is_center = bool(oh[0] or oh[1] or oh[2])
                        is_boundary = bool(oh[4:10].any())
                        if is_center:
                            color = '#c8c8c8'  # light grey
                            alpha = 0.9
                            lw = 1.0
                            label = 'lane centerline' if 'lane centerline' not in ax.get_legend_handles_labels()[1] else None
                        elif is_boundary:
                            color = 'black'
                            alpha = 0.9
                            lw = 1.2
                            label = 'lane boundary' if 'lane boundary' not in ax.get_legend_handles_labels()[1] else None
                        else:
                            # Skip non-lane polylines (e.g., crosswalk edges) for this view
                            continue

                    ax.plot(pts[:, 0].numpy(), pts[:, 1].numpy(), '-', color=color, alpha=alpha, linewidth=lw, label=label)
                    plotted_any = True

                # Plot target history in local frame
                hist = tgt_pos_cpu[i_scene, i_tgt]
                hmask = tgt_valid_cpu[i_scene, i_tgt]
                if hmask.any():
                    hist = hist[hmask]
                    ax.plot(hist[:, 0].numpy(), hist[:, 1].numpy(), 'b.-', label='target_hist', linewidth=1.5)

                # Mark agent origin (current frame at (0,0))
                ax.scatter([0.0], [0.0], c='red', marker='*', s=60, label='agent_origin')

                # Optionally plot other agents' histories wrt this target (already in target-local frame)
                if self.viz_plot_other_agents and other_pos is not None and other_valid is not None:
                    other_pos_cpu = other_pos.detach().cpu()
                    other_valid_cpu = other_valid.detach().cpu()
                    other_type_cpu = other_type.detach().cpu() if other_type is not None else None

                    n_other = other_pos_cpu.shape[2]
                    max_other = n_other if self.viz_max_other_agents < 0 else min(self.viz_max_other_agents, n_other)

                    existing_labels = set(ax.get_legend_handles_labels()[1])
                    for j in range(max_other):
                        traj = other_pos_cpu[i_scene, i_tgt, j]
                        vmask = other_valid_cpu[i_scene, i_tgt, j]
                        if not vmask.any():
                            continue
                        traj_valid = traj[vmask]

                        # Color by agent type if available (vehicle/pedestrian/cyclist)
                        color = '#377eb8'  # blue for vehicle default
                        label = None
                        if other_type_cpu is not None:
                            oh = other_type_cpu[i_scene, i_tgt, j]
                            if bool(oh[0]):
                                color = '#377eb8'  # vehicle
                                if 'other_vehicle' not in existing_labels:
                                    label = 'other_vehicle'
                                    existing_labels.add(label)
                            elif bool(oh[1]):
                                color = '#4daf4a'  # pedestrian
                                if 'other_pedestrian' not in existing_labels:
                                    label = 'other_pedestrian'
                                    existing_labels.add(label)
                            elif bool(oh[2]):
                                color = '#984ea3'  # cyclist
                                if 'other_cyclist' not in existing_labels:
                                    label = 'other_cyclist'
                                    existing_labels.add(label)
                            else:
                                color = '#aaaaaa'
                        ax.plot(traj_valid[:, 0].numpy(), traj_valid[:, 1].numpy(), '-', color=color, linewidth=1.0, alpha=0.8, label=label)
                        # Mark current-step point if within bounds
                        if 0 <= self.step_current < len(vmask) and bool(vmask[self.step_current]):
                            cur_xy = traj[self.step_current]
                            ax.scatter([float(cur_xy[0])], [float(cur_xy[1])], color=color, s=12, marker='o')

                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('X (agent-local)')
                ax.set_ylabel('Y (agent-local)')
                ax.set_title(f"Packed Map | scene {i_scene} target {i_tgt} [{self.stage}]")
                if plotted_any:
                    ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

                fname = os.path.join(
                    self.viz_save_dir,
                    f"packed_map_{self.stage or 'unknown'}_b{self._viz_batches_done:03d}_s{i_scene:02d}_t{i_tgt:02d}.png",
                )
                try:
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    logger.info(f"Saved packed-map viz to: {fname}")
                finally:
                    plt.close(fig)
