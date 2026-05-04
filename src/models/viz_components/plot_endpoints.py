import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from HPTR.src.utils.transform_utils import torch_pos2global, torch_rad2rot
from src.models.viz_components.plot_motion import add_agent_box_2d

log = logging.getLogger(__name__)


def _reshape_selected_anchors(
    selected_anchors: Optional[Tensor],
    gt_pos: Optional[Tensor],
) -> Optional[Tensor]:
    """Normalize selected_anchors to shape [B, T, Q, steps, 2] if possible."""
    if selected_anchors is None or not isinstance(selected_anchors, Tensor):
        return None

    if selected_anchors.ndim == 6:
        # [n_decoder, B, T, Q, steps, 2] -> take first decoder
        return selected_anchors[0]

    if selected_anchors.ndim == 5 and selected_anchors.shape[2] == 1:
        if gt_pos is None:
            return None
        n_batch, n_q, _, n_steps, coord_dim = selected_anchors.shape
        n_scene, n_agent = gt_pos.shape[0], gt_pos.shape[1]
        if n_batch != n_scene * n_agent:
            log.debug(
                "Unable to reshape selected_anchors: batch mismatch (%s vs %s)",
                n_batch,
                n_scene * n_agent,
            )
            return None
        reshaped = selected_anchors.squeeze(2).view(n_scene, n_agent, n_q, n_steps, coord_dim)
        return reshaped

    log.debug("Unsupported selected_anchors shape for endpoint plotting: %s", selected_anchors.shape)
    return None


def plot_endpoint_predictions(
    batch: Dict[str, Tensor],
    pred_dict: Dict[str, Tensor],
    current_epoch: Optional[int] = None,
    max_scenes: int = 2,
    max_agents: Optional[int] = None,
) -> List[Tuple[Figure, str]]:
    """Create endpoint prediction plots with map, anchor, and agent context."""

    def _to_tensor(value: Optional[Tensor]) -> Optional[Tensor]:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return value
        return torch.as_tensor(value)

    offset_pred = pred_dict.get("offset_pred")
    if offset_pred is None:
        log.warning("Skipping endpoint plot: offset_pred not provided")
        return []

    offset_pred = _to_tensor(offset_pred).detach().double()
    if offset_pred.ndim == 5:
        offset_pred = offset_pred[0]
    if offset_pred.ndim == 3:
        offset_pred = offset_pred.unsqueeze(2)
    if offset_pred.ndim != 4:
        log.warning("Unexpected offset_pred shape for endpoint plotting: %s", offset_pred.shape)
        return []

    B, T, Q, _ = offset_pred.shape
    if B == 0 or T == 0 or Q == 0:
        log.warning("Empty offset_pred tensor; skipping endpoint plot")
        return []

    ref_pos = pred_dict.get("ref_pos")
    if ref_pos is None:
        ref_pos = batch.get("ref/pos")
    ref_rot = pred_dict.get("ref_rot")
    if ref_rot is None:
        ref_rot = batch.get("ref/rot")
    gt_pos = batch.get("gt/pos", pred_dict.get("gt_pos"))

    if ref_pos is None or ref_rot is None or gt_pos is None:
        log.warning("Skipping endpoint plot: missing reference pose or ground-truth trajectories")
        return []

    ref_pos_tensor = _to_tensor(ref_pos).double()
    ref_rot_tensor = _to_tensor(ref_rot).double()
    gt_pos_tensor = _to_tensor(gt_pos).double()

    if gt_pos_tensor.shape[0] != B or gt_pos_tensor.shape[1] != T:
        log.warning(
            "Skipping endpoint plot: gt_pos shape mismatch %s (expected [%s, %s, ...])",
            tuple(gt_pos_tensor.shape),
            B,
            T,
        )
        return []

    try:
        ref_pos_tensor = ref_pos_tensor.view(B, T, -1, 2)
        agent_pos_flat = ref_pos_tensor[:, :, 0, :].contiguous().view(B * T, 1, 2)
        agent_rot_tensor = ref_rot_tensor.view(B, T, 2, 2)
        agent_rot_flat = agent_rot_tensor.contiguous().view(B * T, 2, 2)
    except RuntimeError as exc:
        log.warning("Skipping endpoint plot: unable to reshape reference poses (%s)", exc)
        return []

    pred_endpoints_local = offset_pred

    selected_anchors = pred_dict.get("selected_anchors")
    anchor_trajs_local: Optional[Tensor] = None
    if selected_anchors is not None:
        anchor_tensor = _to_tensor(selected_anchors).detach()
        anchor_trajs_local = _reshape_selected_anchors(anchor_tensor, gt_pos_tensor)
        if anchor_trajs_local is not None and anchor_trajs_local.ndim == 5:
            if anchor_trajs_local.shape[0] != B or anchor_trajs_local.shape[1] != T:
                log.debug(
                    "Selected anchors shape mismatch (%s) with batch dims (%s, %s); ignoring anchor trajectories.",
                    tuple(anchor_trajs_local.shape),
                    B,
                    T,
                )
                anchor_trajs_local = None
            else:
                anchor_trajs_local = anchor_trajs_local.double()
                anchor_endpoints = anchor_trajs_local[..., -1, :]
                if anchor_endpoints.shape[2] != Q:
                    q = min(anchor_endpoints.shape[2], Q)
                    anchor_trajs_local = anchor_trajs_local[:, :, :q, :, :]
                    anchor_endpoints = anchor_endpoints[:, :, :q, :]
                    pred_endpoints_local = pred_endpoints_local[:, :, :q, :]
                    Q = q
                pred_endpoints_local = anchor_endpoints + pred_endpoints_local
        else:
            anchor_trajs_local = None

    steps_future = gt_pos_tensor.shape[-2]
    gt_local_flat = gt_pos_tensor.view(B * T, steps_future, 2)
    gt_sdc_flat = torch_pos2global(gt_local_flat, agent_pos_flat, agent_rot_flat)
    gt_sdc = gt_sdc_flat.view(B, T, steps_future, 2)

    pred_local_flat = pred_endpoints_local.view(B * T, Q, 2)
    pred_sdc_flat = torch_pos2global(pred_local_flat, agent_pos_flat, agent_rot_flat)
    pred_sdc = pred_sdc_flat.view(B, T, Q, 2)

    anchors_sdc: Optional[Tensor] = None
    if anchor_trajs_local is not None:
        anchor_steps = anchor_trajs_local.shape[-2]
        anchors_local_flat = anchor_trajs_local.view(B * T * Q, anchor_steps, 2)
        agent_pos_repeat = agent_pos_flat.repeat_interleave(Q, dim=0)
        agent_rot_repeat = agent_rot_flat.repeat_interleave(Q, dim=0)
        anchors_sdc_flat = torch_pos2global(anchors_local_flat, agent_pos_repeat, agent_rot_repeat)
        anchors_sdc = anchors_sdc_flat.view(B, T, Q, anchor_steps, 2)

    def _fetch_scenario_tensor(*keys: str) -> Optional[Tensor]:
        for key in keys:
            if key in pred_dict:
                return _to_tensor(pred_dict[key])
            if key in batch:
                return _to_tensor(batch[key])
        return None

    # Follow plot_motion_2d: stay in SDC frame (map is already in SDC)
    pred_world = pred_sdc
    gt_world = gt_sdc
    anchors_world = anchors_sdc

    pred_valid = pred_dict.get("pred_valid")
    if isinstance(pred_valid, Tensor):
        pred_valid = pred_valid.bool()
    else:
        pred_valid = None

    gt_valid = batch.get("gt/valid", pred_dict.get("gt_valid"))
    gt_valid = gt_valid.bool() if isinstance(gt_valid, Tensor) else None

    map_pos = batch.get("map/pos")
    map_valid = batch.get("map/valid")
    map_type = batch.get("map/type")
    if map_pos is not None:
        map_pos = _to_tensor(map_pos)
    if map_valid is not None:
        map_valid = _to_tensor(map_valid)
    if map_type is not None:
        map_type = _to_tensor(map_type)

    agent_pos_all = batch.get("agent/pos")
    agent_yaw_all = batch.get("agent/yaw_bbox")
    agent_type_all = batch.get("agent/type")
    agent_spd_all = batch.get("agent/spd")
    agent_pos_all = _to_tensor(agent_pos_all)
    agent_yaw_all = _to_tensor(agent_yaw_all)
    agent_type_all = _to_tensor(agent_type_all)
    agent_spd_all = _to_tensor(agent_spd_all)

    ref_idx = batch.get("ref/idx")
    ref_idx = _to_tensor(ref_idx)

    scenes_to_plot = min(B, max_scenes)
    agents_to_plot = T if max_agents is None else min(T, max_agents)

    colors = plt.cm.get_cmap("tab10", max(agents_to_plot, 1))
    anchor_cmap = cm.get_cmap("viridis", max(Q, 1))

    figures: List[Tuple[Figure, str]] = []

    for scene_idx in range(scenes_to_plot):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("lightgrey")

        title = f"Predicted endpoints – scene {scene_idx}"
        if current_epoch is not None:
            title += f" (epoch {current_epoch})"
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

        # Plot map context if available
        if map_pos is not None and map_valid is not None:
            try:
                map_pos_scene = map_pos[scene_idx]
                map_valid_scene = map_valid[scene_idx]
                map_type_scene = map_type[scene_idx] if map_type is not None else None
                for idx_poly, (polyline, valid_mask) in enumerate(zip(map_pos_scene, map_valid_scene)):
                    valid_poly = polyline[valid_mask]
                    if valid_poly.numel() == 0:
                        continue
                    coords = valid_poly.cpu().numpy()
                    if coords.shape[0] < 2:
                        continue
                    color = "black"
                    if map_type_scene is not None:
                        type_vec = map_type_scene[idx_poly]
                        if (
                            type_vec.shape[-1] >= 11
                            and (
                                type_vec[4]
                                or type_vec[5]
                                or type_vec[6]
                                or type_vec[7]
                                or type_vec[8]
                                or type_vec[9]
                                or type_vec[10]
                            )
                        ):
                            color = "white"
                    ax.plot(coords[:, 0], coords[:, 1], "-", c=color, linewidth=1.0, alpha=0.7, zorder=1)
            except Exception as exc:
                log.debug("Failed to plot map context for scene %s: %s", scene_idx, exc)

        legend_flags = {
            "gt_traj": False,
            "gt_endpoint": False,
            "anchors": False,
            "endpoints": False,
        }
        plotted_any = False

        ref_idx_scene = None
        if isinstance(ref_idx, Tensor):
            try:
                ref_idx_scene = ref_idx[scene_idx]
            except IndexError:
                ref_idx_scene = None

        for agent_idx in range(agents_to_plot):
            if pred_valid is not None:
                try:
                    if not pred_valid[scene_idx, agent_idx]:
                        continue
                except IndexError:
                    continue

            preds = pred_world[scene_idx, agent_idx].cpu().numpy()
            gt_traj = gt_world[scene_idx, agent_idx].cpu().numpy()

            if preds.ndim == 1:
                preds = preds[None, :]

            color = colors(agent_idx % max(colors.N, 1))

            if gt_valid is not None:
                try:
                    valid_mask = gt_valid[scene_idx, agent_idx].cpu().numpy().astype(bool)
                    if valid_mask.shape[0] == gt_traj.shape[0]:
                        gt_traj = gt_traj[valid_mask]
                except Exception:
                    pass

            if gt_traj.shape[0] > 0:
                gt_point = gt_traj[-1]
                if gt_traj.shape[0] > 1:
                    gt_label = "Ground-truth trajectory" if not legend_flags["gt_traj"] else None
                    ax.plot(
                        gt_traj[:, 0],
                        gt_traj[:, 1],
                        "--",
                        color=color,
                        linewidth=1.5,
                        alpha=0.85,
                        zorder=3,
                        label=gt_label,
                    )
                    legend_flags["gt_traj"] = True

                gt_endpoint_label = "Ground-truth endpoint" if not legend_flags["gt_endpoint"] else None
                ax.scatter(
                    gt_point[0],
                    gt_point[1],
                    color=color,
                    marker="x",
                    s=70,
                    linewidths=1.2,
                    zorder=4,
                    label=gt_endpoint_label,
                )
                legend_flags["gt_endpoint"] = True

            if anchors_world is not None:
                anchor_trajs = anchors_world[scene_idx, agent_idx]
                for anchor_idx, anchor_traj in enumerate(anchor_trajs):
                    anchor_np = anchor_traj.cpu().numpy()
                    if anchor_np.shape[0] < 2:
                        continue
                    anchor_color = anchor_cmap(anchor_idx / max(Q - 1, 1) if Q > 1 else 0.5)
                    anchor_label = "Anchor trajectories" if not legend_flags["anchors"] else None
                    ax.plot(
                        anchor_np[:, 0],
                        anchor_np[:, 1],
                        color=anchor_color,
                        linewidth=1.0,
                        alpha=0.6,
                        zorder=2,
                        label=anchor_label,
                    )
                    legend_flags["anchors"] = True

            for anchor_idx in range(preds.shape[0]):
                # Color endpoints to match their agent color for clarity
                endpoint_label = "Predicted endpoints" if not legend_flags["endpoints"] else None
                ax.scatter(
                    preds[anchor_idx, 0],
                    preds[anchor_idx, 1],
                    color=color,
                    marker="o",
                    s=45,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=5,
                    label=endpoint_label,
                )
                legend_flags["endpoints"] = True

            target_agent_idx = None
            if isinstance(ref_idx_scene, Tensor):
                if agent_idx < ref_idx_scene.shape[0]:
                    target_agent_idx = int(ref_idx_scene[agent_idx].item())
            elif ref_idx_scene is None:
                target_agent_idx = agent_idx

            if (
                target_agent_idx is not None
                and target_agent_idx >= 0
                and agent_pos_all is not None
                and agent_pos_all.shape[2] > target_agent_idx
            ):
                try:
                    latest_idx = agent_pos_all.shape[1] - 1
                    agent_pos_np = agent_pos_all[scene_idx, latest_idx, target_agent_idx].cpu().numpy()
                    agent_yaw_val = 0.0
                    if agent_yaw_all is not None and agent_yaw_all.shape[2] > target_agent_idx:
                        agent_yaw_val = float(agent_yaw_all[scene_idx, latest_idx, target_agent_idx])
                    agent_type_vec = (
                        agent_type_all[scene_idx, target_agent_idx].cpu().numpy()
                        if agent_type_all is not None and agent_type_all.shape[1] > target_agent_idx
                        else np.array([False, False, False])
                    )
                    box_alpha = 0.7
                    if agent_spd_all is not None and agent_spd_all.shape[2] > target_agent_idx:
                        spd_val = float(agent_spd_all[scene_idx, latest_idx, target_agent_idx])
                        box_alpha = 0.9 if abs(spd_val) > 0.1 else 0.6
                    add_agent_box_2d(
                        ax,
                        agent_pos_np,
                        agent_yaw_val,
                        agent_type_vec,
                        color=color,
                        alpha=box_alpha,
                    )
                    ax.text(
                        agent_pos_np[0],
                        agent_pos_np[1],
                        f"{target_agent_idx}",
                        fontsize=8,
                        ha="center",
                        va="center",
                        color="black",
                        zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.1"),
                    )
                except Exception as exc:
                    log.debug("Failed to plot agent box for scene %s agent %s: %s", scene_idx, agent_idx, exc)

            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        if any(legend_flags.values()):
            ax.legend(loc="best", fontsize="small")

        caption = f"scene_{scene_idx}"
        if current_epoch is not None:
            caption = f"epoch_{current_epoch}_" + caption
        figures.append((fig, caption))

    return figures
