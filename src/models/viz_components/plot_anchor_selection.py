import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from HPTR.src.utils.transform_utils import torch_pos2global

SAVE_PLOTS_LOCALLY = False
RETURN_FIGURES_FOR_WANDB_LOGGING = True
PLOT_OFFSET_PRED = True


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _blend_with_white(color: Tuple[float, float, float, float], blend: float) -> Tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * (1.0 - blend) + blend, g * (1.0 - blend) + blend, b * (1.0 - blend) + blend)


def _normalize_selected_anchors(selected_anchors: Tensor, n_scene: int, n_agent: int) -> Optional[Tensor]:
    if selected_anchors.ndim == 6:
        return selected_anchors

    if selected_anchors.ndim != 5 or selected_anchors.shape[2] != 1:
        return None

    anchors = selected_anchors.squeeze(2)
    if anchors.shape[0] == n_scene * n_agent:
        anchors = anchors.view(n_scene, n_agent, *anchors.shape[1:])
    elif anchors.shape[0] == n_agent and n_scene == 1:
        anchors = anchors.unsqueeze(0)
    else:
        return None

    return anchors.unsqueeze(0)


def _normalize_offset_pred(offset_pred: Optional[Tensor], n_scene: int, n_agent: int, n_anchor: int) -> Optional[Tensor]:
    if offset_pred is None:
        return None

    if offset_pred.ndim == 5:
        offset_pred = offset_pred[0]

    if offset_pred.ndim == 4 and offset_pred.shape[:3] == (n_scene, n_agent, n_anchor) and offset_pred.shape[-1] == 2:
        return offset_pred

    if offset_pred.ndim == 3 and n_scene == 1 and offset_pred.shape[:2] == (n_agent, n_anchor) and offset_pred.shape[-1] == 2:
        return offset_pred.unsqueeze(0)

    return None


def _plot_ac_map_in_sdc(
    ax: plt.Axes,
    map_pos_local: Tensor,
    map_valid: Tensor,
    map_type: Optional[Tensor],
    agent_ref_pos: Tensor,
    agent_ref_rot: Tensor,
) -> None:
    n_map = int(map_pos_local.shape[0])
    pos = map_pos_local.unsqueeze(0)  # [1, n_map, n_nodes, 2]
    local_pos = agent_ref_pos.unsqueeze(1).expand(1, n_map, -1, -1)  # [1, n_map, 1, 2]
    local_rot = agent_ref_rot.unsqueeze(1).expand(1, n_map, -1, -1)  # [1, n_map, 2, 2]
    pos_sdc = torch_pos2global(pos, local_pos, local_rot).squeeze(0).detach().cpu()

    map_valid_cpu = map_valid.detach().cpu()
    map_type_cpu = map_type.detach().cpu() if map_type is not None else None

    labels = set(ax.get_legend_handles_labels()[1])
    for i_map in range(n_map):
        valid_mask = map_valid_cpu[i_map]
        if not bool(valid_mask.any()):
            continue
        pts = pos_sdc[i_map][valid_mask]
        if pts.shape[0] < 2:
            continue

        color = "#f2f2f2"
        lw = 1.0
        alpha = 0.9
        label = None

        if map_type_cpu is not None:
            oh = map_type_cpu[i_map]
            is_center = bool(oh[0] or oh[1] or oh[2])
            is_boundary = bool(oh[3:10].any())
            is_crosswalk = bool(oh[10])

            if is_center:
                label = "centerline" if "centerline" not in labels else None
            elif is_boundary:
                color = "#e0e0e0"
                lw = 1.0
                label = "lane_boundary" if "lane_boundary" not in labels else None
            elif is_crosswalk:
                color = "#efe6ff"
                lw = 1.0
                label = "crosswalk" if "crosswalk" not in labels else None
            else:
                continue

        if label is not None:
            labels.add(label)
        ax.plot(
            pts[:, 0].numpy(),
            pts[:, 1].numpy(),
            "-",
            color=color,
            linewidth=lw,
            alpha=alpha,
            label=label,
            zorder=1,
        )


def plot_anchor_selection(
    metrics_input_dict: Dict[str, Tensor],
    plot_individual_anchors: bool = True,
    current_epoch: Optional[int] = None,
    batch_idx: Optional[int] = None,
    scenario_id: Optional[str] = None,
) -> List[Tuple[plt.Figure, str]]:
    ref_pos = metrics_input_dict.get("ref_pos")
    ref_rot = metrics_input_dict.get("ref_rot")
    ref_role = metrics_input_dict.get("ref_role")
    ref_type = metrics_input_dict.get("ref_type")
    gt_pos = metrics_input_dict.get("gt_pos")
    gt_valid = metrics_input_dict.get("gt_valid")
    selected_anchors = metrics_input_dict.get("selected_anchors")
    offset_pred = metrics_input_dict.get("offset_pred")

    ac_map_pos = metrics_input_dict.get("ac_map_pos")
    ac_map_valid = metrics_input_dict.get("ac_map_valid")
    ac_map_type = metrics_input_dict.get("ac_map_type")

    if (
        ref_pos is None
        or ref_rot is None
        or ref_role is None
        or ref_type is None
        or selected_anchors is None
    ):
        return []

    n_scene, n_agent = int(ref_pos.shape[0]), int(ref_pos.shape[1])
    selected_anchors = _normalize_selected_anchors(selected_anchors, n_scene=n_scene, n_agent=n_agent)
    if selected_anchors is None:
        return []

    offset_pred = _normalize_offset_pred(offset_pred, n_scene=n_scene, n_agent=n_agent, n_anchor=int(selected_anchors.shape[3]))

    tag_parts = []
    if current_epoch is not None:
        tag_parts.append(f"e{int(current_epoch)}")
    if batch_idx is not None:
        tag_parts.append(f"b{int(batch_idx)}")
    if scenario_id:
        tag_parts.append(f"s{_safe_slug(scenario_id)}")
    tag = "_".join(tag_parts) if tag_parts else "run"

    save_dir = os.path.join("outputs/anchor_selection_viz", tag)
    if SAVE_PLOTS_LOCALLY:
        os.makedirs(save_dir, exist_ok=True)

    type_names = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
    cmap = plt.get_cmap("tab20")

    plots: List[Tuple[plt.Figure, str]] = []

    def agent_valid(j_scene: int, k_agent: int) -> bool:
        return bool(ref_role[j_scene, k_agent].any())

    def agent_type_name(j_scene: int, k_agent: int) -> str:
        type_index = int(torch.argmax(ref_type[j_scene, k_agent].float()).item())
        return type_names.get(type_index, "Unknown")

    def local_to_sdc(traj_local: Tensor, agent_ref_pos: Tensor, agent_ref_rot: Tensor) -> Tensor:
        return torch_pos2global(traj_local.unsqueeze(0), agent_ref_pos, agent_ref_rot).squeeze(0)

    def plot_one_agent(
        ax: plt.Axes,
        j_scene: int,
        k_agent: int,
        gt_label: Optional[str],
        anchors_label: Optional[str],
    ) -> Tensor:
        agent_ref_pos = ref_pos[j_scene, k_agent].unsqueeze(0)  # [1, 1, 2]
        agent_ref_rot = ref_rot[j_scene, k_agent].unsqueeze(0)  # [1, 2, 2]

        if ac_map_pos is not None and ac_map_valid is not None:
            map_pos_local = ac_map_pos[j_scene, k_agent]
            map_valid = ac_map_valid[j_scene, k_agent]
            map_type = ac_map_type[j_scene, k_agent] if ac_map_type is not None else None
            _plot_ac_map_in_sdc(ax, map_pos_local, map_valid, map_type, agent_ref_pos, agent_ref_rot)

        agent_color = cmap(k_agent % cmap.N)
        anchor_color = _blend_with_white(agent_color, blend=0.55)

        if gt_pos is not None:
            gt_traj_local = gt_pos[j_scene, k_agent]
            if isinstance(gt_valid, Tensor):
                gt_traj_local = gt_traj_local[gt_valid[j_scene, k_agent]]
            if gt_traj_local.numel() > 0:
                gt_traj_sdc = local_to_sdc(gt_traj_local, agent_ref_pos, agent_ref_rot)
                ax.plot(
                    gt_traj_sdc[:, 0].cpu().numpy(),
                    gt_traj_sdc[:, 1].cpu().numpy(),
                    "-",
                    linewidth=2,
                    color=agent_color,
                    label=gt_label,
                    zorder=3,
                )
                ax.plot(
                    gt_traj_sdc[-1, 0].cpu().numpy(),
                    gt_traj_sdc[-1, 1].cpu().numpy(),
                    ">",
                    markersize=8,
                    color=agent_color,
                    zorder=4,
                )

        anchors_local = selected_anchors[0, j_scene, k_agent]  # [Q, steps, 2]
        for q_idx in range(int(anchors_local.shape[0])):
            anchor_sdc = local_to_sdc(anchors_local[q_idx], agent_ref_pos, agent_ref_rot)
            ax.plot(
                anchor_sdc[:, 0].cpu().numpy(),
                anchor_sdc[:, 1].cpu().numpy(),
                "--",
                linewidth=1,
                alpha=0.9,
                color=anchor_color,
                label=anchors_label if q_idx == 0 else None,
                zorder=2,
            )
            ax.plot(
                anchor_sdc[-1, 0].cpu().numpy(),
                anchor_sdc[-1, 1].cpu().numpy(),
                "o",
                markersize=4,
                alpha=0.9,
                color=anchor_color,
                zorder=4,
            )

            if PLOT_OFFSET_PRED and offset_pred is not None:
                endpoint_local = anchors_local[q_idx, -1]
                corrected_local = endpoint_local + offset_pred[j_scene, k_agent, q_idx]
                endpoints_sdc = local_to_sdc(torch.stack([endpoint_local, corrected_local], dim=0), agent_ref_pos, agent_ref_rot)
                endpoint_sdc, corrected_sdc = endpoints_sdc[0], endpoints_sdc[1]

                offset_label = "offset" if q_idx == 0 and "offset" not in set(ax.get_legend_handles_labels()[1]) else None
                ax.plot(
                    [endpoint_sdc[0].cpu().numpy(), corrected_sdc[0].cpu().numpy()],
                    [endpoint_sdc[1].cpu().numpy(), corrected_sdc[1].cpu().numpy()],
                    "-",
                    linewidth=1,
                    alpha=0.9,
                    color=anchor_color,
                    label=offset_label,
                    zorder=5,
                )
                ax.plot(
                    corrected_sdc[0].cpu().numpy(),
                    corrected_sdc[1].cpu().numpy(),
                    "x",
                    markersize=6,
                    alpha=0.9,
                    color=anchor_color,
                    zorder=6,
                )

        return agent_ref_pos.squeeze(0).squeeze(0)

    if plot_individual_anchors:
        for j_scene in range(n_scene):
            for k_agent in range(n_agent):
                if not agent_valid(j_scene, k_agent):
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))
                agent_xy = plot_one_agent(ax, j_scene, k_agent, gt_label="gt", anchors_label="anchors")
                ax.set_xlim(agent_xy[0].item() - 50.0, agent_xy[0].item() + 50.0)
                ax.set_ylim(agent_xy[1].item() - 50.0, agent_xy[1].item() + 50.0)
                ax.set_aspect("equal")
                ax.set_title(f"scene {j_scene} agent {k_agent} ({agent_type_name(j_scene, k_agent)})")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(loc="upper right")
                plt.tight_layout()
                fname = f"anchor_s{j_scene}_a{k_agent}.png"
                if SAVE_PLOTS_LOCALLY:
                    fig.savefig(os.path.join(save_dir, fname), dpi=150)
                caption = f"{tag} scene {j_scene} agent {k_agent}"
                if RETURN_FIGURES_FOR_WANDB_LOGGING:
                    plots.append((fig, caption))
                else:
                    plt.close(fig)
        return plots

    for j_scene in range(n_scene):
        fig, ax = plt.subplots(figsize=(12, 10))
        centers = []
        for k_agent in range(n_agent):
            if not agent_valid(j_scene, k_agent):
                continue
            centers.append(plot_one_agent(ax, j_scene, k_agent, gt_label=f"agent {k_agent}", anchors_label=None))

        if not centers:
            plt.close(fig)
            continue

        centers_t = torch.stack(centers, dim=0)
        center = centers_t.mean(dim=0)
        radius = torch.norm(centers_t - center.unsqueeze(0), dim=-1).max().item() + 50.0
        ax.set_xlim(center[0].item() - radius, center[0].item() + radius)
        ax.set_ylim(center[1].item() - radius, center[1].item() + radius)
        ax.set_aspect("equal")
        ax.set_title(f"scene {j_scene} anchors (SDC)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fname = f"anchor_scene_s{j_scene}.png"
        if SAVE_PLOTS_LOCALLY:
            fig.savefig(os.path.join(save_dir, fname), dpi=150)
        caption = f"{tag} scene {j_scene}"
        if RETURN_FIGURES_FOR_WANDB_LOGGING:
            plots.append((fig, caption))
        else:
            plt.close(fig)

    return plots
