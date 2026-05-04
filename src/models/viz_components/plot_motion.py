import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import logging
import torch
from src.HPTR.src.utils.transform_utils import torch_pos2global

log = logging.getLogger(__name__)


def rotate_point_2d(point, angle, center=(0, 0)):
    """Rotate a 2D point around a center by given angle in radians."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]
    
    # Rotate
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    
    # Translate back
    return [x_new + center[0], y_new + center[1]]


def get_agent_dimensions(agent_type):
    """Get rectangular dimensions for different agent types based on plot_3d.py reference."""
    if agent_type[0]:  # Car
        # From plot_3d.py: car spans from -2.25 to 2.25 (length=4.5m) and -1 to 1 (width=2m)
        return {"length": 4.5, "width": 2.0}
    elif agent_type[1]:  # Pedestrian
        # From plot_3d.py: pedestrian spans from -0.3 to 0.3 in both directions (0.6m x 0.6m)
        return {"length": 0.6, "width": 0.6}
    elif agent_type[2]:  # Cyclist
        # From plot_3d.py: cyclist spans from -1 to 1 (length=2m) and -0.3 to 0.3 (width=0.6m)
        return {"length": 2.0, "width": 0.6}
    else:
        # Default dimensions for unknown agent types
        return {"length": 1.0, "width": 1.0}


def create_agent_box_2d(agent_pos, agent_yaw, agent_type):
    """Create a 2D rectangular box for an agent with type-specific dimensions."""
    dimensions = get_agent_dimensions(agent_type)
    length = dimensions["length"]
    width = dimensions["width"]
    
    # Define box corners relative to agent position (centered)
    # Length is along the x-axis (forward direction), width is along y-axis
    half_length = length / 2.0
    half_width = width / 2.0
    
    corners = [
        [-half_length, -half_width],  # back-left
        [half_length, -half_width],   # front-left  
        [half_length, half_width],    # front-right
        [-half_length, half_width]    # back-right
    ]
    
    # Rotate corners according to agent yaw
    rotated_corners = []
    for corner in corners:
        rotated_corner = rotate_point_2d(corner, -agent_yaw)  # negative for correct rotation direction
        rotated_corners.append([
            rotated_corner[0] + agent_pos[0],
            rotated_corner[1] + agent_pos[1]
        ])
    
    return np.array(rotated_corners)


def add_agent_box_2d(ax, agent_pos, agent_yaw, agent_type, color="blue", alpha=0.7):
    """Add a 2D rectangular box to represent an agent with type-specific dimensions."""
    box_corners = create_agent_box_2d(agent_pos, agent_yaw, agent_type)
    
    # Create a polygon patch for the rotated rectangle
    polygon = patches.Polygon(box_corners.tolist(), closed=True, facecolor=color, 
                            edgecolor='black', alpha=alpha, linewidth=1)
    ax.add_patch(polygon)


def plot_motion_2d(batch, pred_dict, n_step_future=60, idx_t_now=50, idx_batch=0, 
                   idx_focal=None, save_path='', plot_trajectories=True, plot_gt_trajectories=True):
    """
    Plot 2D motion forecasts with agents represented as realistic rectangular boxes.
    
    Args:
        batch: Input batch data
        pred_dict: Prediction dictionary  
        n_step_future: Number of future steps to plot (default: 60 for AV2)
        idx_t_now: Current time step index (default: 50 for AV2)
        idx_batch: Batch index to plot (default: 1)
        idx_focal: Index of focal agent to highlight (default: None)
        save_path: Path to save the plot (default: '')
        plot_trajectories: Whether to plot predicted trajectories (default: True)
        plot_gt_trajectories: Whether to plot ground truth trajectories (default: True)
    """
    
    fig, ax = plt.subplots(figsize=(15, 15), dpi=120)
    ax.set_facecolor("lightgrey")

    # Build a consistent per-agent color map for this scene
    try:
        agent_types_scene = batch.get("agent/type")[idx_batch] if "agent/type" in batch else None
        n_agents_scene = int(agent_types_scene.shape[0]) if agent_types_scene is not None else 0
        agent_cmap = cm.get_cmap("tab20", max(n_agents_scene, 1))
        agent_colors = [agent_cmap(i % (agent_cmap.N or 20)) for i in range(n_agents_scene)]
    except Exception:
        agent_colors = None
    
    # Plot all map polylines (roads/lanes)
    for map_polyline, map_valid, map_type in zip(
        batch['map/pos'][idx_batch], 
        batch['map/valid'][idx_batch], 
        batch["map/type"][idx_batch]
    ):
        map_polyline = map_polyline[map_valid]
        # Different colors for different map elements
        if map_type[4] or map_type[5] or map_type[6] or map_type[7] or map_type[8] or map_type[9] or map_type[10]:
            # Lane lines and road edges - white
            ax.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="white", linewidth=1, zorder=1)
        else:
            # Other map features - black
            ax.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="black", linewidth=1, zorder=1)

    # Plot ground truth trajectories if available and requested
    if plot_gt_trajectories and "gt/pos" in batch:
        gt_trajs = batch["gt/pos"][idx_batch]  # Shape: [n_agents, n_future_steps, 2] in LOCAL coordinates
        gt_valid = batch.get("gt/valid", None)

        # Need reference position and rotation for transformation to global coordinates
        if "ref/pos" in batch and "ref/rot" in batch:
            ref_pos = batch["ref/pos"][idx_batch]  # [n_target, 1, 2]
            ref_rot = batch["ref/rot"][idx_batch]  # [n_target, 2, 2]
        elif "ref_pos" in pred_dict and "ref_rot" in pred_dict:
            ref_pos = pred_dict["ref_pos"][idx_batch]  # [n_target, 1, 2] 
            ref_rot = pred_dict["ref_rot"][idx_batch]  # [n_target, 2, 2]
        else:
            log.warning("No reference position/rotation found for gt/pos transformation - plotting in local coordinates")
            ref_pos = None
            ref_rot = None
        
        if idx_focal is not None:
            # In agent-centric mode, gt_trajs, ref_pos, etc., only contain `n_target` agents.
            # The provided `idx_focal` is the original agent index. We must map it to the `target_agent` index.
            target_focal_idx = -1
            if 'ref/idx' in batch:
                target_indices = batch['ref/idx'][idx_batch]
                matches = (target_indices == idx_focal).nonzero(as_tuple=True)[0]
                if matches.numel() > 0:
                    target_focal_idx = matches[0].item()
                else:
                    log.warning(f"Focal agent {idx_focal} not in ref/idx. Cannot plot GT trajectory.")
            else:
                # Scene-centric, indices should be consistent
                target_focal_idx = idx_focal

            if target_focal_idx != -1:
                try:
                    # Plot ground truth trajectory for focal agent only
                    gt_trajectory = gt_trajs[target_focal_idx]
                    
                    # Apply validity mask if available
                    if gt_valid is not None:
                        valid_mask = gt_valid[idx_batch, target_focal_idx]
                        gt_trajectory = gt_trajectory[valid_mask]
                    
                    # Transform to global coordinates if reference data is available
                    if ref_pos is not None and ref_rot is not None:
                        gt_trajectory_global = torch_pos2global(
                            gt_trajectory.unsqueeze(0),  # [1, n_step_future, 2]
                            ref_pos[target_focal_idx].unsqueeze(0),  # [1, 1, 2]
                            ref_rot[target_focal_idx].unsqueeze(0)   # [1, 2, 2]
                        ).squeeze(0)  # [n_step_future, 2]
                        gt_trajectory = gt_trajectory_global
                    
                    # Plot GT trajectory with distinct styling (cyan color, dashed line)
                    if len(gt_trajectory) > 0:
                        ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], '--', color='tab:pink', 
                               linewidth=2, zorder=6, alpha=0.9, label='Ground Truth')
                        # # Add small circular markers at trajectory points for better visibility
                        # ax.scatter(gt_trajectory[:, 0], gt_trajectory[:, 1], c='cyan', s=25, 
                        #           marker='s', zorder=7, alpha=0.8, edgecolors='darkblue', linewidth=0.8)
                except IndexError:
                    log.warning(f"Target index {target_focal_idx} out of bounds for GT/Ref tensors.")
        else:
            # Plot ground truth trajectories for all agents (filtered for moving agents only)
            for idx_agent in range(gt_trajs.shape[0]):
                # Get agent info to filter for moving agents
                try:
                    agent_spd = batch["agent/spd"][idx_batch, idx_t_now, idx_agent] if idx_t_now < batch["agent/spd"].shape[1] else torch.tensor(0.0)
                    agent_type = batch["agent/type"][idx_batch, idx_agent]
                    
                    # Only plot GT for moving agents with valid types
                    if (agent_spd.abs() > 0.1 and 
                        (agent_type[0] or agent_type[1] or agent_type[2])):  # Any valid agent type
                        
                        gt_trajectory = gt_trajs[idx_agent]
                        
                        # Apply validity mask if available
                        if gt_valid is not None:
                            valid_mask = gt_valid[idx_batch, idx_agent]
                            gt_trajectory = gt_trajectory[valid_mask]
                        
                        # Transform to global coordinates if reference data is available
                        if ref_pos is not None and ref_rot is not None and idx_agent < ref_pos.shape[0]:
                            gt_trajectory_global = torch_pos2global(
                                gt_trajectory.unsqueeze(0),  # [1, n_step_future, 2]
                                ref_pos[idx_agent].unsqueeze(0),  # [1, 1, 2]
                                ref_rot[idx_agent].unsqueeze(0)   # [1, 2, 2]
                            ).squeeze(0)  # [n_step_future, 2]
                            gt_trajectory = gt_trajectory_global
                        
                        # Only plot if trajectory has meaningful motion (distance > threshold)
                        if len(gt_trajectory) > 1:
                            # Calculate total distance traveled to filter static trajectories
                            distances = torch.norm(gt_trajectory[1:] - gt_trajectory[:-1], dim=-1)
                            total_distance = distances.sum()
                            
                            if total_distance > 1.0:  # Only plot if agent moves > 1 meter total
                                # Use cyan/magenta spectrum for better contrast against viridis predictions
                                agent_color_idx = idx_agent % 6  # Cycle through 6 contrasting colors
                                colors_list = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'deeppink']
                                gt_color = colors_list[agent_color_idx]
                                
                                # Plot with dashed lines and small circular markers
                                ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], '--', 
                                       color=gt_color, linewidth=2, zorder=4, alpha=0.7, label='Ground Truth')
                                # ax.scatter(gt_trajectory[:, 0], gt_trajectory[:, 1], 
                                #          c=gt_color,marker='s', zorder=5, 
                                #          alpha=0.6, edgecolors='black', linewidth=0.5)
                except (IndexError, RuntimeError) as e:
                    # Skip agents that don't have the required data
                    continue

    # Plot predicted trajectories if available and requested
    if plot_trajectories and "waymo_trajs" in pred_dict:
        trajs = pred_dict["waymo_trajs"].movedim(1, -2)  # Reorder dimensions
        idx_top_mode = pred_dict["waymo_scores"].argmax(dim=-1, keepdim=True)
        
        if idx_focal is not None:
            # Plot trajectory for focal agent only
            agent_traj = trajs[idx_batch, idx_focal]
            idx_mode = int(idx_top_mode[idx_batch, idx_focal])
            trajectory = agent_traj[idx_mode, :, :]
            
            # Plot trajectory with color gradient
            cmap = cm.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, len(trajectory)))
            ax.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, s=50, zorder=5, alpha=0.8)
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color='pink', linewidth=2, 
                   zorder=4, alpha=0.6, label='orange')
        else:
            # Plot trajectories for all agents
            for idx_agent in range(trajs.shape[1]):
                agent_traj = trajs[idx_batch, idx_agent]
                idx_mode = int(idx_top_mode[idx_batch, idx_agent])
                trajectory = agent_traj[idx_mode, :, :]
                
                # Plot trajectory with color gradient  
                cmap = cm.get_cmap("viridis")
                colors = cmap(np.linspace(0, 1, len(trajectory)))
                ax.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, marker='s', s=30, zorder=3, alpha=0.2)

    # Overlay all predicted endpoints per agent if available (anchor-corrected if anchors provided)
    # if plot_trajectories:
    #     try:
    #         offset_pred = pred_dict.get("offset_pred", None)
    #         if offset_pred is not None:
    #             # Normalize to [B, T, Q, 2]
    #             if isinstance(offset_pred, torch.Tensor):
    #                 endpoints = offset_pred.detach().double()
    #             else:
    #                 endpoints = torch.as_tensor(offset_pred)
    #             if endpoints.ndim == 5:
    #                 endpoints = endpoints[0]
    #             elif endpoints.ndim == 3:
    #                 endpoints = endpoints.unsqueeze(2)

    #             if endpoints.ndim == 4 and idx_batch < endpoints.shape[0]:
    #                 B, T, Q, _ = endpoints.shape

    #                 # If selected anchors are available, add their endpoints to offsets
    #                 selected_anchors = pred_dict.get("selected_anchors", None)
    #                 anchors = None
    #                 if selected_anchors is not None:
    #                     anchors = selected_anchors
    #                     if not isinstance(anchors, torch.Tensor):
    #                         anchors = torch.as_tensor(anchors)
    #                     anchors = anchors.detach().double()
    #                     try:
    #                         if anchors.ndim == 6:
    #                             anchors = anchors[0]  # [B, T, Q, steps, 2]
    #                         elif anchors.ndim == 5 and anchors.shape[2] == 1:
    #                             # [B*T, Q, 1, steps, 2] -> [B, T, Q, steps, 2]
    #                             gt_pos = batch.get("gt/pos", pred_dict.get("gt_pos"))
    #                             if gt_pos is not None:
    #                                 gt_pos = torch.as_tensor(gt_pos)
    #                                 n_scene, n_agent = gt_pos.shape[0], gt_pos.shape[1]
    #                                 if anchors.shape[0] == n_scene * n_agent:
    #                                     anchors = anchors.squeeze(2).view(n_scene, n_agent, anchors.shape[1], anchors.shape[-2], anchors.shape[-1])
    #                         if anchors.ndim == 5 and anchors.shape[0] == B and anchors.shape[1] == T:
    #                             anchor_endpoints = anchors[..., -1, :]  # [B, T, Q, 2]
    #                             q = min(anchor_endpoints.shape[2], Q)
    #                             anchor_endpoints = anchor_endpoints[:, :, :q, :]
    #                             endpoints = endpoints[:, :, :q, :]
    #                             Q = q
    #                             endpoints = anchor_endpoints + endpoints
    #                         else:
    #                             anchors = None
    #                     except Exception:
    #                         anchors = None

    #                 # Transform from agent-local to SDC/world for plotting
    #                 ref_pos = batch.get("ref/pos", pred_dict.get("ref_pos"))
    #                 ref_rot = batch.get("ref/rot", pred_dict.get("ref_rot"))
    #                 if ref_pos is not None and ref_rot is not None:
    #                     ref_pos_t = torch.as_tensor(ref_pos).double().view(B, T, -1, 2)
    #                     ref_rot_t = torch.as_tensor(ref_rot).double().view(B, T, 2, 2)
    #                     agent_pos_flat = ref_pos_t[:, :, 0, :].contiguous().view(B * T, 1, 2)
    #                     agent_rot_flat = ref_rot_t.contiguous().view(B * T, 2, 2)
    #                     endpoints_flat = endpoints.view(B * T, Q, 2)
    #                     endpoints_sdc = torch_pos2global(endpoints_flat, agent_pos_flat, agent_rot_flat)
    #                     endpoints_sdc = endpoints_sdc.view(B, T, Q, 2)
    #                 else:
    #                     endpoints_sdc = endpoints

    #                 # Plot for all agents in this batch index
    #                 viridis_cmap = cm.get_cmap("viridis")
    #                 for idx_agent in range(min(T, endpoints_sdc.shape[1])):
    #                     pts = endpoints_sdc[idx_batch, idx_agent]  # [Q, 2]
    #                     if pts.numel() == 0:
    #                         continue
    #                     # Color endpoints with their agent-specific color if available
    #                     if agent_colors is not None and idx_agent < len(agent_colors):
    #                         cval = agent_colors[idx_agent]
    #                     else:
    #                         cval = viridis_cmap(0.8)
    #                     ax.scatter(pts[:, 0].cpu().numpy(), pts[:, 1].cpu().numpy(),
    #                                c=[cval], marker='o', s=25, zorder=6, linewidths=0.3, edgecolors='black', alpha=0.85)
    #     except Exception as e:
    #         log.debug(f"Failed to overlay all endpoints: {e}")

    # Plot agents as 2D rectangular boxes
    for idx, (agent_pos, agent_type, agent_yaw, agent_role, agent_spd) in enumerate(zip(
        batch["agent/pos"][idx_batch, idx_t_now], 
        batch["agent/type"][idx_batch], 
        batch["agent/yaw_bbox"][idx_batch, idx_t_now], 
        batch["agent/role"][idx_batch], 
        batch["agent/spd"][idx_batch, idx_t_now]
    )):
        # Determine agent color: focal gets a fixed highlight; otherwise use per-agent palette if available
        if idx_focal is not None and idx == idx_focal:
            color = "tab:orange"
            alpha = 0.85
        elif agent_colors is not None and idx < len(agent_colors):
            color = agent_colors[idx]
            alpha = 0.7 if agent_spd.abs() > 0.1 else 0.5
        else:
            # Fallback coloring based on motion state
            color = "tab:blue" if agent_spd.abs() > 0.1 else "tab:grey"
            alpha = 0.7 if agent_spd.abs() > 0.1 else 0.5
        # Plot rectangular boxes with type-specific dimensions
        if agent_type[0] or agent_type[1] or agent_type[2]:  # Any valid agent type
            add_agent_box_2d(ax, agent_pos, float(agent_yaw), agent_type, color=color, alpha=alpha)

    # Set equal aspect ratio and clean up the plot
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("2D Motion Forecasts with Rectangular Agent Boxes")
    # Add legend if both GT and predictions are plotted for focal agent
    if idx_focal is not None and plot_gt_trajectories and plot_trajectories:
        ax.legend(loc='upper right')
    
    # Save the plot if path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving 2D motion forecasts to {save_path}")
        plt.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)

    return fig

def plot_motion_focal_track_multi_modality(
    batch, pred_dict, n_step_future=60, idx_t_now=50, idx_batch=0, save_path=""
):
    """
    Plot motion forecasts with focal-track multi-modality, highlighting all predictions for a single focal agent.

    The focal agent is automatically determined by looking for the agent with the 'predict' role.

    batch: Dict[str, Tensor]
    pred_dict: Dict[str, Tensor]
    n_step_future: int
    idx_t_now: int
    idx_batch: int 
            This is currently set to 0 since for every epoch we just index into the idx=0 in later case to generate
            nice visualization you can set it to id_batch= [:batch_size]
    save_path: Optional[str]
            When passed it saves it to the path. Generally used for debugging.

    """
    fig, ax = plt.subplots(figsize=(15, 15), dpi=120)
    ax.set_facecolor("lightgrey")

    # Plot all map polylines (roads/lanes)
    for map_polyline, map_valid, map_type in zip(
        batch["map/pos"][idx_batch],
        batch["map/valid"][idx_batch],
        batch["map/type"][idx_batch],
    ):
        map_polyline = map_polyline[map_valid]
        if (
            map_type[4]
            or map_type[5]
            or map_type[6]
            or map_type[7]
            or map_type[8]
            or map_type[9]
            or map_type[10]
        ):
            ax.plot(
                map_polyline[:, 0], map_polyline[:, 1], "-", c="white", linewidth=1, zorder=1
            )
        else:
            ax.plot(
                map_polyline[:, 0], map_polyline[:, 1], "-", c="black", linewidth=1, zorder=1
            )

    # Find the focal agent (the one with 'predict' role)
    idx_focal = None
    for idx, agent_role in enumerate(batch["ref/role"][idx_batch]):
        if agent_role[2]:  # 'predict' role is at index 2
            idx_focal = idx
            break
    
    if idx_focal is None:
        log.warning("No focal agent found with 'predict' role. Using first agent as default.")
        idx_focal = 0

    # Plot all prediction modes for the focal agent using viridis color scheme
    if "waymo_trajs" in pred_dict:
        trajs = pred_dict["waymo_trajs"].movedim(1, -2)  # Reorder dimensions: [batch, agents, modes, steps, 2]
        scores = pred_dict["waymo_scores"]  # [batch, agents, modes]
        
        # Get all trajectories for the focal agent
        focal_agent_trajs = trajs[idx_batch, idx_focal]  # [n_modes, n_steps, 2]
        focal_agent_scores = scores[idx_batch, idx_focal]  # [n_modes]
        
        # Sort modes by score (highest to lowest) for better visualization ordering
        sorted_indices = torch.argsort(focal_agent_scores, descending=True)
        
        # Use viridis colormap for top 6 prediction modes only
        viridis_cmap = cm.get_cmap("viridis")
        n_modes_to_plot = min(6, focal_agent_trajs.shape[0])  # Plot top 6 modes maximum
        
        for mode_rank in range(n_modes_to_plot):
            mode_idx = sorted_indices[mode_rank]
            trajectory = focal_agent_trajs[mode_idx]  # [n_steps, 2]
            score = focal_agent_scores[mode_idx].item()
            
            # Color intensity: highest scored mode gets brightest color (close to 1.0)
            # Map mode_rank 0,1,2,3,4,5 to viridis values 1.0,0.8,0.6,0.4,0.2,0.0
            color_intensity = 1.0 - (mode_rank / max(1, n_modes_to_plot - 1))
            mode_color = viridis_cmap(color_intensity)
            
            # Line width and alpha based on rank
            if mode_rank == 0:  # Best mode
                alpha = 0.9
                zorder = 7
                label = f'Best Prediction (Score: {score:.3f})'
            elif mode_rank < 3:  # Top 3 modes
                alpha = 0.8
                zorder = 6
                label = f'Top-{mode_rank+1} (Score: {score:.3f})'
            else:  # Modes 4-6
                alpha = 0.7
                zorder = 5
                label = f'Mode {mode_rank+1} (Score: {score:.3f})'
            
            # Plot trajectory line only - no scatter points
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                   color=mode_color, linewidth=2, zorder=zorder, alpha=alpha, 
                   label=label)
    else:
        log.warning("No waymo_trajs found in pred_dict. Skipping trajectory plotting.")

    # Plot ground truth trajectories for all agents ON TOP (after predictions)
    if "gt/pos" in batch:
        gt_trajs = batch["gt/pos"][idx_batch]  # Shape: [n_agents, n_future_steps, 2] in LOCAL coordinates
        gt_valid = batch.get("gt/valid", None)

        # Need reference position and rotation for transformation to global coordinates
        if "ref/pos" in batch and "ref/rot" in batch:
            ref_pos = batch["ref/pos"][idx_batch]  # [n_target, 1, 2]
            ref_rot = batch["ref/rot"][idx_batch]  # [n_target, 2, 2]
        elif "ref_pos" in pred_dict and "ref_rot" in pred_dict:
            ref_pos = pred_dict["ref_pos"][idx_batch]  # [n_target, 1, 2] 
            ref_rot = pred_dict["ref_rot"][idx_batch]  # [n_target, 2, 2]
        else:
            log.warning("No reference position/rotation found for gt/pos transformation - plotting in local coordinates")
            ref_pos = None
            ref_rot = None

        # Plot ground truth trajectories for all agents (filtered for moving agents only)
        for idx_agent in range(gt_trajs.shape[0]):
            try:
                agent_spd = batch["agent/spd"][idx_batch, idx_t_now, idx_agent] if idx_t_now < batch["agent/spd"].shape[1] else torch.tensor(0.0)
                agent_type = batch["agent/type"][idx_batch, idx_agent]
                
                # Only plot GT for moving agents with valid types
                if (agent_spd.abs() > 0.1 and 
                    (agent_type[0] or agent_type[1] or agent_type[2])):  # Any valid agent type
                    
                    gt_trajectory = gt_trajs[idx_agent]
                    
                    # Apply validity mask if available
                    if gt_valid is not None:
                        valid_mask = gt_valid[idx_batch, idx_agent]
                        gt_trajectory = gt_trajectory[valid_mask]
                    
                    # Transform to global coordinates if reference data is available
                    if ref_pos is not None and ref_rot is not None and idx_agent < ref_pos.shape[0]:
                        gt_trajectory_global = torch_pos2global(
                            gt_trajectory.unsqueeze(0),  # [1, n_step_future, 2]
                            ref_pos[idx_agent].unsqueeze(0),  # [1, 1, 2]
                            ref_rot[idx_agent].unsqueeze(0)   # [1, 2, 2]
                        ).squeeze(0)  # [n_step_future, 2]
                        gt_trajectory = gt_trajectory_global
                    
                    # Only plot if trajectory has meaningful motion (distance > threshold)
                    if len(gt_trajectory) > 1:
                        # Calculate total distance traveled to filter static trajectories
                        distances = torch.norm(gt_trajectory[1:] - gt_trajectory[:-1], dim=-1)
                        total_distance = distances.sum()
                        
                        if total_distance > 1.0:  # Only plot if agent moves > 1 meter total
                            # Use high contrast colors against viridis (red/magenta spectrum)
                            if idx_agent == idx_focal:
                                # Focal agent GT in bright red with thick line and high zorder
                                ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], '--', 
                                       color='tab:pink', linewidth=2, alpha=0.8)
                            else:
                                # Other agents GT in magenta/pink spectrum (contrasts well with viridis)
                                agent_color_idx = idx_agent % 4
                                colors_list = ['magenta', 'deeppink', 'crimson', 'orangered']
                                gt_color = colors_list[agent_color_idx]
                                
                                ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], '--', 
                                       color=gt_color, linewidth=2, zorder=9, alpha=0.8)
            except (IndexError, RuntimeError):
                # Skip agents that don't have the required data
                continue

    # # --- Optional: overlay offset_pred points for the focal agent (plot all proposals) ---
    # try:
    #     offset_pred = pred_dict.get("offset_pred", None)
    #     selected_anchors = pred_dict.get("selected_anchors", None)

    #     def _to_tensor_local(x):
    #         if x is None:
    #             return None
    #         return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    #     # Normalize offset_pred shape to [B, T, Q, 2]
    #     if offset_pred is not None:
    #         offset_pred = _to_tensor_local(offset_pred).detach().double()
    #         if offset_pred.ndim == 5:
    #             # [n_decoder, B, T, Q, 2] -> first decoder
    #             offset_pred = offset_pred[0]
    #         elif offset_pred.ndim == 3:
    #             # [B, T, 2] -> [B, T, 1, 2]
    #             offset_pred = offset_pred.unsqueeze(2)

    #         if offset_pred.ndim == 4 and idx_batch < offset_pred.shape[0] and idx_focal is not None and idx_focal < offset_pred.shape[1]:
    #             B, T, Q, _ = offset_pred.shape

    #             # If anchors are available, compute corrected endpoints = anchor_endpoints + offset
    #             corrected_points = None
    #             if selected_anchors is not None:
    #                 anchors = _to_tensor_local(selected_anchors).detach().double()
    #                 # Try to reshape anchors to [B, T, Q, steps, 2]
    #                 try:
    #                     if anchors.ndim == 6:
    #                         # [n_decoder, B, T, Q, steps, 2] -> [B, T, Q, steps, 2] (first decoder)
    #                         anchors = anchors[0]
    #                     elif anchors.ndim == 5 and anchors.shape[2] == 1:
    #                         # [B*T, Q, 1, steps, 2] -> [B, T, Q, steps, 2]
    #                         gt_pos = pred_dict.get("gt_pos", batch.get("gt/pos"))
    #                         if gt_pos is not None:
    #                             gt_pos = _to_tensor_local(gt_pos)
    #                             n_scene, n_agent = gt_pos.shape[0], gt_pos.shape[1]
    #                             if anchors.shape[0] == n_scene * n_agent:
    #                                 anchors = anchors.squeeze(2).view(n_scene, n_agent, anchors.shape[1], anchors.shape[-2], anchors.shape[-1])
    #                     # If we successfully have anchors as [B, T, Q, steps, 2], compute endpoints
    #                     if anchors.ndim == 5 and anchors.shape[0] == B and anchors.shape[1] == T:
    #                         anchor_endpoints = anchors[..., -1, :]  # [B, T, Q, 2]
    #                         # Align Q if mismatched
    #                         q = min(anchor_endpoints.shape[2], Q)
    #                         anchor_endpoints = anchor_endpoints[:, :, :q, :]
    #                         offsets_cut = offset_pred[:, :, :q, :]
    #                         corrected_points = anchor_endpoints + offsets_cut  # [B, T, q, 2]
    #                 except Exception:
    #                     corrected_points = None

    #             # Choose points to plot (corrected if available, else raw offsets in agent frame)
    #             points_to_plot = corrected_points if corrected_points is not None else offset_pred

    #             # Transform focal agent endpoints to SDC/world if ref pose available
    #             ref_pos = batch.get("ref/pos", pred_dict.get("ref_pos"))
    #             ref_rot = batch.get("ref/rot", pred_dict.get("ref_rot"))
    #             if ref_pos is not None and ref_rot is not None:
    #                 agent_pos = torch.as_tensor(ref_pos).double()[idx_batch, 0, :]
    #                 agent_rot = torch.as_tensor(ref_rot).double()[idx_batch]
    #                 pts_agent = points_to_plot[idx_batch, idx_focal]  # [Q, 2]
    #                 pts_agent = pts_agent.unsqueeze(0)  # [1, Q, 2]
    #                 pts_world = torch_pos2global(pts_agent, agent_pos.view(1, 1, 2), agent_rot.view(1, 2, 2)).squeeze(0)
    #             else:
    #                 pts_world = points_to_plot[idx_batch, idx_focal]

    #             # Plot all endpoints for focal agent with a consistent color
    #             ax.scatter(pts_world[:, 0].cpu().numpy(), pts_world[:, 1].cpu().numpy(),
    #                        c=["tab:orange"], marker='x', s=60, zorder=10, label='endpoints (all)')
    # except Exception as e:
    #     log.debug(f"Failed to overlay offset_pred points for focal agent: {e}")

    # Plot agents as 2D rectangular boxes
    for idx, (agent_pos, agent_type, agent_yaw, agent_role, agent_spd) in enumerate(zip(
        batch["agent/pos"][idx_batch, idx_t_now], 
        batch["agent/type"][idx_batch], 
        batch["agent/yaw_bbox"][idx_batch, idx_t_now], 
        batch["agent/role"][idx_batch], 
        batch["agent/spd"][idx_batch, idx_t_now]
    )):
        # Determine agent color based on state and type
        if idx == idx_focal:
            color = "tab:orange"  # Focal agent (bright orange)
            alpha = 0.9
        elif agent_spd.abs() > 0.1:
            color = "tab:blue"    # Moving agent
            alpha = 0.7
        else:
            color = "tab:grey"    # Static agent
            alpha = 0.5
            
        # Plot rectangular boxes with type-specific dimensions
        if agent_type[0] or agent_type[1] or agent_type[2]:  # Any valid agent type
            add_agent_box_2d(ax, agent_pos, float(agent_yaw), agent_type, color=color, alpha=alpha)

    # Set equal aspect ratio and clean up the plot
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"Multi-Modal Motion Forecasts for Focal Agent (Agent {idx_focal})")
    
    # Add legend for better interpretation
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    # Save the plot if path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving multi-modal motion forecasts to {save_path}")
        log.debug(f"If not debugging; you should consider not passing the save_path")
        plt.savefig(save_path, dpi=320, bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)

    return fig
