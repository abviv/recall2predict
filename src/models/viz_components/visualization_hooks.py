import torch
import logging
import wandb
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from matplotlib.patches import Rectangle
from src.HPTR.src.utils.transform_utils import torch_rad2rot, torch_pos2global
from src.models.data_alignment.preprocess_probabilitymaps import SimpleTransform
from src.models.viz_components.plot_3d import mplfig_to_npimage, tensor_dict_to_cpu


# TODO: This file has a lot of duplicate code which needs to be refactored
log = logging.getLogger(__name__)

class TrajectoryVisualizer:
    """
    Visualize trajectories in the SDF raster. Everything (map, predictions, GT) is in True World Coordinates (TWC).
    For more definitions of TWC, refer to README.md
    """
    def __init__(self):
        pass

    @staticmethod
    def visualize_trajectories(viz_data,
                               gt_scenario_center, 
                               gt_sdf_map, gt_sdf_map_orig_dims,
                               gt_sim2_R, gt_sim2_t, gt_sim2_s,
                               n_scene, n_agent):
        
        for j_vis in range(n_scene):
            # Check if scene was processed (might have been skipped due to missing sim2)
            if (j_vis, 0) not in viz_data['pred_world'] and (j_vis, 1) not in viz_data['pred_world']: # Basic check, assumes agent 0 or 1 exists if scene processed
                    log.warning(f"Skipping visualization for scene {j_vis} as it might not have been processed for loss/coords.")
                    continue

            # Get SDC World Pose for this scene (Needed for SDC center plot)
            sdc_world_center_scene_vis = gt_scenario_center[j_vis].unsqueeze(0).unsqueeze(0)
            # Map details
            scene_sdf_map_vis = gt_sdf_map[j_vis]
            sim2_transform_vis = SimpleTransform(R=gt_sim2_R[j_vis], t=gt_sim2_t[j_vis], s=gt_sim2_s[j_vis]) # Assuming scene was not skipped

            # Prepare plot
            if gt_sdf_map_orig_dims is None:
                orig_height, orig_width = scene_sdf_map_vis.shape
            else:
                orig_dims = gt_sdf_map_orig_dims[j_vis].cpu().int().numpy()
                orig_height, orig_width = orig_dims[0], orig_dims[1]

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            scene_sdf_map_padded_vis = scene_sdf_map_vis.detach().cpu().numpy()
            scene_sdf_map_cropped_vis = scene_sdf_map_padded_vis[:orig_height, :orig_width]

            s_val_tensor = sim2_transform_vis.s
            s_val = s_val_tensor.item() if torch.is_tensor(s_val_tensor) else s_val_tensor
            t_val_tensor = sim2_transform_vis.t
            t_val = t_val_tensor.cpu().numpy() if torch.is_tensor(t_val_tensor) else t_val_tensor
            true_x_min = -t_val[0]
            true_y_min = -t_val[1]
            true_x_max = true_x_min + orig_width / s_val
            true_y_max = true_y_min + orig_height / s_val
            extent = [true_x_min, true_x_max, true_y_min, true_y_max]

            ax.imshow(scene_sdf_map_cropped_vis, cmap='coolwarm', origin='lower', extent=extent, alpha=0.7, aspect='auto')

            scenario_center_np_vis = sdc_world_center_scene_vis.squeeze().cpu().numpy()
            ax.scatter(scenario_center_np_vis[0], scenario_center_np_vis[1], c='black', s=120, marker='*', label='SDC Center (World)', zorder=10)
            ax.axhline(y=scenario_center_np_vis[1], color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=scenario_center_np_vis[0], color='black', linestyle='--', alpha=0.5)

            type_map = {0: ('Vehicle', 'blue', 'x'), 1: ('Pedestrian', 'green', 's'), 2: ('Cyclist', 'magenta', '^')}
            labeled_types = set()
            plotted_pred = False

            # Plot trajectories using stored viz_data
            for k_vis in range(n_agent):
                viz_key = (j_vis, k_vis)
                if viz_key not in viz_data['pred_world']: # Agent might not be valid for this mode
                    continue

                pred_traj_world = viz_data['pred_world'][viz_key]
                gt_traj_world = viz_data['gt_world'][viz_key]
                type_index = viz_data['type'][viz_key]
                type_label, type_color, type_marker = type_map.get(type_index, ('Unknown', 'gray', '.'))

                # Plot prediction
                ax.scatter(pred_traj_world[:, 0], pred_traj_world[:, 1],
                            c='red', s=15, marker='o', label='Pred (World)' if not plotted_pred else "")
                ax.plot(pred_traj_world[:, 0], pred_traj_world[:, 1], 'r-', alpha=0.6)
                plotted_pred = True

                # Plot GT
                current_label = f"GT {type_label} (World)"
                scatter_label = current_label if current_label not in labeled_types else ""
                ax.scatter(gt_traj_world[:, 0], gt_traj_world[:, 1],
                            c=type_color, s=15, marker=type_marker, label=scatter_label)
                ax.plot(gt_traj_world[:, 0], gt_traj_world[:, 1], color=type_color, linestyle='-', alpha=0.6)
                if scatter_label:
                    labeled_types.add(current_label)

            # Finalize plot
            ax.legend()
            ax.set_xlabel('X (True World)')
            ax.set_ylabel('Y (True World)')
            ax.set_title(f"Scene {j_vis} - True World SDF & Trajectories (Mode-Selected)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()
            plt.close(fig)

class MapVisualizer:
    """
    Visualize the map and the trajectories in the True World Coordinates (TWC).
    source hook: src.HPTR.src.data_modules.agent_centric.AgentCentricPreProcessing -> visualize_world_data
    """
    def __init__(self):
        pass

    @staticmethod
    def visualize_map(batch,
                      n_scene, 
                      n_target,
                      original_map_valid_sdc, 
                      original_map_pos_sdc,
                      device):

        for i_scene in range(n_scene):
            if not original_map_valid_sdc[i_scene].any():
                continue

            # Get SDC pose in True World for this scene
            sdc_world_center = batch["gt/scenario_center"][i_scene].unsqueeze(0).unsqueeze(0) # Shape [1, 1, 2]
            sdc_world_yaw = batch["gt/scenario_yaw"][i_scene] # Scalar tensor
            sdc_world_rot = torch_rad2rot(sdc_world_yaw).unsqueeze(0) # Shape [1, 2, 2]

            # Get SDC-centered map points for this scene
            map_pos_sdc_scene = original_map_pos_sdc[i_scene] #[n_pl, n_nodes, 2]
            map_valid_sdc_scene = original_map_valid_sdc[i_scene] #[n_pl, n_nodes]
            valid_map_pos_sdc = map_pos_sdc_scene[map_valid_sdc_scene]

            if valid_map_pos_sdc.shape[0] == 0:
                continue

            # Transform valid SDC-centered map points to True World
            map_pos_world = torch_pos2global(valid_map_pos_sdc.unsqueeze(0).to(device),
                                                sdc_world_center.to(device),
                                                sdc_world_rot.to(device)).squeeze(0)
            map_pos_world_np = map_pos_world.cpu().numpy()

            xmin, ymin = map_pos_world_np.min(axis=0)
            xmax, ymax = map_pos_world_np.max(axis=0)

            fig, ax = plt.subplots(figsize=(12, 12)) 

            # Plot the world boundary rectangle
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=1, edgecolor='g', facecolor='none', label='Map Boundary (World)')
            ax.add_patch(rect)

            # Plot True World map points
            plotted_map = False
            for i_pl in range(map_pos_sdc_scene.shape[0]):
                valid_mask_pl = map_valid_sdc_scene[i_pl]
                if valid_mask_pl.any():
                    points_sdc_pl = map_pos_sdc_scene[i_pl][valid_mask_pl]
                    points_world_pl = torch_pos2global(points_sdc_pl.unsqueeze(0).to(device),
                                                        sdc_world_center.to(device),
                                                        sdc_world_rot.to(device)).squeeze(0)
                    points_world_pl_np = points_world_pl.cpu().numpy()
                    ax.plot(points_world_pl_np[:, 0], points_world_pl_np[:, 1], '.-', color='gray', markersize=2, alpha=0.5,
                            label='Map Polylines (World)' if not plotted_map else "")
                    plotted_map = True

            # Plot SDC center in world coordinates
            sdc_center_np = sdc_world_center.squeeze().cpu().numpy()
            ax.scatter(sdc_center_np[0], sdc_center_np[1], c='black', marker='*', s=120, label='SDC Center (World)', zorder=10)

            # --- Plot Agent Trajectories ---
            plotted_hist = False
            plotted_fut = False
            for i_target in range(n_target):
                # Agent pose relative to SDC
                agent_ref_pos = batch["ref/pos"][i_scene, i_target] # [1, 2]
                agent_ref_rot = batch["ref/rot"][i_scene, i_target] # [2, 2]

                # History (Agent-Local -> SDC -> World)
                hist_local = batch["ac/target_pos"][i_scene, i_target] # [n_scene, n_target, n_step_hist, 2]
                hist_valid = batch["ac/target_valid"][i_scene, i_target] # [n_scene, n_target, n_step_hist]
                valid_hist_local = hist_local[hist_valid]
                if valid_hist_local.shape[0] > 0:
                    hist_sdc = torch_pos2global(valid_hist_local.unsqueeze(0), agent_ref_pos, agent_ref_rot).squeeze(0)
                    hist_world = torch_pos2global(hist_sdc.unsqueeze(0).to(device),
                                                    sdc_world_center.to(device),
                                                    sdc_world_rot.to(device)).squeeze(0)
                    hist_world_np = hist_world.cpu().numpy()
                    ax.plot(hist_world_np[:, 0], hist_world_np[:, 1], 'b.-', markersize=4, linewidth=1.5,
                            label='History (World)' if not plotted_hist else "")
                    plotted_hist = True

                # Future (Agent-Local -> SDC -> World) - Only plot if gt exists
                if "gt/pos" in batch:
                    fut_local = batch["gt/pos"][i_scene, i_target] # [n_step_future, 2]
                    fut_valid = batch["gt/valid"][i_scene, i_target] # [n_step_future]
                    valid_fut_local = fut_local[fut_valid]
                    if valid_fut_local.shape[0] > 0:
                        fut_sdc = torch_pos2global(valid_fut_local.unsqueeze(0), agent_ref_pos, agent_ref_rot).squeeze(0)
                        fut_world = torch_pos2global(fut_sdc.unsqueeze(0).to(device),
                                                        sdc_world_center.to(device),
                                                        sdc_world_rot.to(device)).squeeze(0)
                        fut_world_np = fut_world.cpu().numpy()
                        ax.plot(fut_world_np[:, 0], fut_world_np[:, 1], 'r.-', markersize=4, linewidth=1.5,
                                label='Future GT (World)' if not plotted_fut else "")
                        plotted_fut = True

            # Set plot limits slightly larger than boundary
            ax.set_xlim(xmin - 10, xmax + 10)
            ax.set_ylim(ymin - 10, ymax + 10)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X (True World)")
            ax.set_ylabel("Y (True World)")
            ax.set_title(f"Scene {i_scene} - True World Map & Trajectories")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()
            plt.close(fig)

class PredDictVisualizer:
    """
    Visualize the predicted trajectories in the True World Coordinates (TWC).
    """
    def __init__(self):
        pass
    
    def plot_probmap_visualization(self, batch, pred_dict, current_epoch=None, logger=None):
        """
        Plot predicted trajectories on SDF rasters following the logic from NLL metrics.
        Visualizes predictions from the first decoder (index 0).
        
        Args:
            batch: The batch data containing ground truth and reference information (already on CPU)
            pred_dict: Dictionary containing model predictions (already on CPU)
            current_epoch: Current training epoch for logging
            logger: PyTorch Lightning logger for WandB logging
        """
        # Check if required components are in the batch
        required_keys = ["gt/sdf_map", "gt/scenario_center", "gt/scenario_yaw", 
                         "gt/sim2_R", "gt/sim2_t", "gt/sim2_s", "ref/pos", "ref/rot", 
                         "ref/type", "gt/pos", "gt/valid"]
        
        for key in required_keys:
            if key not in batch:
                log.warn(f"Missing required key {key} for SDF visualization. Skipping.")
                return
                
        # Check if predictions exist
        if "pred" not in pred_dict or pred_dict["pred"] is None:
                log.warn("Missing 'pred' key in pred_dict for SDF visualization. Skipping.")
                return

        # Extract necessary components (all already on CPU)
        gt_sdf_map = batch["gt/sdf_map"]
        gt_scenario_center = batch["gt/scenario_center"]
        gt_scenario_yaw = batch["gt/scenario_yaw"]
        gt_sim2_R = batch["gt/sim2_R"]
        gt_sim2_t = batch["gt/sim2_t"]
        gt_sim2_s = batch["gt/sim2_s"]
        ref_pos = batch["ref/pos"]
        ref_rot = batch["ref/rot"]
        ref_type = batch["ref/type"]
        gt_pos = batch["gt/pos"]
        gt_valid = batch["gt/valid"]
        
        # Get dimensions from pred_dict
        n_decoder = pred_dict["pred"].shape[0] 
        n_scene = pred_dict["pred"].shape[1]
        n_agent = pred_dict["pred"].shape[2]
        n_pred = pred_dict["pred"].shape[3]  # Number of prediction modes
        
        # Initialize visualization data
        viz_data = {'pred_world': {}, 'gt_world': {}, 'gt_valid': {}, 'type': {}}

        # Define the decoder index to use for visualization (e.g., the first one)
        decoder_idx_to_vis = 0
        if n_decoder <= decoder_idx_to_vis:
                log.warn(f"Decoder index {decoder_idx_to_vis} is out of bounds (n_decoder={n_decoder}). Skipping visualization.")
                return
        
        # For each scene and agent, transform trajectories to world coordinates
        for j in range(n_scene):
            # Skip scenes with missing transform components
            if gt_sim2_R[j] is None or gt_sim2_t[j] is None or gt_sim2_s[j] is None:
                continue
                
            # Get SDC World Pose
            sdc_world_center = gt_scenario_center[j].unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
            sdc_world_yaw = gt_scenario_yaw[j]  # scalar tensor
            sdc_world_rot = torch_rad2rot(sdc_world_yaw).unsqueeze(0)  # [1, 2, 2]
            
            # Process each agent in the scene
            for k in range(n_agent):
                # Skip agents without valid trajectories
                if not gt_valid[j, k].any():
                    continue
                    
                # Get agent reference pose relative to SDC
                # Ensure correct shapes: ref_pos [1, 2], ref_rot [2, 2]
                agent_ref_pos = ref_pos[j, k] 
                agent_ref_rot = ref_rot[j, k] 
                if agent_ref_pos.dim() == 1: agent_ref_pos = agent_ref_pos.unsqueeze(0) # -> [1, 2]
                if agent_ref_rot.dim() != 2 or agent_ref_rot.shape != (2, 2):
                        log.warn(f"Unexpected ref_rot shape for agent {k} in scene {j}: {agent_ref_rot.shape}. Skipping agent.")
                        continue

                # Process each prediction mode for the selected decoder
                for p in range(n_pred):
                    try:
                        # Get predicted trajectory for this mode (full features) using the selected decoder index
                        pred_pos_local_full = pred_dict["pred"][decoder_idx_to_vis, j, k, p]  # [n_step_future, 5]
                        
                        # Extract only the position (x,y) coordinates - first 2 columns
                        pred_pos_local = pred_pos_local_full[:, :2]  # [n_step_future, 2]
                        
                        # Transform prediction: Agent-Local -> SDC -> World
                        # Ensure inputs to torch_pos2global have correct batch dim: [1, N, 2], [1, 1, 2], [1, 2, 2]
                        pred_pos_sdc = torch_pos2global(
                            pred_pos_local.unsqueeze(0),       # [1, n_step_future, 2]
                            agent_ref_pos.unsqueeze(0),        # [1, 1, 2]
                            agent_ref_rot.unsqueeze(0)         # [1, 2, 2]
                        ).squeeze(0)                           # -> [n_step_future, 2]
                        
                        pred_pos_world = torch_pos2global(
                            pred_pos_sdc.unsqueeze(0),    # [1, n_step_future, 2]
                            sdc_world_center,             # [1, 1, 2]
                            sdc_world_rot                 # [1, 2, 2]
                        ).squeeze(0)                      # -> [n_step_future, 2]
                        
                        # Store data for visualization - include prediction mode in the key
                        # Key remains (j, k, p) as we are visualizing modes for a single decoder
                        viz_key = (j, k, p)
                        viz_data['pred_world'][viz_key] = pred_pos_world.detach().cpu().numpy()
                        
                        # Only store GT and type once per agent (not per prediction mode)
                        if p == 0:
                            # Get validity mask for this agent's trajectory
                            validity_mask = gt_valid[j, k]  # [n_step_future]
                            
                            # Only transform valid GT points
                            gt_pos_local = gt_pos[j, k]  # [n_step_future, 2]
                            
                            # Transform valid points to world coordinates
                            gt_pos_sdc = torch_pos2global(
                                gt_pos_local.unsqueeze(0),   # [1, n_step_future, 2]
                                agent_ref_pos.unsqueeze(0),  # [1, 1, 2]
                                agent_ref_rot.unsqueeze(0)   # [1, 2, 2]
                            ).squeeze(0)                     # -> [n_step_future, 2]
                            
                            gt_pos_world = torch_pos2global(
                                gt_pos_sdc.unsqueeze(0),    # [1, n_step_future, 2]
                                sdc_world_center,           # [1, 1, 2]
                                sdc_world_rot               # [1, 2, 2]
                            ).squeeze(0)                    # -> [n_step_future, 2]
                            
                            # Store GT trajectory with validity mask
                            viz_data['gt_world'][(j, k)] = gt_pos_world.detach().cpu().numpy()
                            viz_data['gt_valid'][(j, k)] = validity_mask.detach().cpu().numpy()
                            
                            # Store agent type
                            agent_type_onehot = ref_type[j, k]
                            type_index = torch.argmax(agent_type_onehot.float()).item()
                            viz_data['type'][(j, k)] = type_index
                            
                    except Exception as e:
                        # Print detailed error including shapes
                        log.warning(f"Error processing scene {j}, agent {k}, prediction {p} for decoder {decoder_idx_to_vis}: {e}")
                        log.warning(f"  Shapes involved:")
                        if 'pred_pos_local_full' in locals(): log.warning(f"  pred_pos_local_full: {pred_pos_local_full.shape}")
                        if 'pred_pos_local' in locals(): log.warning(f"  pred_pos_local: {pred_pos_local.shape}")
                        if 'agent_ref_pos' in locals(): log.warning(f"  agent_ref_pos: {agent_ref_pos.shape}")
                        if 'agent_ref_rot' in locals(): log.warning(f"  agent_ref_rot: {agent_ref_rot.shape}")
                        if 'pred_pos_sdc' in locals(): log.warning(f"  pred_pos_sdc: {pred_pos_sdc.shape}")
                        if 'sdc_world_center' in locals(): log.warning(f"  sdc_world_center: {sdc_world_center.shape}")
                        if 'sdc_world_rot' in locals(): log.warning(f"  sdc_world_rot: {sdc_world_rot.shape}")
                        continue
        
        # Generate and log visualizations
        plt_figs = []
        
        # Create visualizations
        for j_vis in range(n_scene):
            # Check if scene was processed for the visualized decoder
            if not any((j_vis, k_vis, 0) in viz_data['pred_world'] for k_vis in range(n_agent)):
                continue
                
            # Get SDC World Pose for this scene
            sdc_world_center_scene_vis = gt_scenario_center[j_vis].unsqueeze(0).unsqueeze(0)
            
            # Map details
            scene_sdf_map_vis = gt_sdf_map[j_vis]
            # Ensure transform components are tensors before creating SimpleTransform if needed
            sim2_R_tensor = gt_sim2_R[j_vis] if isinstance(gt_sim2_R[j_vis], torch.Tensor) else torch.tensor(gt_sim2_R[j_vis])
            sim2_t_tensor = gt_sim2_t[j_vis] if isinstance(gt_sim2_t[j_vis], torch.Tensor) else torch.tensor(gt_sim2_t[j_vis])
            sim2_s_tensor = gt_sim2_s[j_vis] if isinstance(gt_sim2_s[j_vis], torch.Tensor) else torch.tensor(gt_sim2_s[j_vis])

            sim2_transform_vis = SimpleTransform(
                R=sim2_R_tensor, 
                t=sim2_t_tensor, 
                s=sim2_s_tensor
            )
            
            # Prepare plot
            orig_dims = batch.get("gt/sdf_map_orig_dims", None)
            if orig_dims is None:
                orig_height, orig_width = scene_sdf_map_vis.shape
            else:
                # Ensure orig_dims access is safe
                if j_vis < len(orig_dims):
                    orig_dims_scene = orig_dims[j_vis]
                    orig_height, orig_width = orig_dims_scene[0], orig_dims_scene[1]
                else:
                    orig_height, orig_width = scene_sdf_map_vis.shape # Fallback

            # Increased DPI for higher resolution visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=300)
            scene_sdf_map_padded_vis = scene_sdf_map_vis
            # Ensure cropping is safe
            scene_sdf_map_cropped_vis = scene_sdf_map_padded_vis[:int(orig_height), :int(orig_width)]
            
            # Set up coordinate transformation for plotting
            s_val_tensor = sim2_transform_vis.s
            # Ensure s_val is a scalar number
            s_val = s_val_tensor.item() if torch.is_tensor(s_val_tensor) and s_val_tensor.numel() == 1 else float(s_val_tensor) if not isinstance(s_val_tensor, (int, float)) else s_val_tensor
            
            t_val_tensor = sim2_transform_vis.t
            # Ensure t_val is a numpy array
            t_val = t_val_tensor.cpu().numpy() if torch.is_tensor(t_val_tensor) else np.array(t_val_tensor)

            # Ensure calculations are valid
            if s_val != 0:
                true_x_min = -t_val[0]
                true_y_min = -t_val[1]
                true_x_max = true_x_min + orig_width / s_val
                true_y_max = true_y_min + orig_height / s_val
                extent = [true_x_min, true_x_max, true_y_min, true_y_max]
            else:
                # Handle case where scale is zero, provide default extent or log warning
                log.warn(f"Zero scale factor encountered for scene {j_vis}. Using default extent.")
                extent = None

            
            # Plot SDF map
            im = ax.imshow(scene_sdf_map_cropped_vis, cmap='coolwarm', origin='lower', 
                        extent=extent, alpha=0.7, aspect='auto' if extent is None else 'equal')
            fig.colorbar(im, ax=ax, label='Signed distance (m)')
            
            # Define agent type styling
            type_map = {
                0: ('Vehicle', 'red', '.'),    # GT Vehicle is red
                1: ('Pedestrian', 'green', 's'), 
                2: ('Cyclist', 'magenta', 'D')
            }
            
            # Define prediction mode colors
            pred_colors = ['blue', 'orange', 'gold', 'limegreen', 'deepskyblue', 'violet']
            
            labeled_types = set()
            labeled_preds = set()
            
            # Plot trajectories for each agent FIRST, before plotting the SDC center
            for k_vis in range(n_agent):
                # Check if this agent has predictions for the visualized decoder
                if not any((j_vis, k_vis, p_vis) in viz_data['pred_world'] for p_vis in range(n_pred)):
                    continue
                    
                # Get GT and type info (same for all prediction modes)
                agent_key = (j_vis, k_vis)
                if agent_key not in viz_data['gt_world']:
                    continue
                    
                gt_traj_world_data = viz_data['gt_world'][agent_key]
                gt_valid_mask = viz_data['gt_valid'][agent_key]  # Get validity mask
                type_index = viz_data['type'][agent_key]
                type_label, type_color, type_marker = type_map.get(type_index, ('Unknown', 'gray', '.'))
                
                # Plot each prediction mode with different colors
                for p_vis in range(n_pred):
                    pred_key = (j_vis, k_vis, p_vis)
                    if pred_key not in viz_data['pred_world']:
                        continue
                        
                    pred_traj_world_data = viz_data['pred_world'][pred_key]
                    pred_color = pred_colors[p_vis % len(pred_colors)]
                    
                    # Create explicit copies of the data
                    pred_x = pred_traj_world_data[:, 0].copy()
                    pred_y = pred_traj_world_data[:, 1].copy()
                    
                    # Plot prediction with mode-specific color
                    mode_label = f"Pred Mode {p_vis}" if p_vis not in labeled_preds else ""
                    
                    ax.scatter(pred_x, pred_y, c=pred_color, s=15, marker='.', label=mode_label)
                    ax.plot(pred_x, pred_y, color=pred_color, linestyle='-', alpha=0.4)
                    
                    if mode_label:
                        labeled_preds.add(p_vis)
                
                # Filter GT data by validity mask
                valid_indices = np.where(gt_valid_mask)[0]  # Get indices of valid points
                
                if len(valid_indices) > 0:  # Only plot if there are valid points
                    # Extract valid GT points
                    valid_gt_x = gt_traj_world_data[valid_indices, 0].copy()
                    valid_gt_y = gt_traj_world_data[valid_indices, 1].copy()
                    
                    # Plot GT using only valid points
                    current_label = f"GT {type_label}" if type_label not in labeled_types else ""
                    ax.scatter(valid_gt_x, valid_gt_y, c=type_color, s=15, marker=type_marker, label=current_label, alpha=0.5)
                    
                    # Only connect valid points - and only if there are at least 2 points
                    if len(valid_indices) > 1:
                        ax.plot(valid_gt_x, valid_gt_y, color=type_color, linestyle='-', alpha=0.5)
                    
                    if current_label:
                        labeled_types.add(type_label)
            
            # Now plot SDC center AFTER all trajectories, with a clear label to prevent connection
            # Plot SDC center
            scenario_center_np_vis = sdc_world_center_scene_vis.squeeze().cpu().numpy()
            ax.scatter(scenario_center_np_vis[0], scenario_center_np_vis[1], 
                        c='black', s=120, marker='*', label='SDC Center (World)', zorder=10)
            # Reference lines can be removed if needed
            ax.axhline(y=scenario_center_np_vis[1], color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=scenario_center_np_vis[0], color='black', linestyle='--', alpha=0.5)
            
            # Finalize plot
            ax.legend()
            ax.set_xlabel('X (True World)')
            ax.set_ylabel('Y (True World)')
            ax.set_title(f"Scene {j_vis} - Probability Map & Trajectories (Decoder {decoder_idx_to_vis})")
            ax.grid(True, alpha=0.3)
            if extent is not None:
                ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt_figs.append(wandb.Image(img_array, caption=f"Scene {j_vis} - Decoder {decoder_idx_to_vis}"))

            # Close figure to free memory
            plt.close(fig)
        
        # Log visualizations to wandb every 10 epochs TODO: Add this to config and check if it is working
        if plt_figs and current_epoch is not None and logger is not None:
            # Use the PyTorch Lightning logger instead of direct wandb.log()
            logger.experiment.log({f"epoch_{current_epoch}/probability_map_trajectories": plt_figs})

class AnchorSelectionVisualizer:
    """
    Visualize the anchor selection process: plots the ground truth trajectory
    and the Q anchor trajectories selected by the model onto the probability map
    in True World Coordinates (TWC).
    """
    def __init__(self):
        # Static methods don't use self, init can be empty or removed if not needed elsewhere
        pass

    @staticmethod
    def visualize_anchor_selection_trajectories(
        anchor_viz_data,          
        gt_scenario_center,       
        gt_sdf_map,               
        gt_sdf_map_orig_dims,     
        gt_sim2_R,                
        gt_sim2_t,                
        gt_sim2_s,                
        n_scene,                  
        n_agent,
        wandb_logger=None,        
        current_epoch=None, 
    ):
        """
        Visualizes the selected anchor trajectories and the ground truth trajectory on the SDF raster,
        using consistent colors for each agent and its anchors. Logs to WandB if logger is provided.
        """
        log.info("Starting anchor selection visualization...")
        
        import matplotlib.cm as cm
        import wandb 
        import numpy as np 
        
        wandb_imgs = [] 
        corrected_labeled = False

        for j_vis in range(n_scene):
            # Skip scenes with missing components
            if gt_sim2_R is None or gt_sim2_t is None or gt_sim2_s is None or \
               j_vis >= len(gt_sim2_R) or j_vis >= len(gt_sim2_t) or j_vis >= len(gt_sim2_s) or \
               gt_sim2_R[j_vis] is None or gt_sim2_t[j_vis] is None or gt_sim2_s[j_vis] is None:
                 log.warning(f"Skipping anchor viz for scene {j_vis} due to missing or out-of-bounds Sim(2) components.")
                 continue

            has_data_for_scene = any(key[0] == j_vis for key in anchor_viz_data.get('selected_anchors_world', {}))
            if not has_data_for_scene:
                log.debug(f"No anchor data for scene {j_vis}, skipping visualization for this scene.")
                continue

            # Get SDC World Pose for this scene
            sdc_world_center_scene_vis = gt_scenario_center[j_vis].unsqueeze(0).unsqueeze(0) 

            # Map details
            scene_sdf_map_vis = gt_sdf_map[j_vis]
            
            # Get original map dimensions
            if gt_sdf_map_orig_dims is None or j_vis >= len(gt_sdf_map_orig_dims):
                orig_height, orig_width = scene_sdf_map_vis.shape
                log.warning(f"Using full map dimensions for scene {j_vis} due to missing or out-of-bounds gt_sdf_map_orig_dims.")
            else:
                orig_dims = gt_sdf_map_orig_dims[j_vis].cpu().int().numpy()
                orig_height, orig_width = orig_dims[0], orig_dims[1]

            fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=120) # Assuming dpi=150 as in savefig previously
            scene_sdf_map_padded_vis = scene_sdf_map_vis
            # Ensure cropping is safe
            scene_sdf_map_cropped_vis = scene_sdf_map_padded_vis[:int(orig_height), :int(orig_width)]


            # Create properly formatted Sim2 transform components
            sim2_R_tensor = gt_sim2_R[j_vis] if isinstance(gt_sim2_R[j_vis], torch.Tensor) else torch.tensor(gt_sim2_R[j_vis])
            sim2_t_tensor = gt_sim2_t[j_vis] if isinstance(gt_sim2_t[j_vis], torch.Tensor) else torch.tensor(gt_sim2_t[j_vis])
            sim2_s_tensor = gt_sim2_s[j_vis] if isinstance(gt_sim2_s[j_vis], torch.Tensor) else torch.tensor(gt_sim2_s[j_vis])
            
            # Create transform explicitly
            sim2_transform_vis = SimpleTransform(
                R=sim2_R_tensor, 
                t=sim2_t_tensor, 
                s=sim2_s_tensor
            )
            
            # Calculate extent
            s_val_tensor = sim2_transform_vis.s
            s_val = s_val_tensor.item() if torch.is_tensor(s_val_tensor) and s_val_tensor.numel() == 1 else float(s_val_tensor)
            
            t_val_tensor = sim2_transform_vis.t
            t_val = t_val_tensor.cpu().numpy() if torch.is_tensor(t_val_tensor) else np.array(t_val_tensor)

            extent = None
            if s_val != 0:
                true_x_min = -t_val[0] 
                true_y_min = -t_val[1]
                true_x_max = true_x_min + orig_width / s_val
                true_y_max = true_y_min + orig_height / s_val
                extent = [true_x_min, true_x_max, true_y_min, true_y_max]
            else:
                log.warning(f"Zero scale factor (s_val) encountered for scene {j_vis}. Cannot determine world extent.")

            # Plot map background if extent is valid
            if extent is not None:
                ax.imshow(scene_sdf_map_cropped_vis, cmap='coolwarm', origin='lower', 
                          extent=extent, alpha=0.7, aspect='equal')
            else:
                log.warning(f"Skipping map background plot for scene {j_vis}.")

            # Plot SDC center in world coordinates
            scenario_center_np_vis = sdc_world_center_scene_vis.squeeze().cpu().numpy()
            ax.scatter(scenario_center_np_vis[0], scenario_center_np_vis[1], 
                    c='black', s=120, marker='*', label='SDC Center (World)', zorder=10)
            ax.axhline(y=scenario_center_np_vis[1], color='black', linestyle='--', alpha=0.3)
            ax.axvline(x=scenario_center_np_vis[0], color='black', linestyle='--', alpha=0.3)

            # --- Plot Trajectories ---
            valid_agents = []
            for k_vis in range(n_agent):
                viz_key = (j_vis, k_vis)
                if viz_key in anchor_viz_data.get('selected_anchors_world', {}) and \
                   viz_key in anchor_viz_data.get('gt_world', {}):
                    valid_agents.append(k_vis)
            
            agent_colors = {}
            agent_cmap = cm.get_cmap('tab10') 
            for i, agent_idx in enumerate(valid_agents):
                agent_colors[agent_idx] = agent_cmap(i % 10) 
            
            type_names = { 0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist' }
            labeled_agents = set()
            labeled_anchors = False
            max_endpoint_dist = 0  

            for agent_idx in valid_agents:
                viz_key = (j_vis, agent_idx)
                gt_traj_world = anchor_viz_data['gt_world'][viz_key]          
                anchor_trajs_world = anchor_viz_data['selected_anchors_world'][viz_key] 
                type_index = anchor_viz_data['type'][viz_key]
                type_name = type_names.get(type_index, 'Unknown')
                agent_color = agent_colors[agent_idx]
                Q_viz = anchor_trajs_world.shape[0]

                gt_label = f"Agent {agent_idx} (GT {type_name})"
                label_gt_plot = gt_label if agent_idx not in labeled_agents else ""
                ax.plot(gt_traj_world[:, 0], gt_traj_world[:, 1], 
                        color=agent_color, 
                        linestyle='-', 
                        linewidth=2.5, 
                        marker='.', 
                        markersize=5, 
                        alpha=1.0,
                        label=label_gt_plot,
                        zorder=6)
                
                if label_gt_plot:
                    labeled_agents.add(agent_idx)

                for q_idx in range(Q_viz):
                    anchor_label = None
                    if q_idx == 0 and not labeled_anchors:
                        anchor_label = f"Selected Anchors (all agents)"
                        labeled_anchors = True
                    
                    ax.plot(anchor_trajs_world[q_idx, :, 0], anchor_trajs_world[q_idx, :, 1],
                            color=agent_color, 
                            linestyle='--', 
                            linewidth=1.0, 
                            alpha=0.2, 
                            marker=None,
                            label=anchor_label, 
                            zorder=4)
                    
                    endpoint = anchor_trajs_world[q_idx, -1]
                    ax.scatter(endpoint[0], endpoint[1],
                               color=agent_color,
                               s=40, 
                               marker='o', 
                               zorder=3, 
                               alpha=0.2)
                    
                    if len(gt_traj_world) > 0:
                        dist = np.linalg.norm(endpoint - gt_traj_world[-1])
                        max_endpoint_dist = max(max_endpoint_dist, dist)

                # --- Optional: overlay corrected endpoints from offset_pred ---
                corrected_key = 'corrected_endpoints_world'
                anchor_ep_key = 'anchor_endpoints_world'
                if corrected_key in anchor_viz_data and viz_key in anchor_viz_data[corrected_key]:
                    corrected_eps = anchor_viz_data[corrected_key][viz_key]  # [Q, 2]
                    anchor_eps = anchor_viz_data.get(anchor_ep_key, {}).get(viz_key, None)  # [Q, 2] or None

                    # Draw correction vectors (anchor endpoint -> corrected endpoint) if both available
                    if anchor_eps is not None and corrected_eps.shape[0] == anchor_eps.shape[0]:
                        for q_idx in range(corrected_eps.shape[0]):
                            x_vals = [anchor_eps[q_idx, 0], corrected_eps[q_idx, 0]]
                            y_vals = [anchor_eps[q_idx, 1], corrected_eps[q_idx, 1]]
                            ax.plot(x_vals, y_vals, color=agent_color, linewidth=1.2, alpha=0.6, zorder=7)

                    # Scatter corrected endpoints
                    label_corr = "Corrected endpoints" if 'corrected_labeled' not in locals() else ""
                    ax.scatter(corrected_eps[:, 0], corrected_eps[:, 1],
                               color=agent_color,
                               s=50,
                               marker='x',
                               linewidths=1.5,
                               zorder=8,
                               label=label_corr)
                    corrected_labeled = True

            if extent is None and max_endpoint_dist > 0:
                all_x_values, all_y_values = [], []
                for key_type in ['gt_world', 'selected_anchors_world']:
                    for key, traj_data in anchor_viz_data.get(key_type, {}).items():
                        if key[0] == j_vis:
                            if key_type == 'gt_world':
                                all_x_values.extend(traj_data[:, 0])
                                all_y_values.extend(traj_data[:, 1])
                            elif key_type == 'selected_anchors_world':
                                all_x_values.extend(traj_data.reshape(-1, 2)[:, 0])
                                all_y_values.extend(traj_data.reshape(-1, 2)[:, 1])
                
                if all_x_values and all_y_values:
                    margin = max(max_endpoint_dist * 0.5, 10.0)
                    x_min, x_max = min(all_x_values), max(all_x_values)
                    y_min, y_max = min(all_y_values), max(all_y_values)
                    ax.set_xlim(x_min - margin, x_max + margin)
                    ax.set_ylim(y_min - margin, y_max + margin)

            debug_text = f"Sim2 scale: {s_val:.4f}, t: [{t_val[0]:.1f}, {t_val[1]:.1f}]"
            ax.text(0.02, 0.02, debug_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7), 
                    fontsize=8,
                    zorder=10)

            ax.legend(loc='upper right')
            ax.set_xlabel('X (True World)')
            ax.set_ylabel('Y (True World)')
            ax.set_title(f"Scene {j_vis} - Anchor Selection Visualization")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.close(fig)
            # Convert matplotlib figure to wandb image
            try:
                fig.canvas.draw()
                img_array = np.array(fig.canvas.renderer.buffer_rgba())
                caption = f"Anchor Selection - Scene {j_vis}"
                if current_epoch is not None:
                    caption += f" Epoch {current_epoch}"
                wandb_imgs.append(wandb.Image(img_array, caption=caption))
            except Exception as e:
                log.error(f"Error converting plot to wandb.Image for scene {j_vis}: {e}")
            finally:
                plt.close(fig) # Close figure to free memory

        # Log all collected images to wandb if conditions are met
        if wandb_imgs and wandb_logger is not None and current_epoch is not None:
            try:
                wandb_logger.experiment.log({f"epoch_{current_epoch}/anchor_selection_visualizations": wandb_imgs})
            except Exception as e:
                log.error(f"Error logging anchor selection visualizations to wandb: {e}")
        elif not wandb_imgs:
            log.info("No images were generated for anchor selection visualization.")
