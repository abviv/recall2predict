import numpy as np
import h5py
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional, List, Union, Any
import logging
import random
import torch
from torch import Tensor
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

def generate_distance_transform(rasterized_da: np.ndarray) -> np.ndarray:
    """
    Generate distance transform from rasterized drivable area.
    
    Args:
        rasterized_da: Binary mask of drivable area
    
    Returns:
        Normalized distance transform
    """
    rasterized_da = np.array(Image.fromarray(rasterized_da).convert('L'))
    binary_mask = (rasterized_da > 0).astype(np.uint8) * 255
    binary_mask_inv = cv2.bitwise_not(binary_mask)
    dist_transform = cv2.distanceTransform(binary_mask_inv, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    return dist_transform

def generate_probability_map(dist_transform: np.ndarray) -> np.ndarray:
    """
    Generate probability map from distance transform.
    
    Args:
        dist_transform: Normalized distance transform
    
    Returns:
        Final probability map
    """
    dist_transform[dist_transform < 0] = 0
    max_value = np.max(dist_transform)
    inverted_transform = max_value - dist_transform
    normalized = inverted_transform / np.max(inverted_transform)
    emphasized = np.exp(normalized * 5)
    final_map = emphasized / np.max(emphasized)
    
    return final_map

class SimpleTransform:
    """A simple replacement for from av2.geometry.sim2 that handles 2D transformations with scale and translation."""
    def __init__(self, R=None, t=None, s=1.0):
        """
        Initialize a simple transformation.
        
        Args:
            R: 2x2 rotation matrix (defaults to identity)
            t: 2D translation vector
            s: scale factor
        """
        self.R = np.eye(2) if R is None else R
        self.t = np.zeros(2) if t is None else t
        self.s = s
        
    def transform_from(self, points):
        """
        Transform points from source to destination coordinate system.
        
        Args:
            points: Nx2 array of points
            
        Returns:
            Transformed points
        """
        return self.s * (self.R @ points.T).T + self.t
    
    def transform_to(self, points):
        """
        Transform points from destination to source coordinate system.
        
        Args:
            points: Nx2 array of points
            
        Returns:
            Transformed points
        """
        return (np.linalg.inv(self.R) @ ((points - self.t).T / self.s)).T


def get_probability_map_from_raster(raster_filepath: str, scenario_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generate a probability map from raster data for a given scenario key.
    If scenario_key is not provided, a random scenario will be selected.
    
    Args:
        raster_filepath: Path to the HDF5 file containing raster data
        scenario_key: Optional scenario ID to process. If None, a random scenario is selected.
        
    Returns:
        Dictionary containing:
            - 'probability_map': Generated probability map
            - 'scenario_key': The scenario key that was processed
            - 'raster': Original raster data
            - 'sim2_R': Rotation matrix from Sim2 transform
            - 'sim2_s': Scale from Sim2 transform
            - 'sim2_t': Translation from Sim2 transform
        Returns None if scenario_key is not found or an error occurs
    """
    try:
        with h5py.File(raster_filepath, "r", libver="latest", swmr=True) as hf:
            # If no scenario key is provided, select a random one
            if scenario_key is None:
                available_keys = list(hf.keys())
                if not available_keys:
                    logger.warning("No scenarios found in the raster file")
                    return None
                scenario_key = random.choice(available_keys)
                logger.info(f"Randomly selected scenario: {scenario_key}")
            
            if scenario_key not in hf:
                logger.warning(f"Scenario key '{scenario_key}' not found in raster file")
                return None
            
            # Extract raster data
            raster_data = np.ascontiguousarray(hf[scenario_key]["raster"])
            
            # Extract Sim2 transform data
            sim2_group = hf[scenario_key]["sim2_transform"]
            sim2_R = np.ascontiguousarray(sim2_group["R"])
            sim2_s = np.ascontiguousarray(sim2_group["s"])
            sim2_t = np.ascontiguousarray(sim2_group["t"])
            
            # Generate probability map
            dist_transform = generate_distance_transform(raster_data)
            prob_map = generate_probability_map(dist_transform)
            
            return {
                "probability_map": prob_map,
                "scenario_key": scenario_key,
                "raster": raster_data,
                "sim2_R": sim2_R,
                "sim2_s": sim2_s,
                "sim2_t": sim2_t
            }
            
    except Exception as e:
        logger.error(f"Error generating probability map for scenario {scenario_key}: {e}")
        return None


class CoordinateTransformAdapter:
    """
    Adapter class for coordinate transformations between different frames:
    - True World: The absolute coordinate system of the dataset.
    - SDC-Centered: Coordinates relative to the SDC's pose at a reference timestamp (usually time 0).
    - Agent-Local: Coordinates relative to a specific agent's pose at a reference timestamp.
    """

    @staticmethod
    def local_to_sdc(positions: Tensor, sdc_centers: Tensor, sdc_yaws: Tensor) -> Tensor:
        """
        Transforms positions from agent-local coordinates TO SDC-centered coordinates.
        WARNING: This might be conceptually incorrect depending on how 'local' is defined.
                 Usually, local_to_global (using agent's world pose) is needed first.
                 If 'local' here *means* relative to the agent's SDC-centered pose, then this function is misnamed
                 and actually performs local -> SDC-centered transform using the agent's SDC pose, not the SDC's world pose.
                 Assuming it's intended to be local -> SDC using agent's SDC pose for now.
        """
        # Placeholder - implementation depends on the exact definition of 'local'
        # If local is agent-local relative to SDC frame:
        # return torch_pos2global(positions, agent_sdc_pos, agent_sdc_rot) # Requires agent's SDC pose
        # If local is agent-local relative to world frame:
        # 1. local -> world using agent world pose
        # 2. world -> sdc using sdc world pose
        # This function seems incorrectly used in the previous NllMetrics attempt.
        raise NotImplementedError("local_to_sdc logic needs clarification based on 'local' definition")


    @staticmethod
    def sdc_to_true_world(positions: Tensor, centers: Tensor, yaws: Tensor) -> Tensor:
        """
        Transforms positions from SDC-centered coordinates to true world coordinates.

        Args:
            positions: Tensor of positions in SDC frame [..., N, 2].
            centers: Tensor of SDC center positions in world frame [..., N, 2].
            yaws: Tensor of SDC yaws in world frame [..., N, 1].

        Returns:
            Tensor of positions in world frame [..., N, 2].
        """
        device = positions.device
        # Use float32 consistently
        dtype = torch.float32

        # Ensure inputs are float32
        positions = positions.to(dtype)
        centers = centers.to(dtype)
        yaws = yaws.to(dtype)

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)

        # Squeeze the last dimension for assignment
        cos_yaw_squeezed = cos_yaw.squeeze(-1) # Shape [..., N]
        sin_yaw_squeezed = sin_yaw.squeeze(-1) # Shape [..., N]

        # Create rotation matrices [..., N, 2, 2] as float32
        # Construct shape carefully to match broadcasting needs
        rot_shape = list(positions.shape[:-1]) + [2, 2]
        rot_matrices = torch.zeros(rot_shape, device=device, dtype=dtype)

        # Populate rotation matrices
        rot_matrices[..., 0, 0] = cos_yaw_squeezed
        rot_matrices[..., 0, 1] = -sin_yaw_squeezed
        rot_matrices[..., 1, 0] = sin_yaw_squeezed
        rot_matrices[..., 1, 1] = cos_yaw_squeezed

        # Apply rotation
        # positions: [..., N, 2] -> [..., N, 1, 2]
        # rot_matrices: [..., N, 2, 2]
        rotated_positions = torch.matmul(positions.unsqueeze(-2), rot_matrices).squeeze(-2)

        # Add center translation
        world_positions = rotated_positions + centers

        return world_positions

    @staticmethod
    def true_world_to_sdc(points: torch.Tensor,
                         scenario_center: torch.Tensor,
                         scenario_yaw: torch.Tensor) -> torch.Tensor:
        """
        Transform points from true world coordinates to SDC-centered world coordinates.
        
        Args:
            points: Points in true world coordinates [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
            scenario_center: Center position of the scenario (SDC) in true world coordinates [n_scene, 2]
            scenario_yaw: Yaw angle of the SDC in radians [n_scene]
            
        Returns:
            Points in SDC-centered world coordinates [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
        """
        # Get tensor shapes and device
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = points.shape
        device = points.device
        dtype = points.dtype
        
        # We need to expand scenario_center and scenario_yaw to match points dimensions
        # Expand scenario_center: [n_scene, 2] -> [n_decoder, n_scene, n_agent, n_pred, 2]
        if scenario_center.dim() == 2:  # [n_scene, 2]
            scenario_center = scenario_center.unsqueeze(1).unsqueeze(1)  # [n_scene, 1, 1, 2]
            scenario_center = scenario_center.expand(n_scene, n_agent, n_pred, 2)  # [n_scene, n_agent, n_pred, 2]
            scenario_center = scenario_center.unsqueeze(0)  # [1, n_scene, n_agent, n_pred, 2]
            scenario_center = scenario_center.expand(n_decoder, -1, -1, -1, -1)  # [n_decoder, n_scene, n_agent, n_pred, 2]
        
        # Expand scenario_yaw: [n_scene] -> [n_decoder, n_scene, n_agent, n_pred]
        if scenario_yaw.dim() == 1:  # [n_scene]
            scenario_yaw = scenario_yaw.unsqueeze(1).unsqueeze(1)  # [n_scene, 1, 1]
            scenario_yaw = scenario_yaw.expand(n_scene, n_agent, n_pred)  # [n_scene, n_agent, n_pred]
            scenario_yaw = scenario_yaw.unsqueeze(0)  # [1, n_scene, n_agent, n_pred]
            scenario_yaw = scenario_yaw.expand(n_decoder, -1, -1, -1)  # [n_decoder, n_scene, n_agent, n_pred]
        
        # Reshape scenario_center for broadcasting
        scenario_center_expanded = scenario_center.unsqueeze(-2).expand(-1, -1, -1, -1, n_step_future, -1)
        
        # Center points
        centered_points = points - scenario_center_expanded
        
        # Create rotation matrices for inverse transformation (negative yaw)
        cos_yaw = torch.cos(-scenario_yaw)
        sin_yaw = torch.sin(-scenario_yaw)
        
        # Create rotation matrices [n_decoder, n_scene, n_agent, n_pred, 2, 2]
        rot_matrices = torch.zeros((n_decoder, n_scene, n_agent, n_pred, 2, 2), dtype=dtype, device=device)
        rot_matrices[..., 0, 0] = cos_yaw
        rot_matrices[..., 0, 1] = -sin_yaw
        rot_matrices[..., 1, 0] = sin_yaw
        rot_matrices[..., 1, 1] = cos_yaw
        
        # Reshape for matrix multiplication
        centered_points_reshaped = centered_points.reshape(n_decoder * n_scene * n_agent * n_pred, n_step_future, 2)
        rot_matrices_reshaped = rot_matrices.reshape(n_decoder * n_scene * n_agent * n_pred, 2, 2)
        
        # Apply rotation
        rotated_points = torch.bmm(centered_points_reshaped, rot_matrices_reshaped.transpose(1, 2))
        
        # Reshape back
        return rotated_points.reshape(n_decoder, n_scene, n_agent, n_pred, n_step_future, 2)

def extract_probabilities_from_map(probability_map, trajectories_world, sim2_transform):
    """
    Extract probability values from the probability map for each point in the trajectories using PyTorch.
    
    Args:
        probability_map (torch.Tensor): Probability map array [height, width].
        trajectories_world (torch.Tensor): Trajectory points in world coordinates [n_pred, n_steps, 2].
        sim2_transform (SimpleTransform): Transform from world to pixel coordinates.
            Contains t (translation), s (scale), R (rotation matrix) attributes.
        
    Returns:
        torch.Tensor: Probability values for each point [n_pred, n_steps].
    """
    # Ensure inputs are torch tensors
    if isinstance(probability_map, np.ndarray):
        probability_map = torch.from_numpy(probability_map)
    if isinstance(trajectories_world, np.ndarray):
        trajectories_world = torch.from_numpy(trajectories_world)
    
    # Get map dimensions
    map_height, map_width = probability_map.shape
    device = trajectories_world.device
    
    # Create empty tensor for probabilities
    probabilities = torch.zeros(trajectories_world.shape[0], trajectories_world.shape[1], 
                               device=device, dtype=torch.float32)
    
    # Get transform parameters
    # Handle both tensor and non-tensor attributes
    t = sim2_transform.t
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)
    
    s = sim2_transform.s
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, device=device)
    
    # Apply transformation to each point
    for pred_idx in range(trajectories_world.shape[0]):
        for step_idx in range(trajectories_world.shape[1]):
            # Get world coordinates
            point_world = trajectories_world[pred_idx, step_idx]
            
            # Transform to pixel coordinates
            pixel_x = torch.round((point_world[0] + t[0]) * s).long()
            pixel_y = torch.round((point_world[1] + t[1]) * s).long()
            
            # Check if coordinates are within bounds
            if (0 <= pixel_x < map_width) and (0 <= pixel_y < map_height):
                # Extract probability value
                probabilities[pred_idx, step_idx] = probability_map[pixel_y, pixel_x]
    
    return probabilities

def extract_probabilities_from_map_vectorized(probability_map, trajectories_world, sim2_transform):
    """
    Vectorized version that extracts probabilities for all trajectory points at once.
    
    Args:
        probability_map (torch.Tensor): Probability map array [height, width].
        trajectories_world (torch.Tensor): Trajectory points in world coordinates [n_pred, n_steps, 2]
                                           or [n_steps, 2] for a single trajectory.
        sim2_transform (SimpleTransform): Transform from world to pixel coordinates.
        
    Returns:
        torch.Tensor: Probability values for each point with shape matching trajectories_world[..., 0].
    """
    import numpy as np
    import torch
    
    # Ensure inputs are torch tensors
    if isinstance(probability_map, np.ndarray):
        probability_map = torch.from_numpy(probability_map)
    if isinstance(trajectories_world, np.ndarray):
        trajectories_world = torch.from_numpy(trajectories_world)
    
    # Get map dimensions
    map_height, map_width = probability_map.shape
    device = trajectories_world.device
    
    # Get transform parameters as tensors
    t = sim2_transform.t
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)
    
    s = sim2_transform.s
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, device=device)
    
    # Extract x, y coordinates from trajectories
    x_world = trajectories_world[..., 0]  # shape: [n_pred, n_steps] or [n_steps]
    y_world = trajectories_world[..., 1]  # shape: [n_pred, n_steps] or [n_steps]
    
    # Apply transformation to get pixel coordinates
    if t.dim() == 0:  # scalar tensor
        pixel_x = torch.round((x_world + t) * s).long()
        pixel_y = torch.round((y_world + t) * s).long()
    else:  # vector tensor
        t_x = t[0] if t.dim() > 0 and t.size(0) > 0 else t
        t_y = t[1] if t.dim() > 0 and t.size(0) > 1 else t
        pixel_x = torch.round((x_world + t_x) * s).long()
        pixel_y = torch.round((y_world + t_y) * s).long()
    
    # Create mask for valid coordinates (within map bounds)
    valid_mask = (pixel_x >= 0) & (pixel_x < map_width) & (pixel_y >= 0) & (pixel_y < map_height)
    
    # Initialize probabilities tensor with zeros
    probabilities = torch.zeros_like(x_world, dtype=probability_map.dtype, device=device)
    
    # Get flat indices for valid points
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)
    
    # Extract probabilities for valid points
    valid_pixel_x = pixel_x[valid_indices]
    valid_pixel_y = pixel_y[valid_indices]
    
    # Index into probability map and assign values
    probabilities[valid_indices] = probability_map[valid_pixel_y, valid_pixel_x]
    
    debug_plot = False # This is meant to be local variable
    if debug_plot:
        # Plot the scene probability map and overlay the projected trajectory positions.
        if not hasattr(extract_probabilities_from_map_vectorized, 'plotted'):
            import matplotlib.pyplot as plt
            # Convert probability map to numpy (for imshow) and use it as background.
            prob_map_np = probability_map.detach().cpu().numpy()
            # Calculate world coordinate extents for visualization
            height, width = probability_map.shape
            
            # Get transform parameters for extent calculation
            if hasattr(sim2_transform, 't') and hasattr(sim2_transform, 's'):
                s_val = sim2_transform.s
                if not isinstance(s_val, (int, float)) and hasattr(s_val, 'item'):
                    s_val = s_val.item()
                
                t_val = sim2_transform.t
                if isinstance(t_val, torch.Tensor):
                    t_val = t_val.cpu().numpy()
                
                # Calculate world coordinate boundaries
                true_x_min = -t_val[0]
                true_y_min = -t_val[1]
                true_x_max = true_x_min + width / s_val
                true_y_max = true_y_min + height / s_val
                
                # Define the extent for proper scaling in the plot
                extent = [true_x_min, true_x_max, true_y_min, true_y_max]
            else:
                extent = None
            plt.figure(figsize=(8, 6))
            plt.imshow(prob_map_np, cmap='viridis', origin='upper', extent=extent, alpha=0.7)
            plt.colorbar(label='Probability')
            
            # # Determine which trajectory to plot.
            # # If trajectories_world is 2D, it is assumed to be a single trajectory: [n_steps, 2]
            # # If 3D, we select the first trajectory in the batch.
            # if trajectories_world.dim() == 2:
            #     traj_np = trajectories_world.detach().cpu().numpy()
            #     px_np = pixel_x.detach().cpu().numpy()
            #     py_np = pixel_y.detach().cpu().numpy()
            # elif trajectories_world.dim() == 3:
            #     traj_np = trajectories_world[0].detach().cpu().numpy()  # shape: [n_steps, 2]
            #     px_np = pixel_x[0].detach().cpu().numpy()
            #     py_np = pixel_y[0].detach().cpu().numpy()
            # else:
            #     traj_np = trajectories_world.detach().cpu().numpy()
            #     px_np = pixel_x.detach().cpu().numpy()
            #     py_np = pixel_y.detach().cpu().numpy()
            
            # # Overlay the trajectory positions on the probability map.
            # plt.scatter(px_np, py_np, c='red', s=30, label='Trajectory')
            plt.title('Scene Probability Map with Trajectory Projection')
            plt.legend()
            plt.show()
            
            # Ensure we only plot once
            # setattr(extract_probabilities_from_map_vectorized, 'plotted', True)
    
    return probabilities