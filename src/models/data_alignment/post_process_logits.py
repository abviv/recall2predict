import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

log = logging.getLogger(__name__)

class PostProcessLogits(nn.Module):
    """
    Processes trajectory logits to derive trajectories.
    Converts class indices from logits into actual trajectory coordinates.
    """
    def __init__(self, n_pred: int, 
                 traj_tensor: Tensor, 
                 apply_softmax: bool = False,
                 use_classification_tower_head: bool = False):
        """
        Initializes the PostProcessLogits module.

        Args:
            n_pred (int): The number of predictions to generate (top-k from logits).
            traj_tensor (Tensor): A tensor containing the pre-defined trajectory anchors.
                                  Shape: [num_anchors, num_timesteps, 2].
        """
        super().__init__()
        self.n_pred = n_pred
        self.register_buffer('traj_tensor', traj_tensor)
        self.apply_softmax = apply_softmax
        self.use_classification_tower_head = use_classification_tower_head
    def forward(self, pred_dict: dict) -> dict:
        """
        Processes 'pred_pos_logits' in the pred_dict to generate 'pred_pos' and 'pred_conf'.

        Args:
            pred_dict: Specifically the pred_pos_logits key is expected to be present.

        Returns:
            pred_dict: Updated with keys 'pred_pos' and 'pred_conf'
        """
        if 'pred_pos_logits' in pred_dict and pred_dict['pred_pos_logits'] is not None:
            pred_pos_logits = pred_dict['pred_pos_logits']
            
            # For visualization, we expect a single ensemble member or a 4D tensor.
            if pred_pos_logits.dim() == 5 and pred_pos_logits.shape[3] == 1:
                # Shape: [n_decoder, n_scene, n_target, 1, n_classes] -> [n_decoder, n_scene, n_target, n_classes]
                pred_pos_logits_squeezed = pred_pos_logits.squeeze(3)
            elif pred_pos_logits.dim() == 4:
                 # Shape: [n_decoder, n_scene, n_target, n_classes]
                pred_pos_logits_squeezed = pred_pos_logits
            else:
                log.warning(f"Unsupported shape for pred_pos_logits: {pred_pos_logits.shape}. Expected 4 or 5 dims. Skipping conversion.")
                return pred_dict

            if self.apply_softmax:
                # Apply softmax to get class probabilities
                probs = F.softmax(pred_pos_logits_squeezed, dim=-1)
            else:
                probs = pred_pos_logits_squeezed
            
            # Get top-k class indices and probabilities/logits
            _topk_probs, topk_indices = torch.topk(probs, k=self.n_pred, dim=-1)
            
            # Get dimensions for trajectory conversion
            n_decoder, n_scene, n_target, n_pred = topk_indices.shape
            n_step_future = self.traj_tensor.shape[1]

            # Convert class indices to trajectories using vectorized operations
            flat_indices = topk_indices.view(-1)
            
            # Get trajectories for all indices at once
            flat_trajectories = self.traj_tensor[flat_indices]
            
            # Reshape back to desired format
            derived_trajectories = flat_trajectories.view(n_decoder, n_scene, n_target, n_pred, n_step_future, 2)
            
            if self.use_classification_tower_head:
                # Take this path only during the *_gmm_towers.yaml 
                pred_dict['pred_pos_classification'] = derived_trajectories 
                pred_dict['pred_conf_classification'] = _topk_probs
            
            else:
                # Take this path only during the *_classification_head.yaml
                # Replace pred_pos in pred_dict
                pred_dict['pred_pos'] = derived_trajectories 
                pred_dict['pred_conf'] = _topk_probs
                log.debug(f"Converted pred_pos_logits {pred_pos_logits.shape} -> pred_pos {derived_trajectories.shape} and pred_conf {_topk_probs.shape}")
            
        else:
            log.warning("No pred_pos_logits found in pred_dict")
            raise ValueError("No pred_pos_logits found in pred_dict. Are you sure you are using the classification head?")
        
        return pred_dict