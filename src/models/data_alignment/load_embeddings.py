import os
import torch
import logging
import random
log = logging.getLogger(__name__)

def load_embeddings(pretrained_emb_path, subset_percentage=0.10, device='cpu'):
    """
    Loads the embeddings from the pretrained_emb_path.
    """
    # torch.manual_seed(self.seed)
    # random.seed(self.seed)
    
    if not os.path.exists(pretrained_emb_path):
        raise FileNotFoundError(
            f"Pretrained embeddings not found at: {pretrained_emb_path}\n"
            f"Please ensure:\n"
            f"1. The path is correct and accessible\n"
            f"2. The file exists in the specified location\n"
            f"3. Check model.trajectory_selector.pretrained_emb_path within hydra config is getting passed correctly\n"
            f"Current working directory: {os.getcwd()}"
        )
    # Load whole embedding weights
    pre_trained_dict = torch.load(pretrained_emb_path, map_location=device, weights_only=True)
    
    if isinstance(pre_trained_dict, dict):
        weights = pre_trained_dict.get('train_embeddings', None)
        traj_data = pre_trained_dict.get('train_trajectories', None)
        indices = pre_trained_dict.get("original_indices", None)
    else:
        raise ValueError("Could not find the state dict in the file")
    
    if weights is None or traj_data is None:
        raise ValueError("Could not find embeddings in the state dict")

    # subset the embeddings
    if subset_percentage > 0:
        num_use = int(len(weights) * subset_percentage)
        indices = torch.randperm(len(weights))[:num_use]
        indices = indices.sort()[0]
        weights = weights[indices]
        traj_data = traj_data[indices]  
    else:
        indices = None
    
    log.info(f"Shape of loaded weights: {weights.shape}")
    log.info(f"Shape of loaded trajectories: {traj_data.shape}")
    log.info(f"Shape of loaded indices: {indices.shape if indices is not None else None}")
    
    return weights, indices, traj_data

if __name__ == "__main__":
    weights, indices, traj_data = load_embeddings("data/embeddings_trajectory_100-spice-slush.pt", subset_percentage=0.10)
    print(weights.shape)
    print(traj_data.shape)