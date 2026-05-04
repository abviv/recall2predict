import torch
from HPTR.src.data_modules.ac_global import AgentCentricGlobal

class AgentCentricSceneMotion(AgentCentricGlobal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, batch):
        agent_centric_batch = super().forward(batch)
        # the ref_pos and ref_rot are in SDC-centered world coordinates
        agent_centric_batch["input/ref_pos"] = batch["ref/pos"]
        agent_centric_batch["input/ref_rot"] = batch["ref/rot"]

        # Displacement GT for other agents: future positions relative to
        # each agent's last observed (current) position.
        if "gt/other_pos" in agent_centric_batch and "ac/other_pos" in agent_centric_batch:
            other_current_pos = agent_centric_batch["ac/other_pos"][:, :, :, -1, :]  # [B, T, n_others, 2]
            agent_centric_batch["gt/y_disp_others"] = (
                agent_centric_batch["gt/other_pos"] - other_current_pos.unsqueeze(3)
            )

        return agent_centric_batch