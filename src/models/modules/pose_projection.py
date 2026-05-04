"""
Pose-based positional projection utilities.
"""

from typing import Tuple
import torch
from torch import nn, Tensor


class PoseProjection(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_step_hist: int,
        n_pl_node: int,
        pl_aggr: bool,
        use_point_net: bool,
    ) -> None:
        super().__init__()
        self.n_step_hist = n_step_hist
        self.n_pl_node = n_pl_node
        self.pl_aggr = pl_aggr
        self.use_point_net = use_point_net

        # EMP-style positional embedding from [x, y, cos, sin]
        self.pos_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        target_attr: Tensor,
        other_attr: Tensor,
        map_attr: Tensor,
        target_valid: Tensor,
        other_valid: Tensor,
        map_valid: Tensor,
        agent_token_count: int,
        lane_token_count: int,
    ) -> Tuple[Tensor, Tensor]:
        pos_feat_all, pos_valid_all, n_agent_tokens = self._build_pos_features(
            target_attr=target_attr,
            other_attr=other_attr,
            map_attr=map_attr,
            target_valid=target_valid,
            other_valid=other_valid,
            map_valid=map_valid,
        )

        expected_tokens = agent_token_count + lane_token_count
        if pos_feat_all.shape[1] != expected_tokens or n_agent_tokens != agent_token_count:
            raise ValueError(
                "Positional embedding token count mismatch: "
                f"pos_feat={pos_feat_all.shape[1]} (agent={n_agent_tokens}) vs "
                f"expected={expected_tokens} (agent={agent_token_count}, lane={lane_token_count})."
            )

        pos_embed_all = self.pos_embed(pos_feat_all)
        pos_embed_all = pos_embed_all.masked_fill(~pos_valid_all.unsqueeze(-1), 0.0)

        pos_embed_agent = pos_embed_all[:, :agent_token_count, :]
        pos_embed_lane = pos_embed_all[:, agent_token_count:, :]
        return pos_embed_agent, pos_embed_lane

    def _normalize_dir(self, vec: Tensor, eps: float = 1e-6) -> Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp(min=eps)
        return vec / norm

    def _map_center_dir_from_attr(
        self,
        map_attr: Tensor,
        map_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            map_center: [B, T, n_map, 2]
            map_dir:    [B, T, n_map, 2] (unit)
        """
        if map_attr.dim() == 5:
            # [B, T, n_map, n_pl_node, attr_dim]
            map_xy = map_attr[..., :2]
            map_dir = map_attr[..., 2:4]
            node_valid = map_valid
        else:
            # [B, T, n_map, attr_dim] (pl_aggr=True)
            pose_len = self.n_pl_node * 4
            if map_attr.shape[-1] >= pose_len:
                pose = map_attr[..., :pose_len].view(*map_attr.shape[:-1], self.n_pl_node, 4)
                map_xy = pose[..., :2]
                map_dir = pose[..., 2:4]
                node_valid = map_valid.unsqueeze(-1).expand_as(map_xy[..., 0])
            else:
                map_xy = map_attr[..., :2]
                map_dir = map_attr[..., 2:4]
                return map_xy, self._normalize_dir(map_dir)

        node_valid_f = node_valid.float().unsqueeze(-1)
        denom = node_valid_f.sum(-2).clamp(min=1.0)
        map_center = (map_xy * node_valid_f).sum(-2) / denom
        map_dir = self._normalize_dir((map_dir * node_valid_f).sum(-2))
        return map_center, map_dir

    def _build_pos_features(
        self,
        target_attr: Tensor,
        other_attr: Tensor,
        map_attr: Tensor,
        target_valid: Tensor,
        other_valid: Tensor,
        map_valid: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        if target_attr.shape[-1] < 4 or map_attr.shape[-1] < 4:
            raise ValueError("Positional encoding requires attr dim >= 4.")

        if self.pl_aggr:
            pose_dim = 4
            pose_len = self.n_step_hist * pose_dim

            target_pose = target_attr[..., :pose_len]
            if target_pose.shape[-1] >= pose_len:
                target_pose = target_pose.view(*target_attr.shape[:-1], self.n_step_hist, pose_dim)
                target_xy = target_pose[..., -1, :2]
                target_dir = target_pose[..., -1, 2:4]
            else:
                target_xy = target_attr[..., :2]
                target_dir = target_attr[..., 2:4]

            other_pose = other_attr[..., :pose_len]
            if other_pose.shape[-1] >= pose_len:
                other_pose = other_pose.view(*other_attr.shape[:-1], self.n_step_hist, pose_dim)
                other_xy = other_pose[..., -1, :2]
                other_dir = other_pose[..., -1, 2:4]
            else:
                other_xy = other_attr[..., :2]
                other_dir = other_attr[..., 2:4]

            map_center, map_dir = self._map_center_dir_from_attr(map_attr, map_valid)

            agent_feat = torch.cat(
                [
                    torch.cat([target_xy, target_dir], dim=-1).unsqueeze(2),
                    torch.cat([other_xy, other_dir], dim=-1),
                ],
                dim=2,
            ).flatten(0, 1)
            map_feat = torch.cat([map_center, map_dir], dim=-1).flatten(0, 1)

            agent_valid = torch.cat(
                [target_valid.unsqueeze(2), other_valid],
                dim=2,
            ).flatten(0, 1)
            map_valid_flat = map_valid.flatten(0, 1)

            pos_feat_all = torch.cat([agent_feat, map_feat], dim=1)
            pos_valid_all = torch.cat([agent_valid, map_valid_flat], dim=1)
            n_agent_tokens = agent_feat.shape[1]
            return pos_feat_all, pos_valid_all, n_agent_tokens

        if self.use_point_net:
            # Token per agent/polyline; use last history step for agents.
            target_xy = target_attr[..., self.n_step_hist - 1, :2]
            target_dir = target_attr[..., self.n_step_hist - 1, 2:4]
            other_xy = other_attr[..., self.n_step_hist - 1, :2]
            other_dir = other_attr[..., self.n_step_hist - 1, 2:4]

            map_center, map_dir = self._map_center_dir_from_attr(map_attr, map_valid)

            agent_feat = torch.cat(
                [
                    torch.cat([target_xy, target_dir], dim=-1).unsqueeze(2),
                    torch.cat([other_xy, other_dir], dim=-1),
                ],
                dim=2,
            ).flatten(0, 1)
            map_feat = torch.cat([map_center, map_dir], dim=-1).flatten(0, 1)

            agent_valid = torch.cat(
                [target_valid.any(-1).unsqueeze(2), other_valid.any(-1)],
                dim=2,
            ).flatten(0, 1)
            map_valid_flat = map_valid.any(-1).flatten(0, 1)

            pos_feat_all = torch.cat([agent_feat, map_feat], dim=1)
            pos_valid_all = torch.cat([agent_valid, map_valid_flat], dim=1)
            n_agent_tokens = agent_feat.shape[1]
            return pos_feat_all, pos_valid_all, n_agent_tokens

        # Token per history step / polyline node.
        target_feat_flat = target_attr[..., :4].flatten(0, 1)
        other_feat_flat = other_attr[..., :4].flatten(0, 1).flatten(1, 2)
        map_feat_flat = map_attr[..., :4].flatten(0, 1).flatten(1, 2)

        target_valid_flat = target_valid.flatten(0, 1)
        other_valid_flat = other_valid.flatten(0, 1).flatten(1, 2)
        map_valid_flat = map_valid.flatten(0, 1).flatten(1, 2)

        pos_feat_all = torch.cat([target_feat_flat, other_feat_flat, map_feat_flat], dim=1)
        pos_valid_all = torch.cat([target_valid_flat, other_valid_flat, map_valid_flat], dim=1)
        n_agent_tokens = target_feat_flat.shape[1] + other_feat_flat.shape[1]

        return pos_feat_all, pos_valid_all, n_agent_tokens
