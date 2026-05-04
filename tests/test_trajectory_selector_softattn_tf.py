import pytest
import torch

from src.models.modules.trajectory_selector_softattn_tf import SoftAttentionTrajectorySelector


def test_softattn_tf_accepts_agent_type_and_returns_expected_shapes():
    torch.manual_seed(0)
    batch_size = 2
    n_targets = 3
    n_latent_anchors = 2
    embed_dim = 8
    n_bank = 5
    n_steps = 4

    loaded_embeddings = torch.randn(n_bank, embed_dim)
    traj_tensor = torch.randn(n_bank, n_steps, 2)

    selector = SoftAttentionTrajectorySelector(
        loaded_embeddings=loaded_embeddings,
        traj_tensor=traj_tensor,
        n_latent_anchors=n_latent_anchors,
        hidden_dim=embed_dim,
        selection_mode="no_extras",
        use_straight_through=False,
    )

    valid_mask = torch.ones(batch_size, n_targets, dtype=torch.bool)
    target_emb = torch.randn(batch_size * n_targets, 4, embed_dim)
    target_valid = torch.ones(batch_size * n_targets, 4, dtype=torch.bool)
    agent_type = torch.zeros(batch_size, n_targets, 3)
    agent_type[..., 0] = 1.0

    try:
        outputs = selector(
            valid_mask=valid_mask,
            agent_type=agent_type,
            target_emb=target_emb,
            target_valid=target_valid,
            others_emb=None,
            others_valid=None,
            map_emb=None,
            map_valid=None,
        )
    except Exception as exc:
        pytest.fail(f"unexpected error with agent_type: {exc}")

    (
        selected_trajectories,
        selected_embeddings,
        selected_indices,
        soft_traj,
        adapted_queries,
        adapted_queries_128,
    ) = outputs

    assert selected_trajectories.shape == (
        batch_size * n_targets,
        n_latent_anchors,
        1,
        n_steps,
        2,
    )
    assert selected_embeddings.shape == (
        batch_size * n_targets,
        1,
        n_latent_anchors,
        embed_dim,
    )
    assert selected_indices.shape == (batch_size * n_targets, n_latent_anchors, 1)
    assert soft_traj.shape == (batch_size * n_targets, n_latent_anchors, n_steps, 2)
    assert adapted_queries.shape == (batch_size * n_targets, n_latent_anchors, embed_dim)
    assert adapted_queries_128.shape == (batch_size * n_targets, n_latent_anchors, embed_dim)


def test_softattn_tf_soft_retrieval_uses_soft_weights_for_trajectories():
    torch.manual_seed(0)
    batch_size = 1
    n_targets = 1
    n_latent_anchors = 2
    embed_dim = 8
    n_bank = 3
    n_steps = 5

    base_vec = torch.arange(1, embed_dim + 1, dtype=torch.float32)
    loaded_embeddings = base_vec.repeat(n_bank, 1)
    traj_tensor = torch.stack(
        [torch.full((n_steps, 2), float(k + 1)) for k in range(n_bank)], dim=0
    )

    selector = SoftAttentionTrajectorySelector(
        loaded_embeddings=loaded_embeddings,
        traj_tensor=traj_tensor,
        n_latent_anchors=n_latent_anchors,
        hidden_dim=embed_dim,
        selection_mode="no_extras",
        use_straight_through=False,
    )

    valid_mask = torch.ones(batch_size, n_targets, dtype=torch.bool)
    target_emb = torch.randn(batch_size, n_targets, embed_dim)
    target_valid = torch.ones(batch_size, n_targets, dtype=torch.bool)
    agent_type = torch.zeros(batch_size, n_targets, 3)
    agent_type[..., 0] = 1.0

    (
        selected_trajectories,
        _selected_embeddings,
        selected_indices,
        soft_traj,
        _adapted_queries,
        _adapted_queries_128,
    ) = selector(
        valid_mask=valid_mask,
        agent_type=agent_type,
        target_emb=target_emb,
        target_valid=target_valid,
        others_emb=None,
        others_valid=None,
        map_emb=None,
        map_valid=None,
    )

    expected_soft = traj_tensor.mean(dim=0)
    assert torch.allclose(
        soft_traj, expected_soft.expand_as(soft_traj), atol=1e-5, rtol=0.0
    )
    assert torch.allclose(
        selected_trajectories[:, :, 0],
        expected_soft.expand_as(selected_trajectories[:, :, 0]),
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.equal(selected_indices, torch.zeros_like(selected_indices))
