from omegaconf import OmegaConf
import torch

from src.models.ac_model_R2P_gqa import BoundaryAware


def _trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _model_cfg(
    *,
    agent_depth=2,
    lane_depth=2,
    r2p_depth=2,
    anchor_depth=1,
    encoder_ffn=None,
    decoder_depth=1,
    decoder_ffn=None,
):
    return {
        "hidden_dim": 256,
        "agent_attr_dim": 65,
        "map_attr_dim": 35,
        "tl_attr_dim": 62,
        "n_pl_node": 20,
        "use_current_tl": False,
        "pl_aggr": False,
        "n_step_hist": 50,
        "n_decoders": 1,
        "use_encoder": True,
        "tf_cfg": OmegaConf.create(
            {
                "n_head": 8,
                "dropout_p": 0.25,
                "norm_first": True,
                "bias": False,
            }
        ),
        "local_encoder": OmegaConf.create(
            {
                "add_learned_pe": True,
                "use_point_net": True,
                "n_layer_mlp": 3,
                "mlp_cfg": {
                    "end_layer_activation": False,
                    "use_layernorm": True,
                    "use_batchnorm": False,
                    "dropout_p": 0.15,
                },
            }
        ),
        "motion_decoder": OmegaConf.create(
            {
                "n_pred": 6,
                "mlp_head_num_heads": 1,
                "n_refinement_layer": decoder_depth,
                "use_offset_prediction": True,
                "offset_dropout_p": 0.2,
                "use_anchors_as_queries": True,
                "tf_cfg": {
                    "d_feedforward": decoder_ffn,
                },
                "mlp_head_cfg": {
                    "predictions": ["pos", "cov3", "vel", "yaw_bbox"],
                    "use_agent_type": False,
                    "n_step_future": 60,
                    "out_mlp_layernorm": True,
                    "out_mlp_batchnorm": False,
                },
            }
        ),
        "early_fusion_encoder": OmegaConf.create(
            {
                "agent_attn_depth": agent_depth,
                "lane_attn_depth": lane_depth,
                "r2p_attn_depth": r2p_depth,
                "anchor_attn_depth": anchor_depth,
                "d_feedforward": encoder_ffn,
            }
        ),
        "trajectory_selector": OmegaConf.create(
            {
                "use_traj_projection": True,
                "n_latent_anchors": 6,
                "hidden_dim": 256,
                "use_layer": "softattn",
                "selection_mode": "full",
                "retrieval_mode": "endpoint_rerank",
                "endpoint_topM": 6,
                "use_straight_through": False,
                "use_gqa": False,
                "use_gating": True,
                "contexts": {
                    "use_target": True,
                    "use_other": True,
                    "use_map": False,
                },
            }
        ),
        "encoder_init_from": {"activate": False},
        "freeze_modules": [],
    }


def _build_model(**cfg_overrides):
    torch.manual_seed(0)
    loaded_embeddings = torch.randn(128, 128)
    traj_tensor = torch.randn(128, 60, 2)
    return BoundaryAware(
        **_model_cfg(**cfg_overrides),
        loaded_embeddings=loaded_embeddings,
        traj_tensor=traj_tensor,
    )


def test_scaled_core_config_reaches_approx_16m_and_wires_ffn_knobs():
    model = _build_model(
        agent_depth=3,
        lane_depth=3,
        r2p_depth=3,
        anchor_depth=2,
        encoder_ffn=1024,
        decoder_depth=2,
        decoder_ffn=1024,
    )

    assert model.agent_attn_depth == 3
    assert model.lane_attn_depth == 3
    assert model.r2p_attn_depth == 3
    assert model.anchor_attn_depth == 2
    assert model.decoder_depth == 2

    assert model.agent_self_attn[0].mlp.fc1.out_features == 1024
    assert model.lane_self_attn[0].mlp.fc1.out_features == 1024
    assert model.r2p_fusion_attn[0].mlp.fc1.out_features == 1024
    assert model.anchor_self_attn[0].mlp.fc1.out_features == 1024
    assert model.decoder.refinement_layers[0]["agent_ca"].mlp.fc1.out_features == 1024
    assert model.decoder.refinement_layers[0]["lane_ca"].mlp.fc1.out_features == 1024

    total_params = _trainable_params(model)
    selector_params = _trainable_params(model.trajectory_selector)
    assert 16_000_000 <= total_params <= 17_000_000
    assert total_params - selector_params > 14_000_000


def test_ffn_width_increases_core_block_and_decoder_params():
    base = _build_model(encoder_ffn=256, decoder_ffn=256)
    widened = _build_model(encoder_ffn=1024, decoder_ffn=1024)

    assert _trainable_params(widened.agent_self_attn[0]) > _trainable_params(
        base.agent_self_attn[0]
    )
    assert _trainable_params(widened.decoder) > _trainable_params(base.decoder)
