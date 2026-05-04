from typing import Any, Dict, Type

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def collect_test_time_load_overrides(cfg: DictConfig) -> Dict[str, Any]:
    """Collect safe inference-time overrides to apply when loading a checkpoint.

    Two classes of keys are handled:
      * Sub-keys of ``cfg.model`` (legacy: many fields used to live under the
        `model` group, e.g. submission configs, plot toggles, post-processing).
      * Top-level keys that are *siblings* of ``model`` in the FutureMotion
        signature, currently just ``pre_processing``. These are picked up from
        ``cfg`` directly so users can override e.g.
        ``pre_processing.agent_centric.n_target`` for inference-only profiling
        (model weights are agent-centric and therefore T-invariant).
    """
    load_overrides: Dict[str, Any] = {}
    model_cfg = cfg.get("model")
    if not model_cfg:
        return load_overrides

    model_override_keys = [
        "sub_av2",
        "sub_womd",
        "post_processing",
        "n_video_batch",
        "inference_repeat_n",
        "inference_cache_map",
        "plot_motion",
        "plot_motion_focal_track",
        "plot_probmap",
        "plot_anchor_selection",
        "plot_endpoints",
    ]
    for key in model_override_keys:
        value = model_cfg.get(key)
        if value is not None:
            load_overrides[key] = value

    top_level_override_keys = ["pre_processing"]
    for key in top_level_override_keys:
        value = cfg.get(key)
        if value is not None:
            load_overrides[key] = value

    return load_overrides


def load_model_from_checkpoint_for_inference(
    ckpt_path: str,
    cfg: DictConfig,
    model_cls: Type[Any],
    ckpt_strict: bool = True,
) -> Any:
    """Load a model from checkpoint using the checkpoint config plus safe inference overrides."""
    override_anchor = cfg.get("override_anchor_path") or (
        cfg.get("model") and cfg.model.get("pretrained_emb_path")
    )
    load_kwargs: Dict[str, Any] = {
        "checkpoint_path": ckpt_path,
        "strict": ckpt_strict,
    }
    load_kwargs.update(collect_test_time_load_overrides(cfg))

    if override_anchor:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hparams = checkpoint.get("hyper_parameters", {})
        model_hparams = hparams.get("model")

        load_kwargs["pretrained_emb_path"] = override_anchor
        if model_hparams is not None:
            if not OmegaConf.is_config(model_hparams):
                model_hparams = OmegaConf.create(model_hparams)
            with open_dict(model_hparams):
                if not hasattr(model_hparams, "trajectory_selector"):
                    model_hparams.trajectory_selector = OmegaConf.create()
            load_kwargs["model"] = model_hparams

        log.info(
            f"Loading from checkpoint with pretrained_emb_path override: {override_anchor}"
        )

    if len(load_kwargs) > 2:
        override_keys = sorted(
            k for k in load_kwargs.keys() if k not in {"checkpoint_path", "strict"}
        )
        log.info(f"Applying checkpoint load overrides: {override_keys}")

    return model_cls.load_from_checkpoint(**load_kwargs)
