from typing import Any, Dict
from pathlib import Path
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionally saves:
        - Number of model parameters
        - Full Hydra config folder (.hydra) as wandb artifact
    """
    try:
        import sys
        sys.executable = sys.orig_executable if hasattr(sys, 'orig_executable') else '/usr/bin/python3'
        log.info(f"Current sys.executable path: {sys.executable}")
    except Exception as e:
        log.error(f"Error in logging hyperparameters: {e}")

    hparams = {}
    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.loggers:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams["data"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
    
    # Upload .hydra folder to wandb as artifact
    _upload_hydra_config_to_wandb(trainer, object_dict["cfg"])


def _upload_hydra_config_to_wandb(trainer, cfg) -> None:
    """Upload the full .hydra folder as a wandb artifact."""
    try:
        # Find WandbLogger
        wandb_logger = None
        if isinstance(trainer.logger, WandbLogger):
            wandb_logger = trainer.logger
        elif hasattr(trainer.logger, 'loggers'):
            for logger in trainer.logger:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break
        
        if wandb_logger is None:
            log.debug("WandbLogger not found, skipping .hydra upload")
            return
        
        # Get Hydra output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = Path(hydra_cfg.runtime.output_dir)
        hydra_dir = hydra_output_dir / ".hydra"
        
        if not hydra_dir.exists():
            log.warning(f".hydra folder not found at {hydra_dir}")
            return
        
        # Create and log artifact
        artifact = wandb.Artifact(
            name=f"hydra-config-{wandb_logger.experiment.id}",
            type="config",
            description="Full Hydra configuration folder"
        )
        artifact.add_dir(str(hydra_dir), name=".hydra")
        wandb_logger.experiment.log_artifact(artifact)
        
        log.info(f"✓ Uploaded .hydra folder to wandb from {hydra_dir}")
        
    except Exception as e:
        log.warning(f"Failed to upload .hydra folder to wandb: {e}")