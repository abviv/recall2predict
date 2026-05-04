from datetime import datetime
import os
from pathlib import Path
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import logging

log_writer = logging.getLogger(__name__)

def get_wandb_logger(trainer: Trainer):
    """Safely get Weights&Biases logger from Trainer."""
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if hasattr(trainer.logger, "loggers"):
        for logger in trainer.logger.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    return None


class TimedCheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint that organizes checkpoints in time-based directories and integrates with W&B."""
    
    def __init__(
        self, 
        save_only_best=False, 
        root_dir="checkpoints",
        sync_to_wandb=True,
        *args, 
        **kwargs
    ):
        # Create time-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = os.environ.get("WANDB_RUN_ID", "run")
        
        if "dirpath" not in kwargs:
            dirpath = os.path.join(root_dir, f"{timestamp}_{run_id}")
            os.makedirs(dirpath, exist_ok=True)
            kwargs["dirpath"] = dirpath
            
        super().__init__(*args, **kwargs)
        self.save_only_best = save_only_best
        self.sync_to_wandb = sync_to_wandb
        self._logged_model_time = {}
        self.current_score = None

    def save_checkpoint(self, trainer) -> None:
        """Overridden method to save checkpoint and sync with W&B if enabled."""
        super().save_checkpoint(trainer)
        
        if not self.sync_to_wandb:
            return
            
        if not hasattr(self, "_logged_model_time"):
            self._logged_model_time = {}
            
        logger = get_wandb_logger(trainer)
        if logger is None:
            return
            
        if self.current_score is None:
            self.current_score = trainer.callback_metrics.get(self.monitor)
            
        self._scan_and_log_checkpoints(logger)

    @rank_zero_only
    def _scan_and_log_checkpoints(self, wb_logger: WandbLogger) -> None:
        """Scan and log checkpoints to W&B."""
        if self.save_only_best:
            self._log_best_checkpoint(wb_logger)
        else:
            self._log_all_checkpoints(wb_logger)

    def _log_all_checkpoints(self, wb_logger: WandbLogger) -> None:
        """Log all checkpoints to W&B."""
        checkpoints = {
            self.last_model_path: self.current_score,
            self.best_model_path: self.best_model_score,
        }
        checkpoints = sorted(
            (Path(p).stat().st_mtime, p, s)
            for p, s in checkpoints.items()
            if Path(p).is_file()
        )
        checkpoints = [
            c
            for c in checkpoints
            if c[1] not in self._logged_model_time.keys()
            or self._logged_model_time[c[1]] < c[0]
        ]
        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s.item() if hasattr(s, "item") else s,
                "original_filename": Path(p).name,
                "checkpoint_dir": Path(p).parent.name,
                "ModelCheckpoint": {
                    k: getattr(self, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    if hasattr(self, k)
                },
            }
            artifact = wandb.Artifact(
                name=wb_logger.experiment.id, type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == self.best_model_path else ["latest"]
            wb_logger.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t

    def _log_best_checkpoint(self, wb_logger: WandbLogger) -> None:
        """Log only the best checkpoint to W&B."""
        if Path(self.best_model_path).is_file():
            best_model_mtime = Path(self.best_model_path).stat().st_mtime
            # Check if the best model checkpoint is new or has been updated
            if (
                self.best_model_path not in self._logged_model_time
                or self._logged_model_time[self.best_model_path] < best_model_mtime
            ):
                # Attempt to delete the previous best artifact if it exists
                try:
                    api = wandb.Api()
                    runs = api.run(
                        f"{wb_logger.experiment.entity}/{wb_logger.experiment.project}/{wb_logger.experiment.id}"
                    )
                    for artifact in runs.logged_artifacts():
                        if "best" in artifact.aliases:
                            artifact.delete(delete_aliases=True)
                            break
                except Exception:
                    pass
                    
                # Log the best model checkpoint
                metadata = {
                    "score": self.best_model_score.item() if hasattr(self.best_model_score, "item") else self.best_model_score,
                    "original_filename": Path(self.best_model_path).name,
                    "checkpoint_dir": Path(self.best_model_path).parent.name,
                    "ModelCheckpoint": {
                        k: getattr(self, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                            "_every_n_val_epochs",
                        ]
                        if hasattr(self, k)
                    },
                }
                artifact = wandb.Artifact(
                    name=wb_logger.experiment.id, type="model", metadata=metadata
                )
                artifact.add_file(self.best_model_path, name="model.ckpt")
                wb_logger.experiment.log_artifact(artifact, aliases=["best"])
                # Update the log timestamp for this model checkpoint
                self._logged_model_time[self.best_model_path] = best_model_mtime

class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100, log_graph: bool = False):
        self._log = log
        self._log_freq = log_freq
        self._log_graph = log_graph
    
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
        if logger is not None:
            log_writer.info("Watching model with wandb")
            logger.watch(model=trainer.model, log=self._log, log_freq=self._log_freq, log_graph=self._log_graph)
        else:
            log_writer.warning("Wandb logger not found, skipping watch model")