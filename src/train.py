from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch.version
import torch._dynamo as dynamo

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (this way all filepaths are the same no matter where you run the code)
#       (which is used as a base for paths in "configs/paths/default.yaml")
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
import matplotlib.pyplot as plt
from src.main import FutureMotion
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.checkpoint_loading import (
    collect_test_time_load_overrides as _collect_test_time_load_overrides_impl,
    load_model_from_checkpoint_for_inference,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _collect_test_time_load_overrides(cfg: DictConfig) -> Dict[str, Any]:
    return _collect_test_time_load_overrides_impl(cfg)


def _load_model_from_checkpoint_for_test(
    ckpt_path: str,
    cfg: DictConfig,
    ckpt_strict: bool = True,
) -> LightningModule:
    return load_model_from_checkpoint_for_inference(
        ckpt_path=ckpt_path,
        cfg=cfg,
        model_cls=FutureMotion,
        ckpt_strict=ckpt_strict,
    )

plt.switch_backend('agg')

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.

    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # When running test-only with a checkpoint, load the model from the checkpoint's saved
    # config (like viz_infer/eval). Otherwise we'd use the current Hydra config, which may
    # differ from the checkpoint's architecture (e.g. different n_refinement_layer).
    test_only_with_ckpt = (
        not cfg.get("train", True)
        and not cfg.get("validate", True)
        and cfg.get("test", False)
        and cfg.get("ckpt_path")
    )
    if test_only_with_ckpt:
        log.info(f"Loading model from checkpoint (using checkpoint's config): {cfg.ckpt_path}")
        model = _load_model_from_checkpoint_for_test(
            ckpt_path=cfg.ckpt_path,
            cfg=cfg,
            ckpt_strict=cfg.get("ckpt_strict", True),
        )
    else:
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            data_size=datamodule.tensor_size_train,
            _recursive_=False,
        )

    log.info("Instantiating callbacks...")
    # cfg.get("Would go into the folder config/ and will look for sub-dir callbacks/ and will pick the yaml inside")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, fast_dev_run=False)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("use_compile", False):
        dynamo.explain(model.model)
        if hasattr(model, "model"):
            model.model = torch.compile(model.model, mode='reduce-overhead', fullgraph=False)
            log.info("Inner model is being compiled with torch.compile. Whooohooo! \U0001F973 \U0001F973")
    
    if cfg.get("train", True):
        ckpt_path_to_load = cfg.get("ckpt_path")  # Use existing explicit ckpt_path if provided
        
        if ckpt_path_to_load:
            log.info(f"Starting training and attempting to load from checkpoint: {ckpt_path_to_load} \U0001F638 \U0001F638")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path_to_load)
        else:
            log.info("Starting training from scratch (no checkpoint specified or found). \U0001F600 \U0001F600")
            trainer.fit(model=model, datamodule=datamodule)
    
    train_metrics = trainer.callback_metrics

    val_metrics: Dict[str, Any] = {}
    if cfg.get("validate", True):
        log.info("Starting final validation!")
        trainer.validate(model=model, datamodule=datamodule)
        val_metrics = trainer.callback_metrics

    if cfg.get("test", False):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using `ckpt_path` from config...")
            ckpt_path = cfg.get("ckpt_path")

        if not ckpt_path:
            log.warning("No checkpoint path found! Using current weights for testing...")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
        else:
            # When model was loaded from checkpoint (test-only mode), it already has the
            # correct weights; pass ckpt_path=None to avoid redundant load.
            test_ckpt_path = None if test_only_with_ckpt else ckpt_path
            trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **val_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, 
        metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    log.info(f"Training and Validation completed! Metric value: {metric_value}")
    return metric_value

if __name__ == "__main__":
    main()
