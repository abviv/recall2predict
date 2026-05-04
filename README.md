# Recall2Predict

Recall2Predict is an AV2 motion forecasting training pipeline built around
trajectory-bank retrieval and a DETR-style refinement decoder. This public
release is intentionally focused on the R2P AV2 Straight-Through Estimator based GQA training
path.

## Repository Scope

This repository contains the code and configs required to run:

```bash
python src/train.py experiment=ablations/r2p_av2_st_gqa \
  datamodule.batch_size=8 \
  model.pretrained_emb_path=data/av2/processed_dataset/pre_trained_embeddings_new/av2_bucketed_traj_embeddings_128x64_shuffled.pt \
  trainer.max_epochs=20 \
  trainer.limit_train_batches=0.001 \
  trainer.limit_val_batches=0.001 \
  datamodule.data_dir=data/av2/processed_dataset \
  logger.wandb.project=debug \
  trainer.log_every_n_steps=10
```

The bundled trajectory embedding bank is:

```text
data/av2/processed_dataset/pre_trained_embeddings_new/av2_bucketed_traj_embeddings_128x64_shuffled.pt
```

## Dataset Layout

Dataset HDF5 files are not bundled. The public repo keeps the same
project-root layout used by the development repo.

For AV2:

```text
data/
  av2/
    processed_dataset/
      training_new.h5
      validation_new.h5
      testing_new.h5
      pre_trained_embeddings_new/
        av2_bucketed_traj_embeddings_128x64_shuffled.pt
```

The selected config is `configs/datamodule/h5_av2_noraster.yaml`, so raster
HDF5 files are not required for the default Recall2Predict command.

You can either copy the AV2 HDF5 files into the existing directory:

```bash
mkdir -p data/av2/processed_dataset
cp /path/to/av2_processed/{training_new.h5,validation_new.h5,testing_new.h5} \
  data/av2/processed_dataset/
```

Or mirror the original development setup with a symlink:

```bash
mkdir -p data/av2
ln -s /absolute/path/to/av2_processed data/av2/processed_dataset
```

If you use the symlink option, make sure the symlink target also contains
`pre_trained_embeddings_new/av2_bucketed_traj_embeddings_128x64_shuffled.pt`,
or pass `model.pretrained_emb_path=/path/to/the/embedding.pt` on the command
line.

For WOMD configs:

```text
data/
  womd/
    processed_dataset/
      training.h5
      validation.h5
      testing.h5
    pre_trained_embeddings/
      womd_sdc_bucketed_traj_embeddings_96x96_shuffled-fixed-float32.pt
```

The included WOMD experiment configs use `configs/datamodule/h5_womd_noraster.yaml`,
so raster HDF5 files are not required unless you switch to `h5_womd`.

You can copy the WOMD HDF5 files into the repo layout:

```bash
mkdir -p data/womd/processed_dataset data/womd/pre_trained_embeddings
cp /path/to/womd_processed/{training.h5,validation.h5,testing.h5} \
  data/womd/processed_dataset/
cp /path/to/womd_embeddings/womd_sdc_bucketed_traj_embeddings_96x96_shuffled-fixed-float32.pt \
  data/womd/pre_trained_embeddings/
```

Or mirror the development setup with symlinks:

```bash
mkdir -p data/womd
ln -s /absolute/path/to/womd_processed data/womd/processed_dataset
ln -s /absolute/path/to/womd_pre_trained_embeddings data/womd/pre_trained_embeddings
```

The WOMD configs can also be pointed elsewhere without editing YAML:

```bash
export R2P_WOMD_DATA_DIR=/absolute/path/to/womd_processed
export R2P_WOMD_PRETRAINED_EMB_PATH=/absolute/path/to/womd_embedding.pt
```

## Setup

Clone the repository with its submodule:

```bash
git clone --recursive git@github.com:abviv/recall2predict.git
cd recall2predict
```

If you already cloned without `--recursive`, initialize the required
`layers_in_my_way` submodule before running tests or training:

```bash
git submodule update --init --recursive
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate recall2predict
```

Install the package in editable mode if you want console entrypoints:

```bash
pip install -e .
```

## Verify Configuration

You can inspect the fully composed Hydra config without starting training:

```bash
python src/train.py experiment=ablations/r2p_av2_st_gqa \
  datamodule.batch_size=8 \
  model.pretrained_emb_path=data/av2/processed_dataset/pre_trained_embeddings_new/av2_bucketed_traj_embeddings_128x64_shuffled.pt \
  trainer.max_epochs=20 \
  trainer.limit_train_batches=0.001 \
  trainer.limit_val_batches=0.001 \
  datamodule.data_dir=data/av2/processed_dataset \
  logger.wandb.project=debug \
  trainer.log_every_n_steps=10 \
  --cfg job
```

## Tests

```bash
python -m pytest -q tests/test_ac_model_R2P_gqa_scaling.py tests/test_trajectory_selector_softattn_tf.py
```

## Third-Party Code

This release includes selected HPTR-derived utilities and the
`src/layers_in_my_way` Git submodule needed by the R2P path. See
`THIRD_PARTY_NOTICES.md` for license details. The root MIT license applies to
the original Recall2Predict code only.
