<p align="center">
  <h1 align="center">Recall to Predict</h1>
  <h3 align="center">Grounding Motion Forecasting in Interpretable Motion Bank</h3>
  <p align="center">
    <a href="https://github.com/abviv">Abhishek Vivekanandan</a><sup>1,2</sup> &nbsp;&middot;&nbsp;
    Ahmed Abouelazm<sup>1</sup> &nbsp;&middot;&nbsp;
    J. Marius Zöllner<sup>1,2</sup>
  </p>
  <p align="center">
    <sup>1</sup>FZI Forschungszentrum Informatik &nbsp;&nbsp;
    <sup>2</sup>Karlsruhe Institute of Technology (KIT)
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2605.01393"><img alt="Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv"></a>
    <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="https://www.python.org/downloads/release/python-380/"><img alt="Python 3.8+" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c?logo=pytorch&logoColor=white"></a>
  </p>
</p>

---

## Overview

**Recall to Predict (R2P)** is an end-to-end differentiable motion forecasting framework that grounds predictions in a comprehensive *motion bank* — a structured embedding space of physically realizable trajectories constructed via contrastive learning.

Rather than regressing paths from opaque latent queries, R2P dynamically retrieves explicit motion priors using a novel **Anchor Retrieval Layer**, preserving multi-modal diversity while exposing the model's intermediate reasoning.

<p align="center">
  <img src="docs/architecture.png" alt="R2P Architecture" width="90%">
</p>

### Key Contributions

| Component | Description |
|-----------|-------------|
| **Motion Bank** | Pre-trained contrastive embedding space mapping physically realizable trajectories to structured latent vectors |
| **Anchor Retrieval Layer** | Differentiable attention-based retrieval with Dual-Level Gated Cross-Attention and orthogonally initialized queries |
| **Straight-Through Gumbel-Softmax** | Discrete trajectory selection in the forward pass with continuous gradient flow in the backward pass |
| **DETR-style Decoder** | Iterative refinement decoder using retrieved anchors as interpretable queries |
| **Multi-Objective Training** | WTA kinematic GMM + soft-min endpoint loss + latent diversity penalty |


---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Recall to Predict (R2P)                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐  │
│  │ Input Proj.  │     │  Anchor Retrieval  │     │  Factorized Enc. │  │
│  │  (PointNet)  │────▶│      Layer         │     │  (Self-Attn)     │  │
│  └──────────────┘     │                    │     └────────┬─────────┘  │
│                       │  Q_base (orthog.)  │              │            │
│                       │       ↓            │              │            │
│                       │  Dual-Level Gated  │              │            │
│                       │  Cross-Attention   │              │            │
│                       │       ↓            │              │            │
│                       │  Cosine Similarity │              │            │
│                       │  w/ Motion Bank    │              │            │
│                       │       ↓            │              │            │
│                       │  ST Gumbel-Softmax │              │            │
│                       │       ↓            │              │            │
│                       │  Anchor Tokens     │              │            │
│                       └────────┬───────────┘              │            │
│                                │                          │            │
│                                ▼                          ▼            │
│                       ┌────────────────────────────────────┐           │
│                       │       DETR-style Decoder           │           │
│                       │  (Cross-Attn: Target + Env)        │           │
│                       └────────────────┬───────────────────┘           │
│                                        │                               │
│                        ┌───────────────┼───────────────┐               │
│                        ▼               ▼               ▼               │
│                   ┌─────────┐   ┌───────────┐   ┌──────────┐           │
│                   │ Kin. GMM│   │Confidence │   │  Offset  │           │
│                   │  Head   │   │   Head    │   │   Head   │           │
│                   └─────────┘   └───────────┘   └──────────┘           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- Conda (recommended)

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/abviv/recall2predict.git
cd recall2predict

# If already cloned without --recursive
git submodule update --init --recursive

# Create environment
conda env create -f environment.yml
conda activate recall2predict

# Install in editable mode (optional)
pip install -e .
```

---

## Dataset Setup

<details>
<summary><b>Argoverse 2 (AV2)</b></summary>

```
data/
  av2/
    processed_dataset/
      training_new.h5
      validation_new.h5
      testing_new.h5
      pre_trained_embeddings_new/
        av2_bucketed_traj_embeddings_128x64_shuffled.pt
```

**Option A — Copy files:**
```bash
mkdir -p data/av2/processed_dataset
cp /path/to/av2_processed/{training_new.h5,validation_new.h5,testing_new.h5} \
  data/av2/processed_dataset/
```

**Option B — Symlink:**
```bash
mkdir -p data/av2
ln -s /absolute/path/to/av2_processed data/av2/processed_dataset
```

</details>

<details>
<summary><b>Waymo Open Motion Dataset (WOMD)</b></summary>

```
data/
  womd/
    processed_dataset/
      training.h5
      validation.h5
      testing.h5
    pre_trained_embeddings/
      womd_sdc_bucketed_traj_embeddings_96x96_shuffled-fixed-float32.pt
```

**Option A — Copy files:**
```bash
mkdir -p data/womd/processed_dataset data/womd/pre_trained_embeddings
cp /path/to/womd_processed/{training.h5,validation.h5,testing.h5} \
  data/womd/processed_dataset/
cp /path/to/womd_embeddings/womd_sdc_bucketed_traj_embeddings_96x96_shuffled-fixed-float32.pt \
  data/womd/pre_trained_embeddings/
```

**Option B — Symlink:**
```bash
mkdir -p data/womd
ln -s /absolute/path/to/womd_processed data/womd/processed_dataset
ln -s /absolute/path/to/womd_pre_trained_embeddings data/womd/pre_trained_embeddings
```

**Option C — Environment variables:**
```bash
export R2P_WOMD_DATA_DIR=/absolute/path/to/womd_processed
export R2P_WOMD_PRETRAINED_EMB_PATH=/absolute/path/to/womd_embedding.pt
```

</details>

---

## Training

### AV2 — Straight-Through GQA

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

### Verify Configuration (dry run)

```bash
python src/train.py experiment=ablations/r2p_av2_st_gqa \
  datamodule.batch_size=8 \
  model.pretrained_emb_path=data/av2/processed_dataset/pre_trained_embeddings_new/av2_bucketed_traj_embeddings_128x64_shuffled.pt \
  trainer.max_epochs=20 \
  datamodule.data_dir=data/av2/processed_dataset \
  --cfg job
```

---

## Tests

```bash
python -m pytest -q tests/test_ac_model_R2P_gqa_scaling.py tests/test_trajectory_selector_softattn_tf.py
```

---

## Method Summary

<details>
<summary><b>Pre-trained Motion Bank</b></summary>

A structured embedding space of prototypical driving behaviors, built via contrastive learning. Each trajectory maps 1-to-1 to a semantically meaningful latent vector. The bank provides physically realizable motion priors that eliminate the need to regress paths from scratch.

</details>

<details>
<summary><b>Anchor Retrieval Layer</b></summary>

Orthogonally initialized queries attend to heterogeneous scene context via **Dual-Level Gated Cross-Attention**:
- **Micro-Level:** Per-query sigmoid gates control context absorption
- **Macro-Level:** Global softmax with learnable null-sink routes attention across modalities

Discrete selection is performed via Straight-Through Gumbel-Softmax — hard argmax forward, soft gradient backward.

</details>

<details>
<summary><b>Iterative Refinement Decoder</b></summary>

A DETR-style decoder that uses retrieved anchor tokens directly as queries (not learned from scratch). The decoder attends sequentially to the focal agent token and environment context. An offset head on the *pre-decoder* anchors prevents the decoder from compensating for poor retrieval.

</details>

<details>
<summary><b>Training Objectives</b></summary>

$$\mathcal{L}_{\text{total}} = \lambda_{\text{motion}}\,\mathcal{L}_{\text{motion}} + 0.1\,\mathcal{L}_{\text{endpoint}} + \mathcal{L}_{\text{div}}$$

- **WTA Kinematic Loss:** NLL + velocity Huber + heading cosine + confidence CE (only winning mode)
- **Dual-Objective Endpoint Loss:** Soft-min weighted Huber on anchor + offset endpoints
- **Latent Diversity Loss:** Frobenius-norm penalty on query cosine similarity vs. identity

</details>

---

## Project Structure

```
recall2predict/
├── configs/                  # Hydra experiment configs
│   ├── datamodule/           # Dataset configurations
│   └── experiment/           # Training experiments
├── src/
│   ├── train.py              # Main training entrypoint
│   ├── models/               # Model architectures
│   ├── datamodules/          # Data loading & preprocessing
│   └── layers_in_my_way/     # Git submodule (third-party layers)
├── tests/                    # Unit tests
├── data/                     # Dataset root (not tracked)
└── environment.yml           # Conda environment specification
```

---

## Citation

```bibtex
@inproceedings{vivekanandan2026recall,
  title={Recall to Predict: Grounding Motion Forecasting in Interpretable Motion Bank},
  author={Vivekanandan, Abhishek and Abouelazm, Ahmed and Z{\"o}llner, J. Marius},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2026}
}
```

---

## Acknowledgements

This work was built upon the wonderful contributions of [HPTR](https://github.com/zhejz/HPTR), [Future-Motion](https://github.com/kit-mrt/future-motion)

