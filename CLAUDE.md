# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Generative Medical Segmentation (GMS, AAAI 2025). The segmentation pipeline uses a frozen Stable Diffusion 2.0 VAE as a tokenizer for both images and masks, and trains a small CNN (`ResAttnUNet_DS`) that learns the image-latent → mask-latent mapping. Predicted masks are reconstructed from latents via the frozen VAE decoder.

## Environment setup

```
conda env create -f environment.yaml
conda activate GMS
# or
python3 -m venv GMS && source GMS/bin/activate && pip install -r requirements.txt
```

Before running anything, you must place the SD VAE checkpoint at `SD-VAE-weights/768-v-ema-first-stage-VAE.ckpt` (see README for download link). This path is hard-coded in `train.py` and `valid.py`.

## Common commands

Training and validation are config-driven; there are no test, lint, or build steps.

```
# Train all 5 datasets sequentially
sh train.sh

# Train a single dataset (preferred when iterating)
python train.py --config ./configs/busi_train.yaml

# Validate all 5 datasets (writes predicted masks + DSC/IoU CSV)
sh valid.sh

# Validate a single dataset
python valid.py --config ./configs/busi_valid.yaml

# Compare GMS's VAE weights against the official SD 2.0 VAE
python vae_comparison.py
```

GPUs are selected per-config via the `GPUs:` list (sets `CUDA_VISIBLE_DEVICES`). Hyperparameters (batch size, lr, epochs, channel multipliers) live in `configs/<dataset>_train.yaml` — the paper used batch size 8 on an A100 40G; reduce `batch_size` on OOM.

## Architecture

The training/inference loop is the same shape in `train.py` and `valid.py`:

1. Image and mask are independently encoded by the **frozen SD VAE** (`networks/models/autoencoder.py::AutoencoderKL`) into 4-channel latents at 1/8 spatial resolution. Inputs are normalized to `[-1, 1]` before encoding; the encoder posterior is a `DiagonalGaussianDistribution` and we use `mu * scale_factor` (0.18215) as the latent.
2. The **latent mapping model** `ResAttnUNet_DS` (`networks/latent_mapping_model.py`) is a U-Net **without down/up-sampling** — it preserves spatial resolution to avoid information loss in latent space. Channels: `in_channel=4`, `out_channels=4`, `ch=32`, `ch_mult=[1,2,4,4]`. The forward returns a dict with deep-supervision outputs `level3/level2/level1/out`; training supervises `level2/level1/out`, inference uses only `out`.
3. Loss is a weighted sum of (a) MSE between predicted and target mask latents and (b) Dice loss between the **VAE-decoded** predicted mask and the ground-truth mask in image space. Both `w_rec` and `w_dice` default to 1. During training, the image latent is stochastically perturbed by its own predicted std with 50% probability (a VAE-style reparam augmentation).
4. The frozen VAE decoder reconstructs masks back to RGB; `vae_decode` averages channels and clamps to `[0, 1]` to produce a single-channel mask.

The only learnable component is `ResAttnUNet_DS` — the VAE is frozen via `vae_model.freeze()` and never updated. This is the core architectural claim of the paper.

## Data layout

`data/image_dataset.py::Image_Dataset` reads paired PNGs from `Dataset/<dataset>/{images,masks}/` and a `<dataset>_train_test_names.pkl` that holds `{'train': {'name_list': [...]}, 'test': {'name_list': [...]}}`. All inputs are resized to 224×224. Train transforms include flips, 90° rotations, color jitter, and shift-scale-rotate (albumentations); test transforms are resize-only.

Supported datasets (must be obtained separately): `bus`, `busi`, `glas`, `ham10000`, `kvasir-instrument`. Run `python Dataset/show_pkl.py` to inspect a split file.

## Configs

Two YAMLs per dataset under `configs/`:
- `<dataset>_train.yaml` — paths, GPU list, epochs, batch size, model channel config, lr, loss weights.
- `<dataset>_valid.yaml` — points `model_weight` at a `.pth` checkpoint and `save_seg_img_path` at the output mask directory.

`configs/v2-inference-v-first-stage-VAE.yaml` defines the SD VAE (4-channel latent, scale_factor 0.18215) and should not be edited.

## Checkpoints and outputs

- Training writes to `ckpt/<timestamp>/checkpoints/` with `best_valid_dice.pth`, `best_valid_loss*.pth`, and periodic snapshots every `save_freq` epochs. TensorBoard logs go to `ckpt/<timestamp>/logs/`.
- Validation writes predicted masks to `save_seg_img_path` and a per-case + Avg/Std `results.csv` (DSC, IoU) to `snapshot_path`.
- Released per-dataset checkpoints live under `ckpt/valid_<dataset>/best_model.pth`.

## Gotchas

- `SD-VAE-weights/768-v-ema-first-stage-VAE.ckpt` and the dataset folders are **not** in git; the code will fail to start without them.
- The VAE config path (`./configs/v2-inference-v-first-stage-VAE.yaml`) and VAE weights path are hard-coded in `train.py`/`valid.py`, so run from the repo root.
- `valid.py` derives the ground-truth mask directory from `pickle_file_path` as `<parent>/masks` — keep the dataset layout intact.
