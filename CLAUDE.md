# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This codebase implements **Coarse-Grained Boltzmann Generators (CG-BGs)** in JAX — normalizing flows trained to generate equilibrium samples in coarse-grained molecular representations. The pipeline runs in up to four sequential stages: training, sampling, energy/weight evaluation, and plotting.

## Environment Setup

This project uses [Pixi](https://prefix.dev) for reproducible environments (not pip/conda directly):

```bash
pixi install --frozen    # install all dependencies
```

The `external/` directory contains editable installs of `chemtrain` and `chemutils` (local submodule dependencies). Do not modify these unless working on the dependency itself.

## Running Experiments

All experiments are launched via `main.py` with Hydra configuration:

```bash
pixi run <task_name>           # run a preconfigured experiment
pixi run <task_name> stage=2   # run only stage 2 (sampling)
pixi run <task_name> stage=234 device=<gpu_id> hydra.run.dir=<output_dir>
```

Predefined tasks (Pixi shortcuts for `python main.py +experiment=<name>`):
- `mb_ub`, `mb_b` — Müller-Brown potential (unbiased/biased)
- `ala2_cb_b`, `ala2_cb_ub`, `ala2_ha_b`, `ala2_ha_ub` — Alanine dipeptide (CB/HA beads, biased/unbiased)
- `ala3_ha_ub`, `ala6_cb_ub` — Alanine tri/hexapeptide

**Stages:**
- `1` — training (saves `final_params.pkl`)
- `2` — sampling (saves `proposed_samples.npz`)
- `3` — energy evaluation + importance weight computation (saves `samples_and_weights.npz`)
- `4` — plotting (reads from stage 2 or 3 output)

`stage` can be any combination of digits (e.g. `stage=234`) or `stage=all`.

## Environment Variables

Copy `.env.example` to `.env`. The file is loaded automatically by `main.py` via `python-dotenv`; Hydra configs reference variables via `${oc.env:VAR}`.

- `SCRATCH_DIR` — required; local cache directory for downloaded data and outputs
- `HF_TOKEN` — optional; only needed for private/gated HF repos or uploading data
- `HF_REPO_ID` — optional; defaults to `bojuntum/CGPeptides` (set in `configs/main.yaml`)

## Linting

```bash
pixi run -e lint lint          # check
pixi run -e lint lint-fix      # autofix
pixi run -e lint format        # format
pixi run -e lint check         # check lint + format
pixi run -e lint pre-commit-install   # install git hooks
```

Ruff is configured in `pyproject.toml` (line length 120, rules E/F/I, excludes `./external/`).

## Architecture

### Entry Point

`main.py` is the single entry point. It uses `@hydra.main` to compose configuration and dispatches to `cg_bg.flow.cgbg` functions (`train`, `sample`, `energy_evaluate`, `compute_log_weights`).

### Configuration System (`configs/`)

Hydra config tree:
- `configs/main.yaml` — root defaults; references sub-configs for dataset, dataloader, model, optimizer, state, sampler, pmf_trainer
- `configs/experiment/*.yaml` — self-contained experiment overrides (each sets task, n_nodes, n_dim, data_path, etc.)
- `configs/model/` — model architectures (`ala_graph_transformer.yaml`, `mb_mlp.yaml`)
- `configs/pmf_trainer/` — energy model trainer configs

### Source Package (`src/cg_bg/`)

| Module | Purpose |
|---|---|
| `flow/cgbg.py` | High-level `train` (accepts `callbacks`), `sample`, `energy_evaluate`, `compute_log_weights` |
| `flow/callbacks.py` | Lightweight callback base + `CallbackList` + `WandbLoggingCallback` for the hand-rolled JAX loop (no Lightning) |
| `flow/train_utils.py` | `train_step` (JIT'd), `loss_fn` (flow-matching objective: `0.5‖u_t‖² − u_t·v_t`), `create_train_state` |
| `flow/ema.py` | `EMATrainState` — Flax train state with exponential moving average of params |
| `flow/ode_solver_diffrax.py` | `batched_sampler` — integrates the learned ODE via Diffrax (Dopri5/Tsit5), computes log-probabilities via divergence trace |
| `flow/evaluate.py` | Plotting and metric helpers for Müller-Brown and alanine systems (Ramachandran, PMF, Wasserstein-2) |
| `models/graph_transformer.py` | Equivariant `GraphTransformer` model (Flax/Equinox) for molecular systems |
| `models/mlp.py` | Simple MLP model for 2D toy systems (Müller-Brown) |
| `models/rbf_mlp.py` | RBF-kernel MLP for PMF energy evaluation |
| `pmf/trainers.py` | Wrappers around `chemtrain` trainers for the PMF (potential of mean force) energy model |
| `data/ala/` | `ALADataset` (torch Dataset for alanine .npz files), preprocessing, collation |
| `data/mb/` | Data and preprocessing for Müller-Brown potential |
| `data/hf.py` | `resolve_data_path` — download a dataset .npz from the HF Hub (if `hf_repo_id` set) or use the local path |
| `utils/wandb_utils.py` | `init_wandb`/`log_metrics`/`log_image`/`finish_wandb` — no-op-safe wandb wrappers |
| `utils/standarization.py` | `standardize`/`destandardize` for center-of-mass normalization |

### Key Design Patterns

- **JAX + PyTorch DataLoader**: Training uses PyTorch's `DataLoader` for batching but immediately converts batches to JAX arrays (`jax.tree.map(jnp.asarray, batch)`).
- **Flow matching**: The loss is the regression objective `0.5‖u_t‖² − u_t·v_t` where `u_t` is the model's predicted velocity and `v_t` is the target conditional flow.
- **Two energy models**: The flow model (normalizing flow) is separate from the PMF energy model (MACE-based via `chemtrain`). Stage 3 loads pre-trained PMF parameters from `energy_params_path` in the config.
- **Importance weights**: Log weights are `logw = -U/kT - logp` (unnormalized), then normalized via `logsumexp`.
- **COM centering**: Alanine datasets are standardized by centering the center-of-mass and dividing by std; this is undone during sampling with log-probability correction.
