<div align="center">

# Coarse-Grained Boltzmann Generators

<a href="https://arxiv.org/abs/2602.10637"><img src="https://img.shields.io/badge/arXiv-2602.10637-lightgrey?labelColor=b31b1b" alt="arXiv"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lightgrey?labelColor=yellow" alt="License"></a>
<a href="https://prefix.dev"><img src="https://img.shields.io/badge/Pixi-Env-lightgrey?labelColor=blue" alt="Pixi"></a>
<a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Hydra-1.3-lightgrey?labelColor=1f77b4" alt="Hydra"></a>
<a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/JAX-0.4.x-lightgrey?labelColor=orange" alt="JAX"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-lightgrey?labelColor=3776AB" alt="Python"></a>

<p align="center">
  <em>Boltzmann Generators for exact equilibrium sampling in coarse-grained representations, powered by JAX.</em>
</p>

---
</div>

## Overview

This repository implements Coarse-Grained Boltzmann Generators (CG-BGs) in JAX.
The codebase uses Hydra for configuration and experiment launching, Pixi for
reproducible environments, and Rich for terminal progress and status displays.

## Install

This project uses Pixi for package management.

```bash
pixi install --frozen
```

## Quick Start

Preconfigured Hydra experiments are available via Pixi tasks:

- mb_ub
- mb_b
- ala2_cb_b
- ala2_cb_ub
- ala2_ha_b
- ala2_ha_ub
- ala3_ha_ub
- ala6_cb_ub

Run an experiment:

```bash
pixi run <task_name>
```

## Hydra Overrides

All configuration is managed by Hydra. You can override any config value from
the CLI.

Stages:
- 1: training
- 2: sampling
- 3: energy + weights
- 4: plotting

Example:

```bash
pixi run <task_name> device=<gpu_id> stage=234 hydra.run.dir=<output_dir>
```

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

The minimum required setting is `SCRATCH_DIR` — the local directory where downloaded
data and outputs are cached. All other variables are optional for the default dataset.

| Variable | Required | Description |
|---|---|---|
| `SCRATCH_DIR` | Yes | Local cache directory for data and outputs |
| `HF_TOKEN` | No | Only needed for private/gated HF repos or uploading data |
| `HF_REPO_ID` | No | Defaults to `bojuntum/CGPeptides` (public) |

## Data from Hugging Face

All experiment configs download data automatically from the
[`bojuntum/CGPeptides`](https://huggingface.co/datasets/bojuntum/CGPeptides) dataset on
the Hugging Face Hub. No manual download or token is needed — the files are fetched on
first run and cached under `$SCRATCH_DIR`.

To use a different dataset repo, set `HF_REPO_ID` in `.env`:

```bash
HF_REPO_ID="your-username/your-dataset"
```

## Weights & Biases

Training loss / learning rate, evaluation metrics, and plots are logged to wandb.
Configure via the `wandb` group, or disable logging entirely:

```bash
pixi run <task_name> wandb.project=my-project wandb.mode=disabled
```

## Citation

If you use CG-BGs, please cite:

```bibtex
@misc{chen2026coarsegrainedboltzmanngenerators,
      title={Coarse-Grained Boltzmann Generators},
      author={Weilong Chen and Bojun Zhao and Jan Eckwert and Julija Zavadlav},
      year={2026},
      eprint={2602.10637},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.10637},
}
```
