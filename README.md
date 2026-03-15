<div align="center">

# Coarse-Grained Boltzmann Generators

<a href="https://arxiv.org/abs/2602.10637"><img src="https://img.shields.io/badge/arXiv-2602.10637-b31b1b.svg" alt="arXiv"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
<a href="https://prefix.dev"><img src="https://img.shields.io/badge/Environment-Pixi-blue" alt="Pixi"></a>

<p align="center">
  <em>Boltzmann Generators for exact equilibrium sampling in coarse-grained representations, powered by JAX.</em>
</p>

---
</div>

## 📖 Overview

This repository provides the official implementation of **Coarse-Grained Boltzmann Generators (CG-BGs)**. Unlike traditional BGs that operate in all-atom space, CG-BGs act directly in a **coarse-grained coordinate space**. By leveraging Conditional Flow Matching and [Enhanced Sampling Force Matching](https://github.com/tummfm/biased-force-matching), this method enables exact equilibrium sampling and reweighting for CG molecular systems.

## 🚀 Tutorial

### Installation

This project uses [pixi](https://github.com/prefix-dev/pixi) for package management to ensure strict version control and reproducibility across different systems. 

```bash
pixi install --frozen
```

### Quick Start

Six standard tasks are pre-configured:

* mb_flow_unbiased
* mb_flow_biased
* aldp_HA_flow_unbiased
* aldp_HA_flow_biased
* aldp_CB_flow_unbiased
* aldp_CB_flow_biased

Run a task using the following command:

```bash
pixi run <task_name>
```

### Cofiguration Overrides

Configuration Overrides
All configurations are managed by [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen/). Parameters can be modified via the Command Line Interface (CLI) or configuration files.

**Example: Running specific stages on a designated GPU**

```bash
# stage: 1-Training, 2-Sampling, 3-Reweighting, 4-Analysis
pixi run <task_name> device=<gpu_id> stage=234 hydra.run.dir=<output_dir>
```
## Citation
If you use ``CG-BGs`` please cite:
```bash
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