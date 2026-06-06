import logging
import pickle
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing JAX (via cg_bg) so env vars like CUDA_VISIBLE_DEVICES
# and XLA_PYTHON_CLIENT_MEM_FRACTION take effect at device initialization.
load_dotenv(override=True)

import time  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from cg_bg.flow import cgbg, evaluate  # noqa: E402
from cg_bg.utils import get_console, install_rich_traceback  # noqa: E402


def _hf_download(repo_id: str, filename: str) -> str:
    from cg_bg.data.hf import hf_download
    return hf_download(repo_id, filename)


OmegaConf.register_new_resolver("hf", _hf_download, use_cache=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:

    install_rich_traceback()
    console = get_console()
    logger = logging.getLogger(__name__)

    import jax

    rng = jax.random.PRNGKey(cfg.seed)
    rng1, rng2, rng3 = jax.random.split(rng, 3)

    with console.status("[green]Initializing components...[/green]"):
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataloader = hydra.utils.instantiate(cfg.dataloader)
        state = hydra.utils.instantiate(cfg.state, rng=rng1)()
        sampler = hydra.utils.instantiate(cfg.sampler, rng=rng2)
        pmf_trainer = hydra.utils.instantiate(cfg.pmf_trainer, rng=rng3)()
        stage = str(cfg.stage)

    final_state = None
    samples = None
    species = None
    box = None

    features = dataset[0].get("features", None)
    kT = dataset.kT
    if hasattr(dataset, "species"):
        species = dataset.species[0]
    if hasattr(dataset, "box"):
        box = dataset.box[0]
    std = getattr(dataset, "std", None)

    flow_params_path = Path("final_params.pkl").resolve()
    sample_dict_path = Path("proposed_samples.npz").resolve()
    weights_dict_path = Path("samples_and_weights.npz").resolve()

    if "1" in stage or stage == "all":
        console.rule("[bold cyan]Stage 1: Training[/bold cyan]")
        logger.info("Stage 1: Training started")

        train_start = time.perf_counter()
        final_state = cgbg.train(
            epochs=cfg.epochs,
            dataloader=dataloader,
            state=state,
        )
        train_elapsed = time.perf_counter() - train_start
        train_time = time.strftime("%H:%M:%S", time.gmtime(train_elapsed))
        console.print(f"[green]Training completed in {train_time}.[/green]")
        logger.info("Stage 1: Training completed in %.2fs", train_elapsed)

        with open(flow_params_path, "wb") as f:
            pickle.dump(final_state.params, f)
        console.print(f"[green]Parameters for flow model saved to[/green] {flow_params_path}")
        logger.info("Stage 1: Flow parameters saved to %s", flow_params_path)

    if "2" in stage or stage == "all":
        console.rule("[bold cyan]Stage 2: Sampling[/bold cyan]")
        logger.info("Stage 2: Sampling started")

        if final_state is None:
            with open(flow_params_path, "rb") as f:
                params = pickle.load(f)
            final_state = state.replace(params=params)
            console.print(f"[green]Parameters for flow model loaded from[/green] {flow_params_path}")
            logger.info("Stage 2: Flow parameters loaded from %s", flow_params_path)

        sample_start = time.perf_counter()
        samples = cgbg.sample(
            features=features,
            sampler=sampler,
            state=final_state,
            species=species,
            box=box,
            std=std,
        )
        sample_elapsed = time.perf_counter() - sample_start
        sample_time = time.strftime("%H:%M:%S", time.gmtime(sample_elapsed))
        console.print(f"[green]Sampling completed in {sample_time}.[/green]")
        logger.info("Stage 2: Sampling completed in %.2fs", sample_elapsed)

        jax.numpy.savez(sample_dict_path, **samples)
        console.print(f"[green]Proposals saved to[/green] {sample_dict_path}")
        logger.info("Stage 2: Proposals saved to %s", sample_dict_path)

    if "3" in stage or stage == "all":
        console.rule("[bold cyan]Stage 3: Energy and Weights[/bold cyan]")
        logger.info("Stage 3: Energy evaluation started")

        if samples is None:
            samples = jax.numpy.load(sample_dict_path, allow_pickle=True)
            samples = dict(samples)
            console.print(f"[green]Proposals loaded from[/green] {sample_dict_path}")
            logger.info("Stage 3: Proposals loaded from %s", sample_dict_path)

        energy_params_path = Path(cfg.energy_params_path).resolve()
        if energy_params_path.exists():
            with open(energy_params_path, "rb") as f:
                pmf_params = pickle.load(f)
                if isinstance(pmf_params, dict) and "params" in pmf_params:
                    pmf_params = pmf_params["params"]
            console.print(f"[green]Parameters for energy model loaded from:[/green] {energy_params_path}")
            logger.info("Stage 3: Energy parameters loaded from %s", energy_params_path)
        else:
            logger.error("Stage 3: Missing PMF parameters at %s", energy_params_path)
            raise FileNotFoundError(
                f"PMF parameters not found at: {energy_params_path}. Please train PMF parameters first."
            )

        with console.status("[green]Evaluating energy for reference data[/green]"):
            target_path = Path(cfg.target_path)
            data_ref = jax.numpy.load(target_path, allow_pickle=True)
            data_ref = dict(data_ref)
            energy_ref = cgbg.energy_evaluate(data=data_ref, trainer=pmf_trainer, params=pmf_params)
            jax.numpy.savez(target_path, **{**data_ref, "U": energy_ref})
        console.print(f"[green]Energy evaluated and saved for reference data at:[/green] {target_path}")

        energy_start = time.perf_counter()
        with console.status("[green]Evaluating energy and Computing weights for proposals[/green]"):
            energy = cgbg.energy_evaluate(data=samples, trainer=pmf_trainer, params=pmf_params)
            logw = cgbg.compute_log_weights(kT=kT, data=samples, energy=energy)
            jax.numpy.savez(weights_dict_path, **{**samples, "U": energy, "logw": logw})
        energy_elapsed = time.perf_counter() - energy_start
        energy_time = time.strftime("%H:%M:%S", time.gmtime(energy_elapsed))
        console.print(f"[green]Energy and weights evaluated in {energy_time}.[/green]")
        console.print(f"[green]Energy and weights evaluated and saved for proposals at:[/green] {weights_dict_path}")
        logger.info("Stage 3: Energy and weights completed in %.2fs", energy_elapsed)
        logger.info("Stage 3: Energy + weights saved to %s", weights_dict_path)

    if "4" in stage or stage == "all":
        console.rule("[bold cyan]Stage 4: Plotting[/bold cyan]")
        logger.info("Stage 4: Plotting started")

        for sample_path in [weights_dict_path, sample_dict_path]:
            if sample_path.exists():
                console.print(f"[green]Found sample file for plotting:[/green] {sample_path}")
                break
        else:
            raise FileNotFoundError("No samples found for plotting. Please run stages 2 and 3 first.")

        plot_target_path = Path(cfg.target_path)
        if "mb" in cfg.task:
            evaluate.plot_mb_plots(
                target_path=plot_target_path,
                sample_path=sample_path,
                kT=kT,
            )
        elif "ala" in cfg.task:
            implicit_path = Path(cfg.implicit_path)
            variant = "_".join(cfg.task.split("_")[:2])
            evaluate.plot_alanine_plots(
                variant=variant,
                target_path=plot_target_path,
                sample_path=sample_path,
                implicit_path=implicit_path,
                clip=cfg.clip,
                kT=kT,
                run_bootstrap=cfg.bootstrap,
                compute_torus_w2=cfg.compute_torus_w2,
                n_bootstraps=cfg.n_bootstraps,
            )


if __name__ == "__main__":
    main()
