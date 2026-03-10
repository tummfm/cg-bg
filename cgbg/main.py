import os
import sys
for arg in sys.argv:
    if arg.startswith("device="):
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=")[1]
        break
else:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
import pickle
import jax
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable
from hydra_zen.typing import Partial
from torch.utils.data import Dataset, DataLoader
from cgbg.flow import cgbg
from hydra_zen import MISSING, store, zen, builds
from cgbg.store import create_dataloader_store, create_dataset_store, create_experiment_store, create_pmf_trainer_store, create_pstate_store, create_sampler_store
from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.hydra_config import HydraConfig

def main(
    device: str,
    run_stage: str,
    task: str,
    epochs: int,
    dataset: Dataset,
    dataloader: DataLoader,
    pstate: Partial[Callable],
    psampler: Partial[Callable],
    seed: int,
    pmf_ptrainer: Any,
    ):

    print(f"Running on device {device}")
    
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.runtime.cwd
    
    rng = jax.random.PRNGKey(seed)
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    state = pstate(rng=rng1)

    # train stage
    if "1" in run_stage or run_stage == "all":
        
        print("-" * 50)
        print("Stage 1: Training")
        print("-" * 50)
        
        train_state = cgbg.train(
            epochs=epochs,
            dataloader=dataloader,
            state=state,
        )
        
        with open("final_state.pkl", "wb") as f:
            pickle.dump(train_state.params, f)
        print("Final train state parameters saved to final_state.pkl")

    else:
        
        with open("final_state.pkl", "rb") as f:
            train_state = pickle.load(f)
        print("Final train state parameters loaded from final_state.pkl")

    # sample stage
    if "2" in run_stage or run_stage == "all":
        
        print("-" * 50)
        print("Stage 2: Sampling")
        print("-" * 50)
        
        samples = cgbg.sample(
            dataset=dataset,
            psampler=psampler,
            state=train_state,
            rng=rng2,
        )
        
        jnp.savez("proposed_samples.npz", **samples)
        print("Proposed samples saved to proposed_samples.npz")
        
    else:
        
        samples = jnp.load("proposed_samples.npz", allow_pickle=True)
        samples = dict(samples)
        print("Proposed samples loaded from proposed_samples.npz")

    # reweight stage
    if "3" in run_stage or run_stage == "all":
        
        print("-" * 50)
        print("Stage 3: Reweighting")
        print("-" * 50)
        
        parts = task.split("_")
        parts[-1] = "biased"
        new_task = "_".join(parts)
        pmf_params_path = f"{work_dir}/cgbg/pmf/params/{new_task}/best_params.pkl"
        with open(pmf_params_path, "rb") as f:
            pmf_params = pickle.load(f)
        print("PMF parameters loaded from", pmf_params_path)
        
        pmf_trainer = pmf_ptrainer(rng=rng3)
        
        samples_and_weights = cgbg.reweight(
            raw=samples,
            trainer=pmf_trainer,
            params=pmf_params,
        )
        
        jnp.savez("samples_and_weights.npz", **samples_and_weights)
        print("Reweighted samples saved to samples_and_weights.npz")
        
    else:
        
        samples_and_weights = jnp.load("samples_and_weights.npz", allow_pickle=True)
        samples_and_weights = dict(samples_and_weights)
        print("Reweighted samples loaded from samples_and_weights.npz")

    if "4" in run_stage or run_stage == "all":
        
        print("-" * 50)
        print("Stage 4: Plotting")
        print("-" * 50)
        
        cgbg.plots(
            samples=samples_and_weights,
            task=task,
            work_dir=work_dir,
        )
        
        print("Plots saved")
    
if __name__ == "__main__":

    create_dataset_store(store)
    create_dataloader_store(store)
    create_pstate_store(store)
    create_sampler_store(store)
    create_pmf_trainer_store(store)

    BGConfig = builds(
        main,
        seed=0,
        device="0",
        run_stage="all",
        task=MISSING,
        epochs=MISSING,
        hydra_defaults=[
                "_self_",
                {"dataset": MISSING},
                {"dataloader": MISSING},
                {"pstate": MISSING},
                {"psampler": MISSING},
                {"pmf_ptrainer": MISSING},
            ],
        populate_full_signature=True,
    )
    store(BGConfig, name="config")
    create_experiment_store(store, BGConfig)

    run_dir_path = "outputs/${task}/${now:%Y-%m-%d/%H-%M-%S}"
    store(
        HydraConf(
            job=JobConf(
                chdir=True,
            ),
            run=RunDir(dir=run_dir_path)
            ),
        name="config", 
        group="hydra",
    )

    store.add_to_hydra_store(overwrite_ok=True)
    zen(main).hydra_main(config_path=None, config_name="config", version_base="1.3")
