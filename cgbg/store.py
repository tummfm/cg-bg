from hydra_zen import builds, make_config
from cgbg.data.mb.data import MBFLOWDataset, get_mb_dataloader, MBPMFDataset
from cgbg.data.aldp.data import ALDPFLOWDataset, get_aldp_dataloader, get_aldp_pmf_dataset
from cgbg.flow.train_utils import create_train_state
from cgbg.flow.ode_solver_diffrax import batched_sampler
from cgbg.flow.optimizer import get_lr_optimizer
from cgbg.pmf.trainers import get_aldp_trainer, get_mb_trainer
from cgbg.models.graph_transformer import GraphTransformer
from cgbg.models.mlp import MLP

def create_dataset_store(store):
    print("Creating dataset store...")
    dataset_store = store(group="dataset")
    dataset_store(
        builds(
            MBFLOWDataset,
            datapath="${hydra:runtime.cwd}/cgbg/data/mb/raw/mb_unbiased.npz",
            populate_full_signature=True,
        ),
        name="mb_unbiased",
    )
    dataset_store(
        builds(
            MBFLOWDataset,
            datapath="${hydra:runtime.cwd}/cgbg/data/mb/raw/mb_biased.npz",
            populate_full_signature=True,
        ),
        name="mb_biased",
    )
    dataset_store(
        builds(
            ALDPFLOWDataset,
            datapath="${hydra:runtime.cwd}/cgbg/data/aldp/raw/openmm_heavyatomMD.npz",
            feat_type="distinguish",
            n_nodes="${n_nodes}",
            populate_full_signature=True,
        ),
        name="aldp_unbiased",
    )
    dataset_store(
        builds(
            ALDPFLOWDataset,
            datapath="${hydra:runtime.cwd}/cgbg/data/aldp/raw/openmm_welltemp_1dot5_heavyatomMD.npz",
            feat_type="distinguish",
            n_nodes="${n_nodes}",
            populate_full_signature=True,
        ),
        name="aldp_biased",
    )
    return dataset_store

def create_dataloader_store(store):
    print("Creating dataloader store...")
    dataloader_store = store(group="dataloader")
    dataloader_store(
        builds(
            get_mb_dataloader,
            dataset="${dataset}",
            num_samples="${num_samples}",
            batch_size="${batch_size}",
            populate_full_signature=True,
        ),
        name="mb_dataloader",
    )
    dataloader_store(
        builds(
            get_aldp_dataloader,
            dataset="${dataset}",
            num_samples="${num_samples}",
            batch_size="${batch_size}",
            populate_full_signature=True,
        ),
        name="aldp_dataloader",
    )
    return dataloader_store

def create_pstate_store(store):
    print("Creating partial state store...")
    pstate_store = store(group="pstate")
    pstate_store(
        builds(
            create_train_state,
            model=builds(
                MLP,
                hidden_layers=96,
                embedding_layers=16,
                n_layers=3,
                zen_exclude=["parent", "name"],
                populate_full_signature=True,
            ),
            dataloader="${dataloader}",
            optimizer=builds(
                get_lr_optimizer,
                epochs="${epochs}",
                num_samples="${num_samples}",
                batch_size="${batch_size}",
                learning_rate=3e-4,
                min_learning_rate=1e-5,
                clip=1e3,
                schedule="cosine",
                weight_decay=1e-5,
                populate_full_signature=True,
            ),
            ema_decay=0.999,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="mb_pstate",
    )
    pstate_store(
        builds(
            create_train_state,
            model=builds(
                GraphTransformer,
                t0=0.0,
                t1=1.0,
                rescale_time=False,
                clip_time=False,
                hidden_nf=128,
                feature_embedding_dim=16,
                max_z=["${n_nodes}"],
                n_layers=3,
                use_intrinsic_coords=True,
                use_abs_coords=True,
                use_distances=True,
                dropout=0,
                zen_exclude=["parent", "name"],
                populate_full_signature=True,
            ),
            dataloader="${dataloader}",
            optimizer=builds(
                get_lr_optimizer,
                epochs="${epochs}",
                num_samples="${num_samples}",
                batch_size="${batch_size}",
                learning_rate=3e-4,
                min_learning_rate=1e-5,
                clip=1e3,
                schedule="cosine",
                weight_decay=1e-5,
                populate_full_signature=True,
            ),
            ema_decay=0.999,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="aldp_pstate",
    )
    return pstate_store

def create_sampler_store(store):
    print("Creating sampler store...")
    sampler_store = store(group="psampler")
    sampler_store(
        builds(
            batched_sampler,
            dt0=5e-3,
            method="dopri5",
            num_batches=1000,
            batch_size=1000,
            n_nodes="${n_nodes}",
            n_dim="${n_dim}",
            num_z=0,
            mean=True,
            rtol=1e-5,
            atol=1e-5,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="dopri5",
    )
    sampler_store(
        builds(
            batched_sampler,
            dt0=5e-3,
            method="tsit5",
            num_batches=1000,
            batch_size=1000,
            n_nodes="${n_nodes}",
            n_dim="${n_dim}",
            num_z=0,
            mean=True,
            rtol=1e-5,
            atol=1e-5,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="tsit5",
    )
    return sampler_store

def create_pmf_trainer_store(store):
    print("Creating pmf trainer store...")
    pmf_trainer_store = store(group="pmf_ptrainer")
    pmf_trainer_store(
        builds(
            get_aldp_trainer,
            dataset=builds(
                get_aldp_pmf_dataset,
                data_path="${hydra:runtime.cwd}/cgbg/data/aldp/raw/openmm_welltemp_9_heavyatomMD.npz",
                scale_R=1.0,
                scale_U=1.0,
                fractional=True,
                train_frac=1.0,
                n_nodes="${n_nodes}",
                seed="${seed}",
                populate_full_signature=True,
            ),
            batch_size=32,
            epochs=100,
            init_lr=0.001,
            decay_rate=0.9,
            r_cutoff=0.5,
            hidden_irreps="32x0e+32x1o",
            readout_irreps="16x0e",
            output_irreps="1x0e",
            max_ell=3,
            num_interactions=2,
            correlation=3,
            output_dir="",
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="aldp_trainer",
    )
    pmf_trainer_store(
        builds(
            get_mb_trainer,
            kT=1.0,
            dataloader=builds(
                get_mb_dataloader,
                dataset=builds(
                    MBPMFDataset,
                    datapath="${hydra:runtime.cwd}/cgbg/data/mb/raw/mb_biased.npz",
                ),
                num_samples=200_000,
                batch_size=128,
                populate_full_signature=True,
            ),
            hidden_dim=128,
            n_layers=4,
            learning_rate=1e-4,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="mb_trainer",
    )
    return pmf_trainer_store

def create_experiment_store(store, Config):
    print("Creating experiment store...")
    experiment_store = store(group="experiment",  package="_global_")
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "mb_unbiased"},
                {"override /dataloader": "mb_dataloader"},
                {"override /pstate": "mb_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "mb_trainer"},
            ],
            n_nodes=1,
            n_dim=1,
            num_samples=50_000,
            batch_size=256,
            epochs=2000,
            task="mb_unbiased",
            bases=(Config,),
        ),
        name="mb_flow_unbiased",
    )
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "mb_biased"},
                {"override /dataloader": "mb_dataloader"},
                {"override /pstate": "mb_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "mb_trainer"},
            ],
            n_nodes=1,
            n_dim=1,
            num_samples=50_000,
            batch_size=256,
            epochs=2000,
            task="mb_biased",
            bases=(Config,),
        ),
        name="mb_flow_biased",
    )
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "aldp_unbiased"},
                {"override /dataloader": "aldp_dataloader"},
                {"override /pstate": "aldp_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "aldp_trainer"},
            ],
            n_nodes=10,
            n_dim=3,
            num_samples=50_000,
            batch_size=256,
            epochs=5000,
            task="aldp_HA_unbiased",
            bases=(Config,),
        ),
        name="aldp_HA_flow_unbiased",
    )
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "aldp_biased"},
                {"override /dataloader": "aldp_dataloader"},
                {"override /pstate": "aldp_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "aldp_trainer"},
            ],
            n_nodes=10,
            n_dim=3,
            num_samples=50_000,
            batch_size=256,
            epochs=5000,
            task="aldp_HA_biased",
            bases=(Config,),
        ),
        name="aldp_HA_flow_biased",
    )
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "aldp_unbiased"},
                {"override /dataloader": "aldp_dataloader"},
                {"override /pstate": "aldp_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "aldp_trainer"},
            ],
            n_nodes=6,
            n_dim=3,
            num_samples=50_000,
            batch_size=256,
            epochs=5000,
            task="aldp_CB_unbiased",
            bases=(Config,),
        ),
        name="aldp_CB_flow_unbiased",
    )
    experiment_store(
        make_config(
            hydra_defaults=[
                "_self_",
                {"override /dataset": "aldp_biased"},
                {"override /dataloader": "aldp_dataloader"},
                {"override /pstate": "aldp_pstate"},
                {"override /psampler": "dopri5"},
                {"override /pmf_ptrainer": "aldp_trainer"},
            ],
            n_nodes=6,
            n_dim=3,
            num_samples=50_000,
            batch_size=256,
            epochs=5000,
            task="aldp_CB_biased",
            bases=(Config,),
        ),
        name="aldp_CB_flow_biased",
    )
    return experiment_store