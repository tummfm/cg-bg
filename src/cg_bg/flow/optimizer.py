'''
The optimizer is based on the implementation in ScoreMD:
https://github.com/noegroup/ScoreMD/blob/main/src/scoremd/training/optimizer.py
'''
import optax


def get_lr_optimizer(
    epochs: int,
    num_samples: int,
    batch_size: int,
    learning_rate: float,
    min_learning_rate: float = None,
    clip: float = None,
    schedule: str = "constant",
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:

    num_steps = epochs * (num_samples // batch_size)
    if schedule == "constant":
        if clip is not None:
            return optax.chain(optax.clip(max_delta=clip), optax.adamw(learning_rate, weight_decay=weight_decay))
        else:
            return optax.adamw(learning_rate, weight_decay=weight_decay)

    elif schedule == "cosine":
        if min_learning_rate is None:
            raise ValueError("min_learning_rate must be provided for cosine schedule")
        if clip is not None:
            return optax.chain(
                optax.clip(max_delta=clip),
                optax.adamw(
                    learning_rate=optax.cosine_decay_schedule(
                        learning_rate, num_steps, min_learning_rate / learning_rate
                    ),
                    weight_decay=weight_decay,
                ),
            )
        else:
            return optax.adamw(
                learning_rate=optax.cosine_decay_schedule(learning_rate, num_steps, min_learning_rate / learning_rate),
                weight_decay=weight_decay,
            )

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
