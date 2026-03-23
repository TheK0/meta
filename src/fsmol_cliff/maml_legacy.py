from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def split_support_for_validation(
    *,
    support_pos_ids: list[str],
    support_neg_ids: list[str],
    holdout_per_class: int = 1,
) -> tuple[list[str], list[str]]:
    if len(support_pos_ids) <= holdout_per_class or len(support_neg_ids) <= holdout_per_class:
        raise ValueError("Not enough support samples to create validation holdout")
    train_ids = support_pos_ids[:-holdout_per_class] + support_neg_ids[:-holdout_per_class]
    valid_ids = support_pos_ids[-holdout_per_class:] + support_neg_ids[-holdout_per_class:]
    return train_ids, valid_ids


def enable_legacy_adam_optimizer() -> None:
    import tensorflow as tf

    tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam


def maml_checkpoint_args(checkpoint_path: str | Path) -> SimpleNamespace:
    return SimpleNamespace(
        trained_model=str(checkpoint_path),
        use_fresh_param_init=False,
        model_params_override=None,
    )
