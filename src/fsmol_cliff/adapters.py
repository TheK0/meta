from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn.ensemble
import sklearn.neighbors


NAME_TO_MODEL_CLS = {
    "randomForest": sklearn.ensemble.RandomForestClassifier,
    "kNN": sklearn.neighbors.KNeighborsClassifier,
}


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    script_path: str
    callable_name: str
    requires_checkpoint: bool
    family: str


@dataclass(frozen=True)
class SklearnEpisodeMolecule:
    molecule_id: str
    smiles: str
    bool_label: bool
    fingerprint: np.ndarray

    def get_fingerprint(self) -> np.ndarray:
        return self.fingerprint


@dataclass(frozen=True)
class SklearnTaskSample:
    name: str
    train_samples: tuple[SklearnEpisodeMolecule, ...]
    valid_samples: tuple[SklearnEpisodeMolecule, ...]
    test_samples: tuple[SklearnEpisodeMolecule, ...]

    @staticmethod
    def _positive_fraction(samples: tuple[SklearnEpisodeMolecule, ...]) -> float:
        positives = sum(sample.bool_label for sample in samples)
        return positives / len(samples)

    @property
    def train_pos_label_ratio(self) -> float:
        return self._positive_fraction(self.train_samples)

    @property
    def test_pos_label_ratio(self) -> float:
        return self._positive_fraction(self.test_samples)


def default_adapter_registry() -> dict[str, AdapterSpec]:
    return {
        "baseline": AdapterSpec(
            name="baseline",
            script_path="fs_mol/baseline_test.py",
            callable_name="test",
            requires_checkpoint=False,
            family="sklearn",
        ),
        "mat": AdapterSpec(
            name="mat",
            script_path="fs_mol/mat_test.py",
            callable_name="eval_model_by_finetuning_on_task",
            requires_checkpoint=True,
            family="torch",
        ),
        "maml": AdapterSpec(
            name="maml",
            script_path="fs_mol/maml_test.py",
            callable_name="eval_model_by_finetuning_on_task",
            requires_checkpoint=False,
            family="tensorflow",
        ),
        "multitask": AdapterSpec(
            name="multitask",
            script_path="fs_mol/multitask_test.py",
            callable_name="eval_model_by_finetuning_on_task",
            requires_checkpoint=True,
            family="torch",
        ),
        "protonet": AdapterSpec(
            name="protonet",
            script_path="fs_mol/protonet_test.py",
            callable_name="evaluate_protonet_model",
            requires_checkpoint=True,
            family="torch",
        ),
    }


def build_sklearn_task_sample(
    *,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_ids: list[str],
    query_ids: list[str],
) -> SklearnTaskSample:
    return SklearnTaskSample(
        name=assay_id,
        train_samples=tuple(_record_to_episode_molecule(records_by_id[molecule_id]) for molecule_id in support_ids),
        valid_samples=tuple(),
        test_samples=tuple(_record_to_episode_molecule(records_by_id[molecule_id]) for molecule_id in query_ids),
    )


def score_sklearn_episode(
    *,
    model_name: str,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_ids: list[str],
    query_ids: list[str],
    use_grid_search: bool = False,
    model_params: dict | None = None,
) -> dict[str, float]:
    if use_grid_search:
        raise NotImplementedError("Grid search is not implemented in the local sklearn fallback.")
    if model_name not in NAME_TO_MODEL_CLS:
        raise ValueError(f"Unsupported sklearn adapter model: {model_name}")

    task_sample = build_sklearn_task_sample(
        assay_id=assay_id,
        records_by_id=records_by_id,
        support_ids=support_ids,
        query_ids=query_ids,
    )
    X_train = np.array([sample.get_fingerprint() for sample in task_sample.train_samples])
    y_train = np.array([float(sample.bool_label) for sample in task_sample.train_samples])
    X_test = np.array([sample.get_fingerprint() for sample in task_sample.test_samples])

    model = NAME_TO_MODEL_CLS[model_name]()
    if model_params:
        model.set_params(**model_params)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        molecule_id: float(score)
        for molecule_id, score in zip(query_ids, scores)
    }


def _record_to_episode_molecule(record: dict) -> SklearnEpisodeMolecule:
    return SklearnEpisodeMolecule(
        molecule_id=str(record["molecule_id"]),
        smiles=str(record["canonical_isomeric_smiles"]),
        bool_label=bool(record["label"]),
        fingerprint=np.array(record["fingerprint"]),
    )
