from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
import sklearn.ensemble
import sklearn.neighbors
from sklearn.model_selection import GridSearchCV

from .chem import morgan_fingerprint_array
from .metrics import balanced_accuracy_for_subset


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
    model, task_sample = _fit_local_sklearn_estimator(
        model_name=model_name,
        assay_id=assay_id,
        records_by_id=records_by_id,
        support_ids=support_ids,
        query_ids=query_ids,
        use_grid_search=use_grid_search,
        model_params=model_params,
    )
    return _score_episode_samples(model, task_sample.test_samples)


def score_official_baseline_episode(
    *,
    model_name: str,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_ids: list[str],
    query_ids: list[str],
    use_grid_search: bool = True,
    model_params: dict | None = None,
) -> dict[str, float]:
    from .fsmol_bridge import load_module_from_script, resolve_script_path

    if model_name not in {"randomForest", "kNN"}:
        raise ValueError(f"Unsupported official baseline model: {model_name}")

    baseline_spec = default_adapter_registry()["baseline"]
    baseline_module = load_module_from_script(resolve_script_path(baseline_spec))
    task_sample = build_sklearn_task_sample(
        assay_id=assay_id,
        records_by_id=records_by_id,
        support_ids=support_ids,
        query_ids=query_ids,
    )
    X_train = np.array([sample.get_fingerprint() for sample in task_sample.train_samples])
    y_train = np.array([float(sample.bool_label) for sample in task_sample.train_samples])
    X_test = np.array([sample.get_fingerprint() for sample in task_sample.test_samples])

    if use_grid_search:
        grid = dict(baseline_module.DEFAULT_GRID_SEARCH[model_name])
        if model_name == "kNN":
            grid["n_neighbors"] = [value for value in grid["n_neighbors"] if value < int(len(task_sample.train_samples) / 2)]
        model = GridSearchCV(baseline_module.NAME_TO_MODEL_CLS[model_name](), grid)
        model.fit(X_train, y_train)
        estimator = model.best_estimator_
    else:
        estimator = baseline_module.NAME_TO_MODEL_CLS[model_name]()
        if model_params:
            estimator.set_params(**model_params)
        estimator.fit(X_train, y_train)

    scores = estimator.predict_proba(X_test)[:, 1]
    return {
        molecule_id: float(score)
        for molecule_id, score in zip(query_ids, scores)
    }


def select_cliff_aware_hard_negatives(
    *,
    support_pos_ids: list[str],
    excluded_ids: set[str],
    anchor_to_hardnegs: dict[str, list[str]],
    max_per_anchor: int = 1,
) -> list[str]:
    selected: list[str] = []
    seen = set(excluded_ids)
    for anchor_id in support_pos_ids:
        added = 0
        for neg_id in anchor_to_hardnegs.get(anchor_id, []):
            if neg_id in seen:
                continue
            selected.append(neg_id)
            seen.add(neg_id)
            added += 1
            if added >= max_per_anchor:
                break
    return selected


def score_cliff_aware_sklearn_episode(
    *,
    model_name: str,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_pos_ids: list[str],
    support_neg_ids: list[str],
    query_ids: list[str],
    anchor_to_hardnegs: dict[str, list[str]],
    use_grid_search: bool = False,
    model_params: dict | None = None,
) -> dict[str, float]:
    augmented_support_ids = _build_cliff_aware_support_ids(
        support_pos_ids=support_pos_ids,
        support_neg_ids=support_neg_ids,
        query_ids=query_ids,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    return score_sklearn_episode(
        model_name=model_name,
        assay_id=assay_id,
        records_by_id=records_by_id,
        support_ids=augmented_support_ids,
        query_ids=query_ids,
        use_grid_search=use_grid_search,
        model_params=model_params,
    )


def score_decision_aware_sklearn_episode(
    *,
    model_name: str,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_pos_ids: list[str],
    support_neg_ids: list[str],
    query_ids: list[str],
    anchor_to_hardnegs: dict[str, list[str]],
    use_grid_search: bool = False,
    model_params: dict | None = None,
) -> dict[str, object]:
    if model_name != "kNN":
        raise ValueError("The decision-aware backend currently supports only model_name='kNN'.")
    augmented_support_ids = _build_cliff_aware_support_ids(
        support_pos_ids=support_pos_ids,
        support_neg_ids=support_neg_ids,
        query_ids=query_ids,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    model, task_sample = _fit_local_sklearn_estimator(
        model_name=model_name,
        assay_id=assay_id,
        records_by_id=records_by_id,
        support_ids=augmented_support_ids,
        query_ids=query_ids,
        use_grid_search=use_grid_search,
        model_params=model_params,
    )
    support_scores = _score_episode_samples(model, task_sample.train_samples)
    support_labels = {sample.molecule_id: int(sample.bool_label) for sample in task_sample.train_samples}
    return {
        "scores": _score_episode_samples(model, task_sample.test_samples),
        "decision_threshold": select_support_decision_threshold(
            support_scores=support_scores,
            support_labels=support_labels,
        ),
    }


def select_support_decision_threshold(
    *,
    support_scores: Mapping[str, float],
    support_labels: Mapping[str, int],
) -> float:
    valid_ids = [molecule_id for molecule_id in support_scores if molecule_id in support_labels]
    if not valid_ids:
        return 0.5

    unique_scores = sorted({float(support_scores[molecule_id]) for molecule_id in valid_ids})
    candidate_thresholds = {0.5, *unique_scores}
    if unique_scores:
        candidate_thresholds.add(float(np.nextafter(max(unique_scores), np.inf)))

    labels = {molecule_id: int(support_labels[molecule_id]) for molecule_id in valid_ids}
    ranked_candidates = []
    for threshold in candidate_thresholds:
        predictions = {
            molecule_id: int(float(support_scores[molecule_id]) >= float(threshold))
            for molecule_id in valid_ids
        }
        balanced_accuracy = balanced_accuracy_for_subset(None, labels, predictions, valid_ids)
        ranked_candidates.append(
            (
                -1.0 if balanced_accuracy is None else -float(balanced_accuracy),
                abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    _, _, best_threshold = min(ranked_candidates)
    return best_threshold


def diagnose_official_adapter_availability() -> dict[str, dict[str, object]]:
    from .fsmol_bridge import load_callable_from_spec

    report: dict[str, dict[str, object]] = {}
    for name, spec in default_adapter_registry().items():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FileType Enum is Deprecated.*",
                    category=DeprecationWarning,
                )
                callable_obj = load_callable_from_spec(spec)
        except Exception as exc:
            report[name] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
        else:
            report[name] = {
                "available": True,
                "callable": callable_obj.__name__,
            }
    return report


def _record_to_episode_molecule(record: dict) -> SklearnEpisodeMolecule:
    fingerprint = record.get("fingerprint")
    if fingerprint is None:
        fingerprint = morgan_fingerprint_array(record["canonical_isomeric_smiles"])
    if fingerprint is None:
        raise ValueError(f"Could not derive fingerprint for molecule {record['molecule_id']}")
    return SklearnEpisodeMolecule(
        molecule_id=str(record["molecule_id"]),
        smiles=str(record["canonical_isomeric_smiles"]),
        bool_label=bool(record["label"]),
        fingerprint=np.asarray(fingerprint, dtype=np.float32),
    )


def _build_cliff_aware_support_ids(
    *,
    support_pos_ids: list[str],
    support_neg_ids: list[str],
    query_ids: list[str],
    anchor_to_hardnegs: dict[str, list[str]],
) -> list[str]:
    extra_negatives = select_cliff_aware_hard_negatives(
        support_pos_ids=support_pos_ids,
        excluded_ids=set(support_pos_ids) | set(support_neg_ids) | set(query_ids),
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    return [*support_pos_ids, *support_neg_ids, *extra_negatives]


def _fit_local_sklearn_estimator(
    *,
    model_name: str,
    assay_id: str,
    records_by_id: dict[str, dict],
    support_ids: list[str],
    query_ids: list[str],
    use_grid_search: bool,
    model_params: dict | None,
):
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

    model = NAME_TO_MODEL_CLS[model_name]()
    if model_params:
        model.set_params(**model_params)
    model.fit(X_train, y_train)
    return model, task_sample


def _score_episode_samples(model, samples: tuple[SklearnEpisodeMolecule, ...]) -> dict[str, float]:
    if not samples:
        return {}
    scores = model.predict_proba(np.array([sample.get_fingerprint() for sample in samples]))[:, 1]
    return {
        sample.molecule_id: float(score)
        for sample, score in zip(samples, scores)
    }
