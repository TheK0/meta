from __future__ import annotations

import os

import numpy as np

from fsmol_cliff.adapters import build_sklearn_task_sample, score_sklearn_episode


def test_loky_max_cpu_count_is_pinned_for_test_environment() -> None:
    assert os.environ.get("LOKY_MAX_CPU_COUNT") == "1"


def test_build_sklearn_task_sample_exposes_fsmol_like_ratios() -> None:
    records_by_id = {
        "a1": {
            "molecule_id": "a1",
            "canonical_isomeric_smiles": "CCO",
            "label": 1,
            "fingerprint": [1, 1, 0, 0],
        },
        "n1": {
            "molecule_id": "n1",
            "canonical_isomeric_smiles": "CCN",
            "label": 0,
            "fingerprint": [0, 0, 1, 1],
        },
        "q1": {
            "molecule_id": "q1",
            "canonical_isomeric_smiles": "CCF",
            "label": 1,
            "fingerprint": [1, 0, 0, 0],
        },
        "q2": {
            "molecule_id": "q2",
            "canonical_isomeric_smiles": "CCC",
            "label": 0,
            "fingerprint": [0, 0, 1, 0],
        },
    }

    task_sample = build_sklearn_task_sample(
        assay_id="CHEMBL1",
        records_by_id=records_by_id,
        support_ids=["a1", "n1"],
        query_ids=["q1", "q2"],
    )

    assert task_sample.train_pos_label_ratio == 0.5
    assert task_sample.test_pos_label_ratio == 0.5
    assert np.array_equal(task_sample.train_samples[0].get_fingerprint(), np.array([1, 1, 0, 0]))
    assert task_sample.train_samples[0].get_fingerprint().dtype == np.float32


def test_score_sklearn_episode_returns_query_scores_in_manifest_order() -> None:
    records_by_id = {
        "a1": {
            "molecule_id": "a1",
            "canonical_isomeric_smiles": "CCO",
            "label": 1,
            "fingerprint": [1.0, 1.0, 0.0, 0.0],
        },
        "n1": {
            "molecule_id": "n1",
            "canonical_isomeric_smiles": "CCN",
            "label": 0,
            "fingerprint": [0.0, 0.0, 1.0, 1.0],
        },
        "q_active": {
            "molecule_id": "q_active",
            "canonical_isomeric_smiles": "CCF",
            "label": 1,
            "fingerprint": [1.0, 0.9, 0.0, 0.0],
        },
        "q_inactive": {
            "molecule_id": "q_inactive",
            "canonical_isomeric_smiles": "CCC",
            "label": 0,
            "fingerprint": [0.0, 0.0, 0.9, 1.0],
        },
    }

    scores = score_sklearn_episode(
        model_name="kNN",
        assay_id="CHEMBL1",
        records_by_id=records_by_id,
        support_ids=["a1", "n1"],
        query_ids=["q_active", "q_inactive"],
        use_grid_search=False,
        model_params={"n_neighbors": 1},
    )

    assert list(scores) == ["q_active", "q_inactive"]
    assert scores["q_active"] > scores["q_inactive"]


def test_build_sklearn_task_sample_computes_fingerprint_from_smiles_when_missing() -> None:
    records_by_id = {
        "a1": {
            "molecule_id": "a1",
            "canonical_isomeric_smiles": "CCO",
            "label": 1,
        },
        "n1": {
            "molecule_id": "n1",
            "canonical_isomeric_smiles": "CCN",
            "label": 0,
        },
        "q1": {
            "molecule_id": "q1",
            "canonical_isomeric_smiles": "CCF",
            "label": 1,
        },
        "q2": {
            "molecule_id": "q2",
            "canonical_isomeric_smiles": "CCC",
            "label": 0,
        },
    }

    task_sample = build_sklearn_task_sample(
        assay_id="CHEMBL1",
        records_by_id=records_by_id,
        support_ids=["a1", "n1"],
        query_ids=["q1", "q2"],
    )

    assert task_sample.train_samples[0].get_fingerprint().shape[0] == 2048
    assert task_sample.train_samples[0].get_fingerprint().dtype == np.float32
