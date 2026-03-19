from __future__ import annotations

from fsmol_cliff.adapters import (
    diagnose_official_adapter_availability,
    score_official_baseline_episode,
)


def test_score_official_baseline_episode_returns_ordered_scores() -> None:
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
        "qa": {
            "molecule_id": "qa",
            "canonical_isomeric_smiles": "CCF",
            "label": 1,
            "fingerprint": [1.0, 0.8, 0.0, 0.0],
        },
        "qn": {
            "molecule_id": "qn",
            "canonical_isomeric_smiles": "CCC",
            "label": 0,
            "fingerprint": [0.0, 0.0, 0.8, 1.0],
        },
    }

    scores = score_official_baseline_episode(
        model_name="kNN",
        assay_id="CHEMBL1",
        records_by_id=records_by_id,
        support_ids=["a1", "n1"],
        query_ids=["qa", "qn"],
        use_grid_search=False,
        model_params={"n_neighbors": 1},
    )

    assert list(scores) == ["qa", "qn"]
    assert scores["qa"] > scores["qn"]


def test_diagnose_official_adapter_availability_reports_baseline_ready() -> None:
    report = diagnose_official_adapter_availability()

    assert report["baseline"]["available"] is True
    assert "callable" in report["baseline"]
    assert report["mat"]["available"] is False
