from __future__ import annotations

import math

import pytest

from fsmol_cliff.assets import filter_assay_records, mine_assay_pairs
from fsmol_cliff.chem import canonicalize_isomeric_smiles, rdkit_is_available


def test_filter_assay_records_enforces_precision_and_duplicate_collapse() -> None:
    records = [
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "keep-a",
            "Y": 1,
            "Relation": "=",
            "CensoringQualifier": None,
            "LogRegressionProperty": 8.0,
            "CanonicalIsomericSmiles": "C[C@H](O)Cl",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "keep-b",
            "Y": 1,
            "Relation": "",
            "CensoringQualifier": "",
            "LogRegressionProperty": 8.2,
            "CanonicalIsomericSmiles": "C[C@H](O)Cl",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-conflict-a",
            "Y": 0,
            "Relation": "=",
            "LogRegressionProperty": 6.4,
            "CanonicalIsomericSmiles": "CCO",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-conflict-b",
            "Y": 1,
            "Relation": "=",
            "LogRegressionProperty": 6.5,
            "CanonicalIsomericSmiles": "CCO",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-range-a",
            "Y": 1,
            "Relation": "=",
            "LogRegressionProperty": 9.0,
            "CanonicalIsomericSmiles": "CCN",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-range-b",
            "Y": 1,
            "Relation": "=",
            "LogRegressionProperty": 8.3,
            "CanonicalIsomericSmiles": "CCN",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-relation",
            "Y": 0,
            "Relation": "<",
            "LogRegressionProperty": 5.0,
            "CanonicalIsomericSmiles": "CCC",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-qualifier",
            "Y": 0,
            "Relation": "=",
            "CensoringQualifier": ">",
            "LogRegressionProperty": 5.1,
            "CanonicalIsomericSmiles": "CCCO",
        },
        {
            "Assay_ID": "CHEMBL1",
            "compound_id": "drop-nan",
            "Y": 0,
            "Relation": "=",
            "LogRegressionProperty": math.nan,
            "CanonicalIsomericSmiles": "CCCC",
        },
    ]

    filtered = filter_assay_records("CHEMBL1", records)

    assert filtered == [
        {
            "assay_id": "CHEMBL1",
            "molecule_id": "keep-a",
            "canonical_isomeric_smiles": "C[C@H](O)Cl",
            "label": 1,
            "r": pytest.approx(8.1),
            "source_ids": ["keep-a", "keep-b"],
        }
    ]


def test_mine_assay_pairs_derives_protocol_subsets_and_diagnostics() -> None:
    actives = [
        {"molecule_id": "a1", "label": 1, "r": 8.0, "scaffold": "scaf-1", "canonical_isomeric_smiles": "A1"},
        {"molecule_id": "a2", "label": 1, "r": 7.1, "scaffold": "scaf-2", "canonical_isomeric_smiles": "A2"},
    ]
    inactives = [
        {"molecule_id": "n1", "label": 0, "r": 6.5, "scaffold": "scaf-1", "canonical_isomeric_smiles": "N1"},
        {"molecule_id": "n2", "label": 0, "r": 6.3, "scaffold": "scaf-2", "canonical_isomeric_smiles": "N2"},
        {"molecule_id": "n3", "label": 0, "r": 6.25, "scaffold": "scaf-3", "canonical_isomeric_smiles": "N3"},
    ]
    similarities = {
        ("a1", "n1"): 0.90,
        ("a1", "n2"): 0.87,
        ("a1", "n3"): 0.86,
        ("a2", "n1"): 0.86,
        ("a2", "n2"): 0.88,
        ("a2", "n3"): 0.84,
    }

    mined = mine_assay_pairs("CHEMBL1", actives, inactives, pair_similarity=similarities)

    assert [pair["pair_type"] for pair in mined["pairs"]["cliff"]] == ["cliff", "cliff", "cliff"]
    assert [(pair["anchor_id"], pair["neg_id"]) for pair in mined["pairs"]["highsim_discordant"]] == [
        ("a1", "n1"),
        ("a1", "n2"),
        ("a1", "n3"),
        ("a2", "n2"),
        ("a2", "n1"),
    ]
    assert [(pair["anchor_id"], pair["neg_id"]) for pair in mined["pairs"]["highsim_noncliff"]] == [
        ("a2", "n2"),
        ("a2", "n1"),
    ]
    assert [(pair["anchor_id"], pair["neg_id"]) for pair in mined["pairs"]["same_scaffold"]] == [
        ("a1", "n1"),
        ("a2", "n2"),
    ]

    diagnostics = mined["diagnostics"]
    assert diagnostics == {
        "tau": 0.85,
        "delta": 1.0,
        "hard_negative_pool_size": 32,
        "n_actives": 2,
        "n_inactives": 3,
        "n_molecules": 5,
        "n_candidate_pairs": 6,
        "n_highsim_discordant": 5,
        "n_cliff": 3,
        "n_highsim_noncliff": 2,
        "n_same_scaffold": 2,
        "n_highsim_active_anchors": 2,
        "n_cliff_active_anchors": 1,
        "n_same_scaffold_active_anchors": 2,
        "n_hard_negative_anchors": 2,
        "n_hard_negative_pairs": 5,
        "highsim_pair_fraction": pytest.approx(5 / 6),
        "cliff_fraction_within_highsim": pytest.approx(3 / 5),
        "same_scaffold_fraction_within_highsim": pytest.approx(2 / 5),
    }


def test_mine_assay_pairs_builds_stable_hard_negative_pools() -> None:
    actives = [
        {"molecule_id": "a1", "label": 1, "r": 8.5, "scaffold": "scaf-1", "canonical_isomeric_smiles": "A1"},
    ]
    inactives = [
        {
            "molecule_id": f"n{index:02d}",
            "label": 0,
            "r": 7.1,
            "scaffold": "scaf-x",
            "canonical_isomeric_smiles": f"N{index:02d}",
        }
        for index in range(35)
    ]
    similarities = {("a1", record["molecule_id"]): 0.91 for record in inactives}

    mined = mine_assay_pairs("CHEMBL1", actives, inactives, pair_similarity=similarities)

    pool = mined["hard_negative_pools"]["a1"]
    assert len(pool) == 32
    assert [pair["neg_id"] for pair in pool] == [f"n{index:02d}" for index in range(32)]


@pytest.mark.skipif(not rdkit_is_available(), reason="rdkit is not installed")
def test_canonicalize_isomeric_smiles_returns_rdkit_canonical_form() -> None:
    assert canonicalize_isomeric_smiles("OC[C@H](F)Cl") == "OC[C@H](F)Cl"
