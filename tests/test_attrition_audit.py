from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fsmol_cliff.constants import BenchmarkProfile, ProtocolConstants
from fsmol_cliff.audit import (
    AuditThresholds,
    build_attrition_funnel,
    build_attrition_rows,
    sweep_attrition_thresholds,
    write_attrition_audit,
)


def _summary(
    assay_id: str,
    *,
    raw_assay_present: bool = True,
    tau: float = 0.85,
    delta: float = 1.0,
    num_valid_molecules: int = 60,
    num_positive_molecules: int = 20,
    num_negative_molecules: int = 20,
    num_highsim_discordant_pairs: int = 40,
    num_cliff_pairs: int = 30,
    num_noncliff_highsim_pairs: int = 10,
    num_anchor_molecules: int = 12,
    num_cliff_negatives: int = 12,
    num_same_scaffold_cliff_pairs: int = 0,
    m_avail: int = 3,
) -> dict:
    return {
        "assay_id": assay_id,
        "raw_assay_present": raw_assay_present,
        "tau": tau,
        "delta": delta,
        "num_valid_molecules": num_valid_molecules,
        "num_positive_molecules": num_positive_molecules,
        "num_negative_molecules": num_negative_molecules,
        "num_highsim_discordant_pairs": num_highsim_discordant_pairs,
        "num_cliff_pairs": num_cliff_pairs,
        "num_noncliff_highsim_pairs": num_noncliff_highsim_pairs,
        "num_anchor_molecules": num_anchor_molecules,
        "num_cliff_negatives": num_cliff_negatives,
        "num_same_scaffold_cliff_pairs": num_same_scaffold_cliff_pairs,
        "m_avail": m_avail,
    }


def test_build_attrition_rows_assigns_first_failure_stage_and_funnel_counts() -> None:
    thresholds = AuditThresholds(min_cliff_pairs=25, min_noncliff_pairs=10)
    summaries = [
        _summary("missing", raw_assay_present=False),
        _summary("legal", num_valid_molecules=49),
        _summary("class_balance", num_positive_molecules=14),
        _summary("highsim", num_highsim_discordant_pairs=34),
        _summary("cliff", num_cliff_pairs=24),
        _summary("noncliff", num_noncliff_highsim_pairs=9),
        _summary("anchors", num_anchor_molecules=9),
        _summary("cliff_negs", num_cliff_negatives=9),
        _summary("m_avail", m_avail=1),
        _summary("eligible", num_same_scaffold_cliff_pairs=4),
    ]

    rows = build_attrition_rows(summaries, thresholds=thresholds)
    by_assay = {row["assay_id"]: row for row in rows}

    assert by_assay["missing"]["failure_stage"] == "raw_assay_present"
    assert by_assay["legal"]["failure_stage"] == "legal_samples"
    assert by_assay["class_balance"]["failure_stage"] == "active_inactive_minimums"
    assert by_assay["highsim"]["failure_stage"] == "highsim_discordant_support"
    assert by_assay["cliff"]["failure_stage"] == "c_t"
    assert by_assay["noncliff"]["failure_stage"] == "d_t"
    assert by_assay["anchors"]["failure_stage"] == "a_t"
    assert by_assay["cliff_negs"]["failure_stage"] == "n_t_cliff"
    assert by_assay["m_avail"]["failure_stage"] == "m_avail"
    assert by_assay["eligible"]["failure_stage"] is None

    assert by_assay["m_avail"]["benchmark_eligible"] is True
    assert by_assay["m_avail"]["adversarial_eligible"] is False
    assert by_assay["eligible"]["benchmark_eligible"] is True
    assert by_assay["eligible"]["adversarial_eligible"] is True

    funnel = build_attrition_funnel(rows)
    assert funnel == [
        {"stage": "raw_assay_present", "passed_count": 9, "failed_count": 1},
        {"stage": "legal_samples", "passed_count": 8, "failed_count": 1},
        {"stage": "active_inactive_minimums", "passed_count": 7, "failed_count": 1},
        {"stage": "highsim_discordant_support", "passed_count": 6, "failed_count": 1},
        {"stage": "c_t", "passed_count": 5, "failed_count": 1},
        {"stage": "d_t", "passed_count": 4, "failed_count": 1},
        {"stage": "a_t", "passed_count": 3, "failed_count": 1},
        {"stage": "n_t_cliff", "passed_count": 2, "failed_count": 1},
        {"stage": "m_avail", "passed_count": 1, "failed_count": 1},
    ]


def test_sweep_attrition_thresholds_summarizes_each_threshold_combination() -> None:
    summaries = [
        _summary(
            "A",
            tau=0.85,
            delta=1.0,
            num_highsim_discordant_pairs=42,
            num_cliff_pairs=30,
            num_noncliff_highsim_pairs=12,
            num_anchor_molecules=11,
            num_cliff_negatives=11,
            num_same_scaffold_cliff_pairs=5,
            m_avail=3,
        ),
        _summary(
            "B",
            tau=0.85,
            delta=1.0,
            num_highsim_discordant_pairs=35,
            num_cliff_pairs=25,
            num_noncliff_highsim_pairs=10,
            num_anchor_molecules=10,
            num_cliff_negatives=10,
            num_same_scaffold_cliff_pairs=0,
            m_avail=1,
        ),
        _summary(
            "A",
            tau=0.90,
            delta=1.0,
            num_highsim_discordant_pairs=36,
            num_cliff_pairs=26,
            num_noncliff_highsim_pairs=10,
            num_anchor_molecules=10,
            num_cliff_negatives=10,
            num_same_scaffold_cliff_pairs=2,
            m_avail=2,
        ),
        _summary(
            "B",
            tau=0.90,
            delta=1.0,
            num_highsim_discordant_pairs=28,
            num_cliff_pairs=22,
            num_noncliff_highsim_pairs=6,
            num_anchor_molecules=9,
            num_cliff_negatives=9,
            num_same_scaffold_cliff_pairs=0,
            m_avail=0,
        ),
        _summary(
            "A",
            tau=0.85,
            delta=1.2,
            num_highsim_discordant_pairs=40,
            num_cliff_pairs=24,
            num_noncliff_highsim_pairs=16,
            num_anchor_molecules=10,
            num_cliff_negatives=10,
            num_same_scaffold_cliff_pairs=1,
            m_avail=1,
        ),
        _summary(
            "B",
            tau=0.85,
            delta=1.2,
            num_highsim_discordant_pairs=33,
            num_cliff_pairs=20,
            num_noncliff_highsim_pairs=13,
            num_anchor_molecules=8,
            num_cliff_negatives=9,
            num_same_scaffold_cliff_pairs=0,
            m_avail=0,
        ),
        _summary(
            "A",
            tau=0.90,
            delta=1.2,
            num_highsim_discordant_pairs=31,
            num_cliff_pairs=21,
            num_noncliff_highsim_pairs=10,
            num_anchor_molecules=9,
            num_cliff_negatives=9,
            num_same_scaffold_cliff_pairs=1,
            m_avail=0,
        ),
        _summary(
            "B",
            tau=0.90,
            delta=1.2,
            num_highsim_discordant_pairs=25,
            num_cliff_pairs=18,
            num_noncliff_highsim_pairs=7,
            num_anchor_molecules=7,
            num_cliff_negatives=8,
            num_same_scaffold_cliff_pairs=0,
            m_avail=0,
        ),
    ]

    sweep = sweep_attrition_thresholds(
        summaries,
        taus=[0.85, 0.90],
        deltas=[1.0, 1.2],
        min_cliff_pairs=[25, 30],
        min_noncliff_pairs=[10],
    )

    assert len(sweep) == 8

    by_combo = {
        (row["tau"], row["delta"], row["min_cliff_pairs"], row["min_noncliff_pairs"]): row for row in sweep
    }

    assert by_combo[(0.85, 1.0, 25, 10)] == {
        "tau": 0.85,
        "delta": 1.0,
        "min_cliff_pairs": 25,
        "min_noncliff_pairs": 10,
        "eligible_assay_count": 2,
        "total_cliff_pairs": 55,
        "total_anchors": 21,
        "adversarial_eligible_assay_count": 1,
        "same_scaffold_assay_count": 1,
        "same_scaffold_assay_fraction": 0.5,
        "same_scaffold_cliff_pair_count": 5,
    }
    assert by_combo[(0.85, 1.0, 30, 10)] == {
        "tau": 0.85,
        "delta": 1.0,
        "min_cliff_pairs": 30,
        "min_noncliff_pairs": 10,
        "eligible_assay_count": 1,
        "total_cliff_pairs": 30,
        "total_anchors": 11,
        "adversarial_eligible_assay_count": 1,
        "same_scaffold_assay_count": 1,
        "same_scaffold_assay_fraction": 1.0,
        "same_scaffold_cliff_pair_count": 5,
    }
    assert by_combo[(0.90, 1.0, 25, 10)] == {
        "tau": 0.9,
        "delta": 1.0,
        "min_cliff_pairs": 25,
        "min_noncliff_pairs": 10,
        "eligible_assay_count": 1,
        "total_cliff_pairs": 26,
        "total_anchors": 10,
        "adversarial_eligible_assay_count": 1,
        "same_scaffold_assay_count": 1,
        "same_scaffold_assay_fraction": 1.0,
        "same_scaffold_cliff_pair_count": 2,
    }
    assert by_combo[(0.85, 1.2, 25, 10)] == {
        "tau": 0.85,
        "delta": 1.2,
        "min_cliff_pairs": 25,
        "min_noncliff_pairs": 10,
        "eligible_assay_count": 0,
        "total_cliff_pairs": 0,
        "total_anchors": 0,
        "adversarial_eligible_assay_count": 0,
        "same_scaffold_assay_count": 0,
        "same_scaffold_assay_fraction": 0.0,
        "same_scaffold_cliff_pair_count": 0,
    }
    assert by_combo[(0.90, 1.2, 30, 10)] == {
        "tau": 0.9,
        "delta": 1.2,
        "min_cliff_pairs": 30,
        "min_noncliff_pairs": 10,
        "eligible_assay_count": 0,
        "total_cliff_pairs": 0,
        "total_anchors": 0,
        "adversarial_eligible_assay_count": 0,
        "same_scaffold_assay_count": 0,
        "same_scaffold_assay_fraction": 0.0,
        "same_scaffold_cliff_pair_count": 0,
    }


def test_write_attrition_audit_reads_profile_specific_summaries_and_writes_v4_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_dir = tmp_path / "release"
    audit_dir = tmp_path / "audit"
    release_dir.mkdir()
    pd.DataFrame(
        [
            _summary("CHEMBL1", tau=0.8, delta=1.0, num_same_scaffold_cliff_pairs=3),
            _summary("CHEMBL2", tau=0.8, delta=1.0, num_cliff_pairs=20),
        ]
    ).to_parquet(release_dir / "task_summaries_relaxed.parquet", index=False)

    monkeypatch.setattr(
        "fsmol_cliff.audit.build_real_audit_summaries",
        lambda **_: [
            _summary("CHEMBL1", tau=0.8, delta=1.0, num_same_scaffold_cliff_pairs=3),
            _summary("CHEMBL2", tau=0.8, delta=1.0, num_cliff_pairs=20),
        ],
    )

    summary = write_attrition_audit(
        release_dir=release_dir,
        data_dir=tmp_path / "fsmol",
        output_dir=audit_dir,
        profile="relaxed",
        taus=[0.8],
        deltas=[1.0],
        min_cliff_pairs=[25],
        min_noncliff_pairs=[10],
    )

    assert summary["raw_assays"] == 2
    assert summary["profile"] == "relaxed"
    assert (audit_dir / "attrition_by_assay.parquet").exists()
    assert (audit_dir / "threshold_sensitivity.parquet").exists()
    assert not (audit_dir / "threshold_sweep.parquet").exists()
    payload = json.loads((audit_dir / "attrition_summary.json").read_text())
    assert payload["profile"] == "relaxed"


def test_write_attrition_audit_infers_thresholds_and_protocol_from_requested_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_dir = tmp_path / "release"
    audit_dir = tmp_path / "audit"
    release_dir.mkdir()
    pd.DataFrame(
        [
            {
                "assay_id": "CHEMBL_AUX",
                "num_valid_molecules": 60,
                "num_positive_molecules": 20,
                "num_negative_molecules": 20,
                "num_cliff_pairs": 10,
                "num_noncliff_highsim_pairs": 5,
                "num_anchor_molecules": 10,
                "num_cliff_negatives": 10,
                "num_same_scaffold_cliff_pairs": 2,
                "m_avail": 2,
            }
        ]
    ).to_parquet(release_dir / "task_summaries_relaxed_covext_10_5.parquet", index=False)

    monkeypatch.setattr(
        "fsmol_cliff.audit.PROFILE_SPECS",
        {
            "relaxed_covext_10_5": BenchmarkProfile(
                name="relaxed_covext_10_5",
                constants=ProtocolConstants(
                    similarity_threshold=0.77,
                    activity_gap_threshold=1.3,
                    hard_negative_pool_size=32,
                    adversarial_injection_ratio=0.5,
                ),
                min_cliff_pairs=10,
                min_noncliff_pairs=5,
            )
        },
        raising=False,
    )
    monkeypatch.setattr(
        "fsmol_cliff.audit.build_real_audit_summaries",
        lambda **_: [
            _summary(
                "CHEMBL_AUX",
                tau=0.77,
                delta=1.3,
                num_highsim_discordant_pairs=15,
                num_cliff_pairs=10,
                num_noncliff_highsim_pairs=5,
                num_anchor_molecules=10,
                num_cliff_negatives=10,
                num_same_scaffold_cliff_pairs=2,
                m_avail=2,
            )
        ],
    )

    summary = write_attrition_audit(
        release_dir=release_dir,
        data_dir=tmp_path / "fsmol",
        output_dir=audit_dir,
        profile="relaxed_covext_10_5",
        taus=[0.77],
        deltas=[1.3],
        min_cliff_pairs=[10],
        min_noncliff_pairs=[5],
    )

    attrition_rows = pd.read_parquet(audit_dir / "attrition_by_assay.parquet").to_dict(orient="records")

    assert summary["profile"] == "relaxed_covext_10_5"
    assert summary["eligible_assays"] == 1
    assert summary["adversarial_eligible_assays"] == 1
    assert attrition_rows == [
        {
            "assay_id": "CHEMBL_AUX",
            "num_valid_molecules": 60,
            "num_positive_molecules": 20,
            "num_negative_molecules": 20,
            "num_cliff_pairs": 10,
            "num_noncliff_highsim_pairs": 5,
            "num_anchor_molecules": 10,
            "num_cliff_negatives": 10,
            "num_same_scaffold_cliff_pairs": 2,
            "m_avail": 2,
            "raw_assay_present": True,
            "tau": 0.77,
            "delta": 1.3,
            "num_highsim_discordant_pairs": 15,
            "failure_stage": None,
            "benchmark_eligible": True,
            "adversarial_eligible": True,
        }
    ]


def test_write_attrition_audit_rejects_legacy_task_summaries_for_auxiliary_profile(
    tmp_path: Path,
) -> None:
    release_dir = tmp_path / "release"
    audit_dir = tmp_path / "audit"
    release_dir.mkdir()
    pd.DataFrame([_summary("CHEMBL_LEGACY")]).to_parquet(release_dir / "task_summaries.parquet", index=False)

    with pytest.raises(FileNotFoundError, match="task_summaries_relaxed_covext_10_10.parquet"):
        write_attrition_audit(
            release_dir=release_dir,
            data_dir=tmp_path / "fsmol",
            output_dir=audit_dir,
            profile="relaxed_covext_10_10",
            taus=[0.8],
            deltas=[1.0],
            min_cliff_pairs=[10],
            min_noncliff_pairs=[10],
        )


def test_write_attrition_audit_threshold_sensitivity_preserves_profile_non_swept_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_dir = tmp_path / "release"
    audit_dir = tmp_path / "audit"
    release_dir.mkdir()
    pd.DataFrame(
        [
            {
                "assay_id": "CHEMBL_SWEEP",
                "num_valid_molecules": 60,
                "num_positive_molecules": 20,
                "num_negative_molecules": 20,
                "num_cliff_pairs": 10,
                "num_noncliff_highsim_pairs": 5,
                "num_anchor_molecules": 10,
                "num_cliff_negatives": 10,
                "num_same_scaffold_cliff_pairs": 1,
                "m_avail": 2,
            }
        ]
    ).to_parquet(release_dir / "task_summaries_relaxed_covext_10_5.parquet", index=False)

    monkeypatch.setattr(
        "fsmol_cliff.audit.PROFILE_SPECS",
        {
            "relaxed_covext_10_5": BenchmarkProfile(
                name="relaxed_covext_10_5",
                constants=ProtocolConstants(
                    similarity_threshold=0.8,
                    activity_gap_threshold=1.0,
                    hard_negative_pool_size=32,
                    adversarial_injection_ratio=0.5,
                ),
                min_cliff_pairs=10,
                min_noncliff_pairs=5,
                min_valid_molecules=65,
            )
        },
        raising=False,
    )
    monkeypatch.setattr(
        "fsmol_cliff.audit.build_real_audit_summaries",
        lambda **_: [
            _summary(
                "CHEMBL_SWEEP",
                tau=0.8,
                delta=1.0,
                num_valid_molecules=60,
                num_highsim_discordant_pairs=15,
                num_cliff_pairs=10,
                num_noncliff_highsim_pairs=5,
                num_anchor_molecules=10,
                num_cliff_negatives=10,
                num_same_scaffold_cliff_pairs=1,
                m_avail=2,
            )
        ],
    )

    write_attrition_audit(
        release_dir=release_dir,
        data_dir=tmp_path / "fsmol",
        output_dir=audit_dir,
        profile="relaxed_covext_10_5",
        taus=[0.8],
        deltas=[1.0],
        min_cliff_pairs=[10],
        min_noncliff_pairs=[5],
    )

    threshold_rows = pd.read_parquet(audit_dir / "threshold_sensitivity.parquet").to_dict(orient="records")

    assert threshold_rows == [
        {
            "tau": 0.8,
            "delta": 1.0,
            "min_cliff_pairs": 10,
            "min_noncliff_pairs": 5,
            "eligible_assay_count": 0,
            "total_cliff_pairs": 0,
            "total_anchors": 0,
            "adversarial_eligible_assay_count": 0,
            "same_scaffold_assay_count": 0,
            "same_scaffold_assay_fraction": 0.0,
            "same_scaffold_cliff_pair_count": 0,
        }
    ]
