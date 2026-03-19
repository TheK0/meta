from __future__ import annotations

import importlib

import pytest

from fsmol_cliff.models import BenchmarkManifest, PairRecord


def _metrics_module():
    return importlib.import_module("fsmol_cliff.metrics")


def _aggregate_module():
    return importlib.import_module("fsmol_cliff.aggregate")


def _hypotheses_module():
    return importlib.import_module("fsmol_cliff.hypotheses")


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest.default()


def test_rank_1_assigns_half_credit_for_top_score_tie() -> None:
    metrics = _metrics_module()

    score = metrics.rank_1(
        _manifest(),
        labels={"q_active": 1, "q_inactive": 0},
        scores={"q_active": 0.91, "q_inactive": 0.91},
    )

    assert score == pytest.approx(0.5)


def test_cliff_balanced_accuracy_returns_na_when_subset_has_one_class() -> None:
    metrics = _metrics_module()

    score = metrics.c_bacc(
        _manifest(),
        labels={"q1": 1, "q2": 1},
        predictions={"q1": 1, "q2": 0},
        query_ids=["q1", "q2"],
    )

    assert score is None


def test_noncliff_balanced_accuracy_averages_class_recalls() -> None:
    metrics = _metrics_module()

    score = metrics.nc_bacc(
        _manifest(),
        labels={"q1": 1, "q2": 0, "q3": 1, "q4": 0},
        predictions={"q1": 1, "q2": 1, "q3": 0, "q4": 0},
        query_ids=["q1", "q2", "q3", "q4"],
    )

    assert score == pytest.approx(0.5)


def test_pair_success_rate_wrappers_support_ties_and_same_scaffold_filters() -> None:
    metrics = _metrics_module()

    query_pairs = [
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="qa1",
            neg_id="qn1",
            sim=0.93,
            gap_abs=1.2,
            same_scaffold=True,
            pair_type="cliff",
        ),
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="qa2",
            neg_id="qn2",
            sim=0.91,
            gap_abs=1.4,
            same_scaffold=False,
            pair_type="cliff",
        ),
    ]
    noncliff_pairs = [
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="na1",
            neg_id="nn1",
            sim=0.94,
            gap_abs=0.4,
            same_scaffold=True,
            pair_type="highsim_noncliff",
        ),
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="na2",
            neg_id="nn2",
            sim=0.90,
            gap_abs=0.3,
            same_scaffold=False,
            pair_type="highsim_noncliff",
        ),
    ]
    support_query_pairs = [
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="sa1",
            neg_id="sn1",
            sim=0.89,
            gap_abs=1.1,
            same_scaffold=True,
            pair_type="support_query",
        ),
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="sa2",
            neg_id="sn2",
            sim=0.88,
            gap_abs=1.0,
            same_scaffold=False,
            pair_type="support_query",
        ),
    ]

    assert metrics.q_psr(
        _manifest(),
        query_pairs,
        {"qa1": 0.40, "qn1": 0.40, "qa2": 0.95, "qn2": 0.10},
    ) == pytest.approx(0.75)
    assert metrics.ss_q_psr(
        _manifest(),
        query_pairs,
        {"qa1": 0.40, "qn1": 0.40, "qa2": 0.95, "qn2": 0.10},
    ) == pytest.approx(0.5)

    assert metrics.nc_psr(
        _manifest(),
        noncliff_pairs,
        {"na1": 0.05, "nn1": 0.65, "na2": 0.90, "nn2": 0.30},
    ) == pytest.approx(0.5)
    assert metrics.ss_nc_psr(
        _manifest(),
        noncliff_pairs,
        {"na1": 0.05, "nn1": 0.65, "na2": 0.90, "nn2": 0.30},
    ) == pytest.approx(0.0)

    assert metrics.sq_psr(
        _manifest(),
        support_query_pairs,
        {"sa1": 0.80, "sn1": 0.10, "sa2": 0.55, "sn2": 0.55},
    ) == pytest.approx(0.75)
    assert metrics.ss_sq_psr(
        _manifest(),
        support_query_pairs,
        {"sa1": 0.80, "sn1": 0.10, "sa2": 0.55, "sn2": 0.55},
    ) == pytest.approx(1.0)


def test_same_scaffold_pair_success_rate_returns_na_without_matching_pairs() -> None:
    metrics = _metrics_module()

    pairs = [
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="qa1",
            neg_id="qn1",
            sim=0.93,
            gap_abs=1.2,
            same_scaffold=False,
            pair_type="cliff",
        )
    ]

    assert metrics.ss_q_psr(
        _manifest(),
        pairs,
        {"qa1": 0.80, "qn1": 0.10},
    ) is None


def test_collapse_rates_track_prediction_collapse_and_same_scaffold_subset() -> None:
    metrics = _metrics_module()

    pairs = [
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="a1",
            neg_id="n1",
            sim=0.95,
            gap_abs=1.5,
            same_scaffold=True,
            pair_type="cliff",
        ),
        PairRecord(
            assay_id="CHEMBL1",
            anchor_id="a2",
            neg_id="n2",
            sim=0.90,
            gap_abs=1.1,
            same_scaffold=False,
            pair_type="cliff",
        ),
    ]

    discrete_predictions = {
        "a1": 1,
        "n1": 1,
        "a2": 1,
        "n2": 0,
    }

    assert metrics.scr(_manifest(), pairs, discrete_predictions) == pytest.approx(0.5)
    assert metrics.ss_scr(_manifest(), pairs, discrete_predictions) == pytest.approx(1.0)


def test_task_mean_ignores_na_values_and_reports_coverage() -> None:
    aggregate = _aggregate_module()

    summary = aggregate.task_mean([1.0, None, float("nan"), 0.5])

    assert summary == {
        "mean": pytest.approx(0.75),
        "coverage": pytest.approx(0.5),
        "valid_count": 2,
        "total_count": 4,
    }


def test_macro_mean_averages_only_tasks_with_valid_episode_means() -> None:
    aggregate = _aggregate_module()

    task_summaries = {
        "task-a": aggregate.task_mean([1.0, 0.6]),
        "task-b": aggregate.task_mean([None, float("nan")]),
        "task-c": aggregate.task_mean([0.2, None]),
    }

    summary = aggregate.macro_mean(task_summaries)

    assert summary == {
        "mean": pytest.approx(0.5),
        "coverage": pytest.approx(2 / 3),
        "valid_count": 2,
        "total_count": 3,
    }


def test_h1_accepts_when_cliff_metrics_lag_noncliff_controls() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h1(
        {
            "c_bacc": {"mean": 0.58},
            "nc_bacc": {"mean": 0.71},
            "q_psr": {"mean": 0.62},
            "nc_psr": {"mean": 0.85},
        }
    )

    assert decision["accepted"] is True


def test_h1_rejects_when_control_gap_is_missing() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h1(
        {
            "c_bacc": {"mean": 0.72},
            "nc_bacc": {"mean": 0.70},
            "q_psr": {"mean": 0.62},
            "nc_psr": {"mean": 0.85},
        }
    )

    assert decision["accepted"] is False


def test_h2_accepts_when_support_query_cliffs_are_harder_than_query_cliffs() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h2(
        {
            "q_psr": {"mean": 0.77},
            "sq_psr": {"mean": 0.65},
            "ss_sq_psr": {"mean": 0.51},
        }
    )

    assert decision["accepted"] is True


def test_h2_rejects_when_support_query_gap_is_not_ordered() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h2(
        {
            "q_psr": {"mean": 0.70},
            "sq_psr": {"mean": 0.74},
            "ss_sq_psr": {"mean": 0.60},
        }
    )

    assert decision["accepted"] is False


def test_h3_accepts_when_same_scaffold_collapse_exceeds_overall_collapse() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h3(
        {
            "scr": {"mean": 0.18},
            "ss_scr": {"mean": 0.44},
        }
    )

    assert decision["accepted"] is True


def test_h3_rejects_when_same_scaffold_collapse_is_not_worse() -> None:
    hypotheses = _hypotheses_module()

    decision = hypotheses.validate_h3(
        {
            "scr": {"mean": 0.18},
            "ss_scr": {"mean": 0.10},
        }
    )

    assert decision["accepted"] is False
