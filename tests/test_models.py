from __future__ import annotations

from fsmol_cliff.constants import DEFAULT_EPISODE_CONFIG, DEFAULT_PROTOCOL_CONSTANTS, DEFAULT_SEEDS
from fsmol_cliff.models import BenchmarkManifest, PairRecord


def test_benchmark_manifest_default_round_trips_to_protocol_dict() -> None:
    manifest = BenchmarkManifest.default().to_dict()

    assert manifest["benchmark_version"] == "v3.0"
    assert manifest["episode_config"] == DEFAULT_EPISODE_CONFIG.to_dict()
    assert manifest["constants"] == DEFAULT_PROTOCOL_CONSTANTS.to_dict()
    assert manifest["seeds"] == list(DEFAULT_SEEDS)
    assert manifest["episodes_per_split"] == 400


def test_pair_record_sort_key_matches_protocol_ordering() -> None:
    pair = PairRecord(
        assay_id="CHEMBL1",
        anchor_id="mol-active",
        neg_id="mol-neg",
        sim=0.92,
        gap_abs=1.4,
        same_scaffold=True,
        pair_type="cliff",
        anchor_label=1,
        neg_label=0,
    )

    assert pair.sort_key() == (-0.92, -1.4, "mol-neg")


def test_pair_record_serialization_preserves_boolean_as_int_fields() -> None:
    pair = PairRecord(
        assay_id="CHEMBL1",
        anchor_id="mol-active",
        neg_id="mol-neg",
        sim=0.85,
        gap_abs=0.9,
        same_scaffold=False,
        pair_type="highsim_noncliff",
        anchor_label=1,
        neg_label=0,
    )

    assert pair.to_dict()["same_scaffold"] is False
