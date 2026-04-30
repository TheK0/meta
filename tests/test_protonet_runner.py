from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import torch

from fsmol_cliff.io import write_jsonl, write_parquet


def test_load_protonet_model_uses_weights_only_false_for_legacy_checkpoint(monkeypatch) -> None:
    from fsmol_cliff.protonet_runner import load_protonet_model

    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.loaded = None
            self.moved_to = None
            self.eval_called = False

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    def fake_torch_load(path, *, map_location, weights_only):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {
            "model_config": {"used_features": "ecfp"},
            "model_state_dict": {"weight": 1.0},
        }

    monkeypatch.setattr("fsmol_cliff.protonet_runner.install_fs_mol_compat_patches", lambda: None)
    monkeypatch.setattr("fsmol_cliff.protonet_runner._get_protonet_trainer_class", lambda: FakeTrainer)
    monkeypatch.setattr("fsmol_cliff.protonet_runner.torch.load", fake_torch_load)

    model = load_protonet_model(Path("/tmp/protonet.pt"))

    assert captured["path"] == Path("/tmp/protonet.pt")
    assert captured["weights_only"] is False
    assert model.loaded == {"weight": 1.0}
    assert model.eval_called is True


def test_load_protonet_model_warms_mps_runtime_before_loading_checkpoint(monkeypatch) -> None:
    from fsmol_cliff.protonet_runner import load_protonet_model

    events: list[tuple[str, object]] = []

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state_dict):
            events.append(("load_state_dict", state_dict))

        def to(self, device):
            events.append(("model_to", device))
            return self

        def eval(self):
            events.append(("eval", None))
            return self

    def fake_torch_load(path, *, map_location, weights_only):
        events.append(("torch_load", map_location))
        return {
            "model_config": {"used_features": "ecfp"},
            "model_state_dict": {"weight": 1.0},
        }

    monkeypatch.setattr("fsmol_cliff.protonet_runner.install_fs_mol_compat_patches", lambda: None)
    monkeypatch.setattr("fsmol_cliff.protonet_runner._get_protonet_trainer_class", lambda: FakeTrainer)
    monkeypatch.setattr("fsmol_cliff.protonet_runner.torch.load", fake_torch_load)
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.torch.ones",
        lambda *args, **kwargs: events.append(("warmup_tensor", kwargs.get("device"))) or object(),
    )
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.torch.mps.empty_cache",
        lambda: events.append(("empty_cache", None)),
    )

    load_protonet_model(Path("/tmp/protonet.pt"), device="mps")

    assert events[:3] == [
        ("warmup_tensor", "mps"),
        ("empty_cache", None),
        ("torch_load", "mps"),
    ]


@dataclass
class _FakeBatch:
    num_query_samples: int


@dataclass
class _FakePNTaskSample:
    batches: list[_FakeBatch]


def test_score_protonet_target_ids_maps_batch_probabilities_back_in_target_order(monkeypatch) -> None:
    from fsmol_cliff.protonet_runner import score_protonet_target_ids

    fake_pn_task_sample = _FakePNTaskSample(batches=[_FakeBatch(num_query_samples=2), _FakeBatch(num_query_samples=1)])

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.array(self._values, dtype=float)

    class FakeModel:
        def __init__(self):
            self.device = "cpu"
            self.calls = []

        def eval(self):
            return self

        def __call__(self, batch):
            self.calls.append(batch.num_query_samples)
            if batch.num_query_samples == 2:
                return "two"
            return "one"

    softmax_outputs = {
        "two": FakeTensor([[0.2, 0.8], [0.7, 0.3]]),
        "one": FakeTensor([[0.1, 0.9]]),
    }

    monkeypatch.setattr("fsmol_cliff.protonet_runner.get_protonet_batcher", lambda max_num_graphs: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.task_sample_to_pn_task_sample", lambda task_sample, batcher: fake_pn_task_sample)
    monkeypatch.setattr("fsmol_cliff.protonet_runner.torchify", lambda value, device: value)
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.torch.nn.functional.softmax",
        lambda logits, dim: softmax_outputs[logits],
    )

    model = FakeModel()
    scores = score_protonet_target_ids(
        model=model,
        task_id="CHEMBL1",
        sample_map={"s1": object(), "s2": object(), "q1": object(), "q2": object(), "q3": object()},
        support_ids=["s1", "s2"],
        target_ids=["q1", "q2", "q3"],
        batch_size=320,
    )

    assert model.calls == [2, 1]
    assert scores == {"q1": 0.8, "q2": 0.3, "q3": 0.9}


@dataclass
class _FakeFeatureBlock:
    fingerprints: torch.Tensor
    descriptors: torch.Tensor


@dataclass
class _FakeTypedBatch:
    support_features: _FakeFeatureBlock
    query_features: _FakeFeatureBlock
    num_query_samples: int


def test_score_protonet_target_ids_casts_feature_tensors_to_float(monkeypatch) -> None:
    from fsmol_cliff.protonet_runner import score_protonet_target_ids

    fake_pn_task_sample = _FakePNTaskSample(
        batches=[
            _FakeTypedBatch(
                support_features=_FakeFeatureBlock(
                    fingerprints=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
                    descriptors=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                ),
                query_features=_FakeFeatureBlock(
                    fingerprints=torch.tensor([[1, 1]], dtype=torch.int32),
                    descriptors=torch.tensor([[5, 6]], dtype=torch.int32),
                ),
                num_query_samples=1,
            )
        ]
    )

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.array(self._values, dtype=float)

    class FakeModel:
        def __init__(self):
            self.device = "cpu"

        def eval(self):
            return self

        def __call__(self, batch):
            assert batch.support_features.fingerprints.dtype == torch.float32
            assert batch.support_features.descriptors.dtype == torch.float32
            assert batch.query_features.fingerprints.dtype == torch.float32
            assert batch.query_features.descriptors.dtype == torch.float32
            return "one"

    monkeypatch.setattr("fsmol_cliff.protonet_runner.get_protonet_batcher", lambda max_num_graphs: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.task_sample_to_pn_task_sample", lambda task_sample, batcher: fake_pn_task_sample)
    monkeypatch.setattr("fsmol_cliff.protonet_runner.torchify", lambda value, device: value)
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.torch.nn.functional.softmax",
        lambda logits, dim: FakeTensor([[0.1, 0.9]]),
    )

    scores = score_protonet_target_ids(
        model=FakeModel(),
        task_id="CHEMBL1",
        sample_map={"s1": object(), "s2": object(), "q1": object()},
        support_ids=["s1", "s2"],
        target_ids=["q1"],
        batch_size=320,
    )

    assert scores == {"q1": 0.9}


def test_evaluate_release_with_protonet_writes_rows_and_uses_support_scores_for_sq_psr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fsmol_cliff.runner import evaluate_release_with_protonet

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC"},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC"},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC"},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC"},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_adversarial.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "adversarial",
                "episode_id": 1,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [
                    {
                        "assay_id": "CHEMBL1",
                        "anchor_id": "a1",
                        "neg_id": "qn",
                        "sim": 0.95,
                        "gap_abs": 1.4,
                        "same_scaffold": True,
                        "pair_type": "cliff",
                        "anchor_label": 1,
                        "neg_label": 0,
                    }
                ],
            }
        ],
    )

    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_protonet_model", lambda checkpoint_path, device=None: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_task_sample_map", lambda data_dir, task_id: {"dummy": object()})

    def fake_score_episode(*, episode, **kwargs):
        if episode["split_type"] == "adversarial":
            return {"a1": 0.9, "n1": 0.1, "qa": 0.8, "qn": 0.2}
        return {"qa": 0.8, "qn": 0.2}

    monkeypatch.setattr("fsmol_cliff.protonet_runner.score_protonet_manifest_episode", fake_score_episode)

    output_path = tmp_path / "task_results_protonet.parquet"
    rows = evaluate_release_with_protonet(
        release_dir=release_dir,
        data_dir=tmp_path / "data",
        checkpoint_path=tmp_path / "pn.pt",
        output_path=output_path,
        split_types=("standard", "adversarial"),
    )

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    q_psr_row = next(row for row in rows if row["split_type"] == "standard" and row["metric"] == "q_psr")
    sq_psr_row = next(row for row in rows if row["split_type"] == "adversarial" and row["metric"] == "sq_psr")
    assert q_psr_row["score"] == 1.0
    assert sq_psr_row["score"] == 1.0
    assert set(saved["metric"]) >= {"average_precision_score", "q_psr", "sq_psr"}


def test_evaluate_release_with_protonet_supports_identity_calibration_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fsmol_cliff.evaluation import summarize_task_metric_rows as summarize_task_metric_rows_impl
    from fsmol_cliff.runner import evaluate_release_with_protonet

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC"},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC"},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC"},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC"},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )

    captured: dict[str, list[dict]] = {}

    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_protonet_model", lambda checkpoint_path, device=None: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_task_sample_map", lambda data_dir, task_id: {"dummy": object()})
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.score_protonet_manifest_episode",
        lambda **kwargs: {
            "raw_scores": {"qa": 0.5, "qn": 0.49},
            "calibrated_scores": {"qa": 0.5, "qn": 0.49},
            "raw_margins": {"qa": 0.0, "qn": -0.01},
            "calibrated_margins": {"qa": 0.0, "qn": -0.01},
        },
    )
    def capture_and_summarize(episode_results):
        captured["episode_results"] = episode_results
        return summarize_task_metric_rows_impl(episode_results)

    monkeypatch.setattr(
        "fsmol_cliff.runner.summarize_task_metric_rows",
        capture_and_summarize,
    )

    output_path = tmp_path / "task_results_protonet.parquet"
    rows = evaluate_release_with_protonet(
        release_dir=release_dir,
        data_dir=tmp_path / "data",
        checkpoint_path=tmp_path / "pn.pt",
        output_path=output_path,
        split_types=("standard",),
    )

    assert output_path.exists()
    q_psr_row = next(row for row in rows if row["split_type"] == "standard" and row["metric"] == "q_psr")
    assert q_psr_row["score"] == 1.0
    assert captured["episode_results"][0]["score_bundle"]["raw_scores"] == {"qa": 0.5, "qn": 0.49}
    assert captured["episode_results"][0]["score_bundle"]["calibrated_scores"] == {"qa": 0.5, "qn": 0.49}
    assert captured["episode_results"][0]["score_bundle"]["raw_margins"] == {"qa": 0.0, "qn": -0.01}
    assert captured["episode_results"][0]["score_bundle"]["calibrated_margins"] == {"qa": 0.0, "qn": -0.01}


def test_evaluate_release_with_protonet_query_only_calibration_uses_calibrated_scores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fsmol_cliff.runner import evaluate_release_with_protonet

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC"},
            {"molecule_id": "a2", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 7.8, "scaffold_smiles": "CC"},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC"},
            {"molecule_id": "n2", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.2, "scaffold_smiles": "CC"},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCCl", "label": 1, "r": 8.1, "scaffold_smiles": "CC"},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCBr", "label": 0, "r": 6.1, "scaffold_smiles": "CC"},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            },
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "a1",
                "neg_id": "n1",
                "sim": 0.94,
                "gap_abs": 1.2,
                "same_scaffold": False,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            },
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1", "a2"],
                "support_neg_ids": ["n1", "n2"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )

    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_protonet_model", lambda checkpoint_path, device=None: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_task_sample_map", lambda data_dir, task_id: {"dummy": object()})

    def fake_target_scores(*, target_ids, **kwargs):
        lookup = {
            "a1": 0.82,
            "a2": 0.74,
            "n1": 0.24,
            "n2": 0.31,
            "qa": 0.52,
            "qn": 0.48,
        }
        return {molecule_id: lookup[molecule_id] for molecule_id in target_ids}

    monkeypatch.setattr("fsmol_cliff.protonet_runner.score_protonet_target_ids", fake_target_scores)

    output_path = tmp_path / "task_results_protonet.parquet"
    rows = evaluate_release_with_protonet(
        release_dir=release_dir,
        data_dir=tmp_path / "data",
        checkpoint_path=tmp_path / "pn.pt",
        output_path=output_path,
        split_types=("standard",),
        calibration_mode="query_only",
    )

    assert output_path.exists()
    q_psr_row = next(row for row in rows if row["split_type"] == "standard" and row["metric"] == "q_psr")
    assert q_psr_row["score"] == 1.0


def test_runner_evaluate_release_with_protonet_passes_assay_context_to_query_only_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fsmol_cliff.runner import evaluate_release_with_protonet

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC"},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC"},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC"},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC"},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )

    monkeypatch.setattr("fsmol_cliff.runner.load_protonet_model", lambda checkpoint_path, device=None: object())
    monkeypatch.setattr("fsmol_cliff.runner.load_fsmol_task_sample_map", lambda data_dir, task_id: {"dummy": object()})

    def fake_score_protonet_episode(*, assay_context=None, **kwargs):
        assert assay_context is not None
        return {
            "raw_scores": {"qa": 0.5, "qn": 0.49},
            "calibrated_scores": {"qa": 0.5, "qn": 0.49},
            "raw_margins": {"qa": 0.0, "qn": -0.01},
            "calibrated_margins": {"qa": 0.0, "qn": -0.01},
        }

    monkeypatch.setattr("fsmol_cliff.runner.score_protonet_episode", fake_score_protonet_episode)

    output_path = tmp_path / "task_results_runner_protonet.parquet"
    rows = evaluate_release_with_protonet(
        release_dir=release_dir,
        data_dir=tmp_path / "data",
        checkpoint_path=tmp_path / "pn.pt",
        output_path=output_path,
        split_types=("standard",),
        calibration_mode="query_only",
    )

    assert output_path.exists()
    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    assert q_psr_row["score"] == 1.0


def test_evaluate_release_with_protonet_reads_profile_specific_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fsmol_cliff.runner import evaluate_release_with_protonet

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC"},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC"},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC"},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC"},
        ],
    )
    write_jsonl(
        assay_dir / "pairs_relaxed.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard_relaxed.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "profile": "relaxed",
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )
    write_parquet(release_dir / "episodes_adversarial_relaxed.parquet", [])

    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_protonet_model", lambda checkpoint_path, device=None: object())
    monkeypatch.setattr("fsmol_cliff.protonet_runner.load_task_sample_map", lambda data_dir, task_id: {"dummy": object()})
    monkeypatch.setattr(
        "fsmol_cliff.protonet_runner.score_protonet_manifest_episode",
        lambda **kwargs: {"qa": 0.8, "qn": 0.2},
    )

    output_path = tmp_path / "task_results_protonet_relaxed.parquet"
    rows = evaluate_release_with_protonet(
        release_dir=release_dir,
        data_dir=tmp_path / "data",
        checkpoint_path=tmp_path / "pn.pt",
        output_path=output_path,
        split_types=("standard",),
        profile="relaxed",
        result_tier="final",
    )

    assert output_path.exists()
    assert {row["profile"] for row in rows} == {"relaxed"}
    assert {row["result_tier"] for row in rows} == {"final"}
