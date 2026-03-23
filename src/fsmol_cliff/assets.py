from __future__ import annotations

import math
from collections import OrderedDict
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from .chem import canonicalize_isomeric_smiles, murcko_scaffold_smiles, tanimoto_similarity
from .constants import DEFAULT_PROTOCOL_CONSTANTS
from .models import PairRecord

_CANONICAL_SMILES_KEYS = (
    "canonical_isomeric_smiles",
    "canonical-isomeric-smiles",
    "CanonicalIsomericSmiles",
    "CanonicalIsomericSMILES",
)
_RAW_SMILES_KEYS = ("smiles", "SMILES", "Smiles")
_RELATION_KEYS = ("relation", "Relation")
_QUALIFIER_KEYS = ("censoring_qualifier", "CensoringQualifier", "qualifier", "Qualifier")
_R_KEYS = ("r", "LogRegressionProperty", "log_regression_property")
_LABEL_KEYS = ("label", "Label", "Y", "y", "Property")
_MOLECULE_ID_KEYS = ("molecule_id", "Molecule_ID", "compound_id", "Compound_ID", "id", "mol_id")
_SCAFFOLD_KEYS = ("scaffold", "murcko_scaffold", "scaffold_smiles")


def filter_assay_records(assay_id: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    for index, record in enumerate(records):
        if not _is_precise_measurement(record):
            continue

        r_value = _coerce_finite_float(_first_present(record, _R_KEYS))
        if r_value is None:
            continue

        canonical_smiles = _resolve_canonical_smiles(record)
        if canonical_smiles is None:
            continue

        label = _coerce_label(_first_present(record, _LABEL_KEYS))
        if label is None:
            continue

        molecule_id = _coerce_molecule_id(record, assay_id=assay_id, index=index)
        normalized = {
            "assay_id": assay_id,
            "molecule_id": molecule_id,
            "canonical_isomeric_smiles": canonical_smiles,
            "label": label,
            "r": r_value,
        }
        groups.setdefault(canonical_smiles, []).append(normalized)

    collapsed: list[dict[str, Any]] = []
    for canonical_smiles, group in groups.items():
        labels = {record["label"] for record in group}
        if len(labels) != 1:
            continue

        values = [record["r"] for record in group]
        if max(values) - min(values) > 0.5:
            continue

        collapsed.append(
            {
                "assay_id": assay_id,
                "molecule_id": group[0]["molecule_id"],
                "canonical_isomeric_smiles": canonical_smiles,
                "label": group[0]["label"],
                "r": float(median(values)),
                "source_ids": [record["molecule_id"] for record in group],
            }
        )

    return collapsed


def mine_assay_pairs(
    assay_id: str,
    actives: Sequence[Mapping[str, Any]],
    inactives: Sequence[Mapping[str, Any]],
    *,
    pair_similarity: Mapping[tuple[str, str], float] | Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
    tau: float = DEFAULT_PROTOCOL_CONSTANTS.similarity_threshold,
    delta: float = DEFAULT_PROTOCOL_CONSTANTS.activity_gap_threshold,
    hard_negative_pool_size: int = DEFAULT_PROTOCOL_CONSTANTS.hard_negative_pool_size,
) -> dict[str, Any]:
    highsim_pairs: list[dict[str, Any]] = []
    cliff_pairs: list[dict[str, Any]] = []
    noncliff_pairs: list[dict[str, Any]] = []
    same_scaffold_pairs: list[dict[str, Any]] = []
    same_scaffold_cliff_pairs: list[dict[str, Any]] = []
    same_scaffold_noncliff_pairs: list[dict[str, Any]] = []
    hard_negative_pools: dict[str, list[dict[str, Any]]] = {}

    for active in actives:
        anchor_pairs: list[dict[str, Any]] = []
        anchor_id = _coerce_required_text(active.get("molecule_id")) or "<missing-active-id>"
        anchor_r = float(active["r"])

        for inactive in inactives:
            neg_id = _coerce_required_text(inactive.get("molecule_id")) or "<missing-inactive-id>"
            neg_r = float(inactive["r"])
            sim = _resolve_similarity(active, inactive, pair_similarity)
            if sim < tau:
                continue

            same_scaffold = _resolve_same_scaffold(active, inactive)
            pair_type = "cliff" if abs(anchor_r - neg_r) >= delta else "highsim_noncliff"
            pair = PairRecord(
                assay_id=assay_id,
                anchor_id=anchor_id,
                neg_id=neg_id,
                sim=float(sim),
                gap_abs=float(abs(anchor_r - neg_r)),
                same_scaffold=same_scaffold,
                pair_type=pair_type,
                anchor_label=1,
                neg_label=0,
            ).to_dict()

            highsim_pairs.append(pair)
            anchor_pairs.append(pair)
            if pair_type == "cliff":
                cliff_pairs.append(pair)
            else:
                noncliff_pairs.append(pair)
            if same_scaffold:
                same_scaffold_pairs.append(pair)
                if pair_type == "cliff":
                    same_scaffold_cliff_pairs.append(pair)
                else:
                    same_scaffold_noncliff_pairs.append(pair)

        anchor_pairs.sort(key=_pair_sort_key)
        if anchor_pairs:
            hard_negative_pools[anchor_id] = anchor_pairs[:hard_negative_pool_size]

    highsim_pairs = _sort_pair_group(highsim_pairs)
    cliff_pairs = _sort_pair_group(cliff_pairs)
    noncliff_pairs = _sort_pair_group(noncliff_pairs)
    same_scaffold_pairs = _sort_pair_group(same_scaffold_pairs)
    same_scaffold_cliff_pairs = _sort_pair_group(same_scaffold_cliff_pairs)
    same_scaffold_noncliff_pairs = _sort_pair_group(same_scaffold_noncliff_pairs)

    diagnostics = _build_diagnostics(
        tau=tau,
        delta=delta,
        hard_negative_pool_size=hard_negative_pool_size,
        actives=actives,
        inactives=inactives,
        highsim_pairs=highsim_pairs,
        cliff_pairs=cliff_pairs,
        noncliff_pairs=noncliff_pairs,
        same_scaffold_pairs=same_scaffold_pairs,
        same_scaffold_cliff_pairs=same_scaffold_cliff_pairs,
        hard_negative_pools=hard_negative_pools,
    )

    return {
        "assay_id": assay_id,
        "pairs": {
            "highsim_discordant": highsim_pairs,
            "cliff": cliff_pairs,
            "highsim_noncliff": noncliff_pairs,
            "same_scaffold": same_scaffold_pairs,
            "same_scaffold_cliff": same_scaffold_cliff_pairs,
            "same_scaffold_noncliff": same_scaffold_noncliff_pairs,
        },
        "hard_negative_pools": hard_negative_pools,
        "diagnostics": diagnostics,
    }


def build_assay_assets(
    assay_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    pair_similarity: Mapping[tuple[str, str], float] | Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
    tau: float = DEFAULT_PROTOCOL_CONSTANTS.similarity_threshold,
    delta: float = DEFAULT_PROTOCOL_CONSTANTS.activity_gap_threshold,
    hard_negative_pool_size: int = DEFAULT_PROTOCOL_CONSTANTS.hard_negative_pool_size,
) -> dict[str, Any]:
    molecules = filter_assay_records(assay_id, records)
    actives = [record for record in molecules if record["label"] == 1]
    inactives = [record for record in molecules if record["label"] == 0]
    mined = mine_assay_pairs(
        assay_id,
        actives,
        inactives,
        pair_similarity=pair_similarity,
        tau=tau,
        delta=delta,
        hard_negative_pool_size=hard_negative_pool_size,
    )
    return {
        "assay_id": assay_id,
        "molecules": molecules,
        "actives": actives,
        "inactives": inactives,
        "pairs": mined["pairs"],
        "hard_negative_pools": mined["hard_negative_pools"],
        "diagnostics": mined["diagnostics"],
    }


def _build_diagnostics(
    *,
    tau: float,
    delta: float,
    hard_negative_pool_size: int,
    actives: Sequence[Mapping[str, Any]],
    inactives: Sequence[Mapping[str, Any]],
    highsim_pairs: Sequence[dict[str, Any]],
    cliff_pairs: Sequence[dict[str, Any]],
    noncliff_pairs: Sequence[dict[str, Any]],
    same_scaffold_pairs: Sequence[dict[str, Any]],
    same_scaffold_cliff_pairs: Sequence[dict[str, Any]],
    hard_negative_pools: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_pairs = len(actives) * len(inactives)
    n_highsim = len(highsim_pairs)
    all_pair_gaps = [float(pair["gap_abs"]) for pair in highsim_pairs]
    all_pair_sims = [float(pair["sim"]) for pair in highsim_pairs]
    greater_count = sum(1 for active in actives for inactive in inactives if active["r"] > inactive["r"])
    lower_count = sum(1 for active in actives for inactive in inactives if active["r"] < inactive["r"])
    directional_denominator = len(actives) * len(inactives)

    return {
        "tau": tau,
        "delta": delta,
        "hard_negative_pool_size": hard_negative_pool_size,
        "n_actives": len(actives),
        "n_inactives": len(inactives),
        "n_molecules": len(actives) + len(inactives),
        "n_candidate_pairs": candidate_pairs,
        "n_highsim_discordant": n_highsim,
        "n_cliff": len(cliff_pairs),
        "n_highsim_noncliff": len(noncliff_pairs),
        "n_same_scaffold": len(same_scaffold_pairs),
        "n_highsim_active_anchors": len({pair["anchor_id"] for pair in highsim_pairs}),
        "n_cliff_active_anchors": len({pair["anchor_id"] for pair in cliff_pairs}),
        "n_same_scaffold_active_anchors": len({pair["anchor_id"] for pair in same_scaffold_pairs}),
        "n_hard_negative_anchors": len(hard_negative_pools),
        "n_hard_negative_pairs": sum(len(pool) for pool in hard_negative_pools.values()),
        "highsim_pair_fraction": (n_highsim / candidate_pairs) if candidate_pairs else 0.0,
        "cliff_fraction_within_highsim": (len(cliff_pairs) / n_highsim) if n_highsim else 0.0,
        "same_scaffold_fraction_within_highsim": (len(same_scaffold_pairs) / n_highsim) if n_highsim else 0.0,
        "num_highsim_discordant_pairs": n_highsim,
        "num_cliff_pairs": len(cliff_pairs),
        "num_noncliff_highsim_pairs": len(noncliff_pairs),
        "num_same_scaffold_cliff_pairs": len(same_scaffold_cliff_pairs),
        "median_sim": float(median(all_pair_sims)) if all_pair_sims else 0.0,
        "median_gap_abs": float(median(all_pair_gaps)) if all_pair_gaps else 0.0,
        "frac_pairs_with_r_active_gt_r_inactive": (
            greater_count / directional_denominator if directional_denominator else 0.0
        ),
        "frac_pairs_with_r_active_lt_r_inactive": (
            lower_count / directional_denominator if directional_denominator else 0.0
        ),
    }


def _is_precise_measurement(record: Mapping[str, Any]) -> bool:
    relation = _coerce_optional_text(_first_present(record, _RELATION_KEYS))
    qualifier = _coerce_optional_text(_first_present(record, _QUALIFIER_KEYS))
    return qualifier is None and relation in (None, "=")


def _resolve_canonical_smiles(record: Mapping[str, Any]) -> str | None:
    canonical = _coerce_optional_text(_first_present(record, _CANONICAL_SMILES_KEYS))
    if canonical is not None:
        return canonicalize_isomeric_smiles(canonical)
    raw_smiles = _coerce_optional_text(_first_present(record, _RAW_SMILES_KEYS))
    return canonicalize_isomeric_smiles(raw_smiles)


def _resolve_same_scaffold(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_scaffold = _coerce_optional_text(_first_present(left, _SCAFFOLD_KEYS))
    right_scaffold = _coerce_optional_text(_first_present(right, _SCAFFOLD_KEYS))
    if left_scaffold is None:
        left_scaffold = murcko_scaffold_smiles(_coerce_optional_text(left.get("canonical_isomeric_smiles")))
    if right_scaffold is None:
        right_scaffold = murcko_scaffold_smiles(_coerce_optional_text(right.get("canonical_isomeric_smiles")))
    return bool(left_scaffold and right_scaffold and left_scaffold == right_scaffold)


def _resolve_similarity(
    active: Mapping[str, Any],
    inactive: Mapping[str, Any],
    pair_similarity: Mapping[tuple[str, str], float] | Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None,
) -> float:
    if callable(pair_similarity):
        return float(pair_similarity(active, inactive))

    if pair_similarity is not None:
        active_id = _coerce_optional_text(active.get("molecule_id"))
        inactive_id = _coerce_optional_text(inactive.get("molecule_id"))
        active_smiles = _coerce_optional_text(active.get("canonical_isomeric_smiles"))
        inactive_smiles = _coerce_optional_text(inactive.get("canonical_isomeric_smiles"))
        for key in (
            (active_id, inactive_id),
            (inactive_id, active_id),
            (active_smiles, inactive_smiles),
            (inactive_smiles, active_smiles),
        ):
            if key in pair_similarity:
                return float(pair_similarity[key])

    similarity = tanimoto_similarity(
        _coerce_optional_text(active.get("canonical_isomeric_smiles")),
        _coerce_optional_text(inactive.get("canonical_isomeric_smiles")),
    )
    if similarity is None:
        raise ValueError("pair_similarity is required when RDKit similarity cannot be computed")
    return similarity


def _sort_pair_group(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(pairs, key=lambda pair: (pair["anchor_id"],) + _pair_sort_key(pair))


def _pair_sort_key(pair: Mapping[str, Any]) -> tuple[float, float, str]:
    return (-float(pair["sim"]), -float(pair["gap_abs"]), str(pair["neg_id"]))


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _coerce_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _coerce_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        return None
    text = str(value).strip().lower()
    if text in {"0", "inactive", "false"}:
        return 0
    if text in {"1", "active", "true"}:
        return 1
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric in (0.0, 1.0):
        return int(numeric)
    return None


def _coerce_molecule_id(record: Mapping[str, Any], *, assay_id: str, index: int) -> str:
    molecule_id = _coerce_optional_text(_first_present(record, _MOLECULE_ID_KEYS))
    if molecule_id is not None:
        return molecule_id
    return f"{assay_id}:{index}"


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_required_text(value: Any) -> str | None:
    return _coerce_optional_text(value)


__all__ = [
    "build_assay_assets",
    "filter_assay_records",
    "mine_assay_pairs",
]
