from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from statistics import mean, pstdev


def apply_boundary_uncertainty_calibration(
    *,
    episode: Mapping[str, object],
    assay_context: Mapping[str, object],
    raw_scores: Mapping[str, float],
    raw_margins: Mapping[str, float],
    top_k: int = 2,
    uncertainty_scale: float = 0.1,
    margin_floor: float = 0.1,
) -> dict[str, object]:
    scores = {molecule_id: float(score) for molecule_id, score in raw_scores.items()}
    margins = {molecule_id: float(margin) for molecule_id, margin in raw_margins.items()}
    support_pos_ids = list(episode["support_pos_ids"])
    support_neg_ids = list(episode["support_neg_ids"])
    support_ids = [*support_pos_ids, *support_neg_ids]
    support_labels = {
        molecule_id: 1 if molecule_id in support_pos_ids else 0
        for molecule_id in support_ids
    }

    pos_support_margins = [margins[mid] for mid in support_pos_ids if mid in margins]
    neg_support_margins = [margins[mid] for mid in support_neg_ids if mid in margins]
    pos_center = mean(pos_support_margins) if pos_support_margins else 0.0
    neg_center = mean(neg_support_margins) if neg_support_margins else 0.0
    prototype_margin = abs(pos_center - neg_center)
    support_dispersion = _safe_pstdev(pos_support_margins) + _safe_pstdev(neg_support_margins)
    effective_margin_scale = max(float(margin_floor), prototype_margin, 1e-6)
    dispersion_ratio = min(1.0, support_dispersion / effective_margin_scale)

    # Use cached neighbour index built once per assay (lazy, stored in assay_context).
    molecule_index = _get_assay_molecule_index(assay_context)
    support_neighbors = _lookup_support_neighbors(
        molecule_index=molecule_index,
        support_labels=support_labels,
        top_k=max(1, int(top_k)),
    )

    calibrated_scores: dict[str, float] = {}
    calibrated_margins: dict[str, float] = {}
    uncertainty_summary: dict[str, dict[str, float]] = {}
    for molecule_id, raw_margin in margins.items():
        local_ambiguity = max(0.0, 1.0 - min(abs(raw_margin) / effective_margin_scale, 1.0))
        neighborhood_disagreement = _neighbor_disagreement(
            molecule_id=molecule_id,
            support_labels=support_labels,
            labels=assay_context["labels"],
            support_neighbors=support_neighbors,
        )
        composite_uncertainty = min(
            1.0,
            (local_ambiguity + dispersion_ratio + neighborhood_disagreement) / 3.0,
        )
        shrink_factor = max(0.0, 1.0 - float(uncertainty_scale) * composite_uncertainty)
        calibrated_margin = raw_margin * shrink_factor
        calibrated_score = min(1.0, max(0.0, 0.5 + calibrated_margin))

        calibrated_margins[molecule_id] = float(calibrated_margin)
        calibrated_scores[molecule_id] = float(calibrated_score)
        uncertainty_summary[molecule_id] = {
            "prototype_margin": float(prototype_margin),
            "support_dispersion": float(support_dispersion),
            "local_ambiguity": float(local_ambiguity),
            "neighborhood_disagreement": float(neighborhood_disagreement),
            "composite_uncertainty": float(composite_uncertainty),
            "shrink_factor": float(shrink_factor),
        }

    return {
        "raw_scores": scores,
        "calibrated_scores": calibrated_scores,
        "raw_margins": margins,
        "calibrated_margins": calibrated_margins,
        "uncertainty_summary": uncertainty_summary,
    }


def _safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


# ---------------------------------------------------------------------------
# Pre-indexed molecule neighbour lookup (built once per assay, reused across
# all episodes for that assay).  Stored lazily in assay_context["_neighbor_index"].
# ---------------------------------------------------------------------------

def _get_assay_molecule_index(
    assay_context: Mapping[str, object],
) -> dict[str, list[tuple[float, str]]]:
    """Return (or build once and cache) the full molecule → [(sim, neighbor_id), ...] index."""
    cache_key = "_neighbor_index"
    cached = assay_context.get(cache_key)  # type: ignore[union-attr]
    if cached is not None:
        return cached  # type: ignore[return-value]
    by_molecule: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for pair in [*assay_context.get("cliff_pairs", []), *assay_context.get("noncliff_pairs", [])]:
        sim = float(pair.get("sim", 0.0))
        anchor_id = str(pair["anchor_id"])
        neg_id = str(pair["neg_id"])
        by_molecule[anchor_id].append((sim, neg_id))
        by_molecule[neg_id].append((sim, anchor_id))
    # Sort each list once; per-episode filtering does not re-sort.
    index: dict[str, list[tuple[float, str]]] = {}
    for mid, neighbors in by_molecule.items():
        index[mid] = sorted(set(neighbors), reverse=True)
    assay_context[cache_key] = index  # type: ignore[index]
    return index


def _lookup_support_neighbors(
    *,
    molecule_index: Mapping[str, list[tuple[float, str]]],
    support_labels: Mapping[str, int],
    top_k: int,
) -> dict[str, list[str]]:
    """Build per-molecule top-k support-only neighbour lists from the precomputed index."""
    support_set = frozenset(support_labels)
    result: dict[str, list[str]] = {}
    for molecule_id, neighbors in molecule_index.items():
        filtered: list[str] = []
        for _, neighbor_id in neighbors:
            if neighbor_id not in support_set:
                continue
            if neighbor_id in filtered:
                continue
            filtered.append(neighbor_id)
            if len(filtered) >= top_k:
                break
        if filtered:
            result[molecule_id] = filtered
    return result


# ---------------------------------------------------------------------------
# Neighbour disagreement (unchanged semantics)
# ---------------------------------------------------------------------------

def _neighbor_disagreement(
    *,
    molecule_id: str,
    support_labels: Mapping[str, int],
    labels: Mapping[str, int],
    support_neighbors: Mapping[str, list[str]],
) -> float:
    neighbors = [
        neighbor_id
        for neighbor_id in support_neighbors.get(molecule_id, [])
        if neighbor_id in support_labels
    ]
    if not neighbors or molecule_id not in labels:
        return 0.0
    molecule_label = int(labels[molecule_id])
    disagreements = sum(
        int(support_labels[neighbor_id] != molecule_label) for neighbor_id in neighbors
    )
    return disagreements / len(neighbors)
