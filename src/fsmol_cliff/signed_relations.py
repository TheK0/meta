from __future__ import annotations

from collections.abc import Mapping


def build_pair_relation_dataset(
    *,
    episode: Mapping[str, object],
    assay_context: Mapping[str, object],
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Build support-pair relation training data from cliff/noncliff annotations.

    Returns (pair_ids, anchor_mols, neighbor_mols, feature_dicts).
    Labels are assigned as:
      - "flip": cliff pair or discordant highsim_noncliff (anchor label != neg label)
      - "same": highsim_noncliff with concordant labels (anchor label == neg label)

    Pairs not in assay_context["cliff_pairs"] or ["noncliff_pairs"] are NOT included.
    """
    support_pos = [str(mid) for mid in episode["support_pos_ids"]]
    support_neg = [str(mid) for mid in episode["support_neg_ids"]]
    support_set = set(support_pos + support_neg)
    support_labels: dict[str, int] = {}
    for mid in support_pos:
        support_labels[mid] = 1
    for mid in support_neg:
        support_labels[mid] = 0

    pair_ids: list[str] = []
    anchors: list[str] = []
    neighbors: list[str] = []
    features: list[dict] = []
    seen: set[tuple[str, str]] = set()

    _collect_pairs(
        assay_context.get("cliff_pairs", []),
        support_set,
        support_labels,
        pair_ids,
        anchors,
        neighbors,
        features,
        seen,
        default_relation="flip",
    )
    _collect_pairs(
        assay_context.get("noncliff_pairs", []),
        support_set,
        support_labels,
        pair_ids,
        anchors,
        neighbors,
        features,
        seen,
        default_relation=None,
    )
    return pair_ids, anchors, neighbors, features


def _collect_pairs(
    pairs: list[dict],
    support_set: set[str],
    support_labels: dict[str, int],
    pair_ids: list[str],
    anchors: list[str],
    neighbors: list[str],
    features: list[dict],
    seen: set[tuple[str, str]],
    *,
    default_relation: str | None,
) -> None:
    for pair in pairs:
        anchor_id = str(pair["anchor_id"])
        neg_id = str(pair["neg_id"])
        if anchor_id not in support_set and neg_id not in support_set:
            continue
        key = (anchor_id, neg_id) if anchor_id < neg_id else (neg_id, anchor_id)
        if key in seen:
            continue
        seen.add(key)

        if default_relation is not None:
            relation = default_relation
        else:
            label_a = support_labels.get(anchor_id)
            label_n = support_labels.get(neg_id)
            if label_a is not None and label_n is not None:
                relation = "same" if label_a == label_n else "flip"
            else:
                relation = "same"

        pair_ids.append(relation)
        anchors.append(anchor_id)
        neighbors.append(neg_id)
        features.append(
            {
                "sim": float(pair.get("sim", 0.0)),
                "gap_abs": float(pair.get("gap_abs", 0.0)),
                "same_scaffold": bool(pair.get("same_scaffold", False)),
                "pair_type": str(pair.get("pair_type", "")),
            }
        )
