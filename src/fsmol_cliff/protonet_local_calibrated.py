from __future__ import annotations

from collections.abc import Mapping
from statistics import mean, pstdev

from sklearn.linear_model import LogisticRegression


def apply_identity_local_calibration(
    *,
    raw_scores: Mapping[str, float],
    raw_margins: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    return {
        "raw_scores": {molecule_id: float(score) for molecule_id, score in raw_scores.items()},
        "calibrated_scores": {molecule_id: float(score) for molecule_id, score in raw_scores.items()},
        "raw_margins": {molecule_id: float(margin) for molecule_id, margin in raw_margins.items()},
        "calibrated_margins": {molecule_id: float(margin) for molecule_id, margin in raw_margins.items()},
    }


def apply_query_only_local_calibration(
    *,
    episode: dict,
    assay_context: Mapping[str, object],
    raw_scores: Mapping[str, float],
    raw_margins: Mapping[str, float],
) -> dict[str, object]:
    local_features = _build_query_only_local_features(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
    )
    support_ids = [*episode["support_pos_ids"], *episode["support_neg_ids"]]
    labels = dict(assay_context["labels"])
    X_support = [_feature_vector(local_features[molecule_id]) for molecule_id in support_ids]
    y_support = [int(labels[molecule_id]) for molecule_id in support_ids]

    if len(set(y_support)) < 2:
        bundle = apply_identity_local_calibration(raw_scores=raw_scores, raw_margins=raw_margins)
        bundle["local_features"] = local_features
        return bundle

    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=200,
    )
    model.fit(X_support, y_support)

    ordered_ids = list(raw_scores)
    calibrated_scores_list = model.predict_proba([_feature_vector(local_features[molecule_id]) for molecule_id in ordered_ids])[:, 1]
    calibrated_scores = {
        molecule_id: float(score)
        for molecule_id, score in zip(ordered_ids, calibrated_scores_list, strict=True)
    }
    calibrated_margins = {
        molecule_id: float(score) - 0.5
        for molecule_id, score in calibrated_scores.items()
    }
    return {
        "raw_scores": {molecule_id: float(score) for molecule_id, score in raw_scores.items()},
        "calibrated_scores": calibrated_scores,
        "raw_margins": {molecule_id: float(margin) for molecule_id, margin in raw_margins.items()},
        "calibrated_margins": calibrated_margins,
        "local_features": local_features,
    }


def _build_query_only_local_features(
    *,
    episode: Mapping[str, object],
    assay_context: Mapping[str, object],
    raw_scores: Mapping[str, float],
    raw_margins: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    support_pos_ids = list(episode["support_pos_ids"])
    support_neg_ids = list(episode["support_neg_ids"])
    query_ids = [*episode["query_pos_ids"], *episode["query_neg_ids"]]
    all_target_ids = [*support_pos_ids, *support_neg_ids, *query_ids]

    pos_support_scores = [float(raw_scores[molecule_id]) for molecule_id in support_pos_ids]
    neg_support_scores = [float(raw_scores[molecule_id]) for molecule_id in support_neg_ids]
    prototype_gap = mean(pos_support_scores) - mean(neg_support_scores)
    support_dispersion = pstdev(pos_support_scores) + pstdev(neg_support_scores)

    pair_rows = [*assay_context.get("cliff_pairs", []), *assay_context.get("noncliff_pairs", [])]
    cliff_edge_lookup = {
        (pair["anchor_id"], pair["neg_id"])
        for pair in assay_context.get("cliff_pairs", [])
    }

    local_features: dict[str, dict[str, float]] = {}
    for molecule_id in all_target_ids:
        cross_class_neighbors = []
        cliff_neighbors = 0
        if molecule_id in support_pos_ids or molecule_id in episode["query_pos_ids"]:
            for pair in pair_rows:
                if pair["anchor_id"] == molecule_id and pair["neg_id"] in support_neg_ids:
                    cross_class_neighbors.append(pair["neg_id"])
                    cliff_neighbors += int((pair["anchor_id"], pair["neg_id"]) in cliff_edge_lookup)
        if molecule_id in support_neg_ids or molecule_id in episode["query_neg_ids"]:
            for pair in pair_rows:
                if pair["neg_id"] == molecule_id and pair["anchor_id"] in support_pos_ids:
                    cross_class_neighbors.append(pair["anchor_id"])
                    cliff_neighbors += int((pair["anchor_id"], pair["neg_id"]) in cliff_edge_lookup)

        density = len(cross_class_neighbors) / max(len(support_pos_ids), len(support_neg_ids), 1)
        cliff_fraction = 0.0 if not cross_class_neighbors else cliff_neighbors / len(cross_class_neighbors)
        local_features[molecule_id] = {
            "raw_score": float(raw_scores[molecule_id]),
            "raw_margin": float(raw_margins[molecule_id]),
            "prototype_gap": float(prototype_gap),
            "support_dispersion": float(support_dispersion),
            "cross_class_density": float(density),
            "cross_class_cliff_fraction": float(cliff_fraction),
        }
    return local_features


def _feature_vector(features: Mapping[str, float]) -> list[float]:
    return [
        float(features["raw_score"]),
        float(features["raw_margin"]),
        float(features["prototype_gap"]),
        float(features["support_dispersion"]),
        float(features["cross_class_density"]),
        float(features["cross_class_cliff_fraction"]),
    ]
