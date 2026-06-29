from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphFeatureExtractorConfig:
    hidden_dim: int = 128


class GraphFeatureExtractor:
    def __init__(self, config: GraphFeatureExtractorConfig):
        self.config = config


def add_graph_feature_extractor_arguments(parser):
    return parser


def make_graph_feature_extractor_config_from_args(args):
    return GraphFeatureExtractorConfig()
