from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _SimpleBatch:
    num_query_samples: int


@dataclass
class _SimplePNTaskSample:
    batches: list[_SimpleBatch]


class _SimpleBatcher:
    def __init__(self, max_num_graphs: int):
        self._max_num_graphs = max_num_graphs

    def batch(self, samples):
        labels = [int(getattr(sample, "bool_label", False)) for sample in samples]
        batch = type(
            "SimpleFeatureBatch",
            (),
            {
                "num_graphs": len(samples),
                "num_query_samples": len(samples),
            },
        )()
        yield batch, labels


def get_protonet_batcher(max_num_graphs: int):
    return _SimpleBatcher(max_num_graphs=max_num_graphs)


def task_sample_to_pn_task_sample(task_sample, batcher):
    return _SimplePNTaskSample(batches=[_SimpleBatch(num_query_samples=len(task_sample.test_samples))])
