from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DataFold(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class StratifiedTaskSampler:
    def __init__(self, train_size_or_ratio: int, test_size_or_ratio: int):
        self.train_size_or_ratio = int(train_size_or_ratio)
        self.test_size_or_ratio = int(test_size_or_ratio)

    def sample(self, task, seed: int = 0):
        train = list(task.samples[: self.train_size_or_ratio])
        test = list(task.samples[self.train_size_or_ratio : self.train_size_or_ratio + self.test_size_or_ratio])
        return FSMolTaskSample(name=task.name, train_samples=train, valid_samples=[], test_samples=test)


from .fsmol_task import FSMolTaskSample  # noqa: E402

__all__ = ["DataFold", "StratifiedTaskSampler"]
