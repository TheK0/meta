from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FSMolSample:
    smiles: str
    bool_label: bool
    fingerprint: list[float] | None = None


@dataclass
class FSMolTaskSample:
    name: str
    train_samples: list
    valid_samples: list
    test_samples: list


@dataclass
class FSMolTask:
    name: str
    samples: list[FSMolSample]

    @classmethod
    def load_from_file(cls, task_path) -> "FSMolTask":
        rows = list(task_path.read_by_file_suffix())
        name = rows[0].get("Assay_ID", "task") if rows else "task"
        samples = [
            FSMolSample(
                smiles=str(row.get("CanonicalIsomericSmiles") or row.get("SMILES") or ""),
                bool_label=bool(int(row.get("Y", row.get("Property", 0)))),
            )
            for row in rows
        ]
        return cls(name=name, samples=samples)
