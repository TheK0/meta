from __future__ import annotations

import gzip
import json
from pathlib import Path


class RichPath:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def create(cls, path: str | Path) -> "RichPath":
        return cls(path)

    def read_by_file_suffix(self):
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __str__(self) -> str:
        return str(self.path)
