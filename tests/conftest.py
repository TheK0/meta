from __future__ import annotations

import os


# The CI/sandbox environment can trigger loky core-detection warnings even when
# the sklearn code path itself is healthy. Pin a deterministic CPU count for tests.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
