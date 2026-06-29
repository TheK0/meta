from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

NAME_TO_MODEL_CLS = {
    "randomForest": RandomForestClassifier,
    "kNN": KNeighborsClassifier,
}

DEFAULT_GRID_SEARCH = {
    "randomForest": {"n_estimators": [100]},
    "kNN": {"n_neighbors": [1, 3, 5]},
}


def test(*args, **kwargs):
    return None
