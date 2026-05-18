"""Persistance simple (pickle) pour les pipelines sklearn.

Joblib serait plus efficace pour les gros modèles, mais ici le pipeline
TF-IDF + LogReg tient en quelques Mo : pickle stdlib suffit.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def save_model(model: Any, path: PathLike) -> Path:
    """Sérialise `model` en pickle sous `path`. Crée les dossiers parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: PathLike) -> Any:
    """Charge un modèle pickle. Lève FileNotFoundError si absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)
