"""Loader pour le dataset Allociné (reviews de films en français).

Source : https://huggingface.co/datasets/tblard/allocine

Le dataset complet contient ~200k reviews labellisées (0 = négatif, 1 = positif).
On expose un loader paramétrable qui :
  - télécharge un sous-ensemble (par défaut 2000 reviews) via `datasets.load_dataset`
  - renvoie une liste de `AllocineSample` (text, label)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

DATASET_NAME = "tblard/allocine"
DEFAULT_SUBSET = 2_000


@dataclass(frozen=True)
class AllocineSample:
    text: str
    label: int  # 0 = négatif, 1 = positif


def load_allocine(
    split: str = "train",
    subset: Optional[int] = DEFAULT_SUBSET,
    loader: Optional[Callable[..., object]] = None,
) -> List[AllocineSample]:
    """Charge un sous-ensemble du dataset Allociné.

    Args:
        split: split HF (`train`, `validation`, `test`).
        subset: si défini, ne renvoie que les `subset` premières lignes
                (utile pour itérer vite sans tout télécharger).
        loader: hook d'injection pour les tests — par défaut
                `datasets.load_dataset`. Doit renvoyer un itérable de dicts
                avec les clés `review` et `label`.

    Returns:
        Liste de `AllocineSample`.
    """
    if loader is None:
        from datasets import load_dataset  # import paresseux : datasets est lourd

        loader = load_dataset

    raw = loader(DATASET_NAME, split=split)

    samples: List[AllocineSample] = []
    for i, row in enumerate(raw):
        if subset is not None and i >= subset:
            break
        text = row["review"]
        label = int(row["label"])
        samples.append(AllocineSample(text=text, label=label))
    return samples
