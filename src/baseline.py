"""Baseline TF-IDF + Régression logistique pour sentiment binaire FR.

Sert de référence honnête contre laquelle on comparera les approches
plus lourdes (embeddings, transformer fine-tuné).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    f1_macro: float
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "n_samples": self.n_samples,
        }


def _preprocess(texts: Iterable[str]) -> List[str]:
    return [clean_text(t, remove_accents=True) for t in texts]


def build_baseline_pipeline(
    *,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Construit le pipeline TF-IDF + LogReg, **sans** l'étape de nettoyage
    (le caller doit l'appliquer en amont via `_preprocess` ou équivalent).

    On garde un pipeline sklearn pur pour pouvoir utiliser `cross_val_score`
    et `GridSearchCV` plus tard sans surprise.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_baseline(
    texts: Sequence[str],
    labels: Sequence[int],
    **pipeline_kwargs,
) -> Pipeline:
    """Entraîne le pipeline sur (texts, labels). Renvoie le pipeline fitté."""
    if len(texts) != len(labels):
        raise ValueError("texts et labels doivent avoir la même longueur")

    pipe = build_baseline_pipeline(**pipeline_kwargs)
    pipe.fit(_preprocess(texts), list(labels))
    return pipe


def evaluate_baseline(
    pipeline: Pipeline,
    texts: Sequence[str],
    labels: Sequence[int],
) -> BaselineMetrics:
    """Évalue un pipeline fitté sur (texts, labels) et retourne accuracy + F1 macro."""
    preds = pipeline.predict(_preprocess(texts))
    return BaselineMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        f1_macro=float(f1_score(labels, preds, average="macro", zero_division=0)),
        n_samples=len(labels),
    )
