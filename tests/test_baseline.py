"""Tests de la baseline TF-IDF + LogReg sur un mini jeu synthétique FR."""

from src.baseline import (
    BaselineMetrics,
    build_baseline_pipeline,
    evaluate_baseline,
    train_baseline,
)


POS = [
    "Film magnifique, j'ai adoré du début à la fin.",
    "Une pépite, scénario brillant et acteurs excellents.",
    "Sublime, émouvant, à voir absolument.",
    "Très belle réalisation, je recommande chaudement.",
    "Un chef-d'œuvre, performance d'acteur incroyable.",
    "Vraiment génial, on en redemande.",
]

NEG = [
    "Film ennuyeux, je me suis endormi.",
    "Scénario incompréhensible et acteurs médiocres.",
    "Très mauvais, à éviter absolument.",
    "Une perte de temps, vraiment décevant.",
    "Réalisation paresseuse, dialogues plats.",
    "Affreux, je suis sorti avant la fin.",
]


def _dataset():
    X = POS + NEG
    y = [1] * len(POS) + [0] * len(NEG)
    return X, y


def test_pipeline_has_tfidf_and_logreg():
    pipe = build_baseline_pipeline()
    names = [n for n, _ in pipe.steps]
    assert names == ["tfidf", "logreg"]


def test_train_and_evaluate_perfect_on_training_set():
    X, y = _dataset()
    pipe = train_baseline(X, y)
    metrics = evaluate_baseline(pipe, X, y)
    assert isinstance(metrics, BaselineMetrics)
    assert metrics.n_samples == len(X)
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0


def test_train_rejects_length_mismatch():
    try:
        train_baseline(["a", "b"], [0])
    except ValueError as exc:
        assert "longueur" in str(exc)
    else:
        raise AssertionError("attendu ValueError")


def test_metrics_as_dict():
    X, y = _dataset()
    pipe = train_baseline(X, y)
    metrics = evaluate_baseline(pipe, X, y)
    d = metrics.as_dict()
    assert set(d) == {"accuracy", "f1_macro", "n_samples"}
    assert d["n_samples"] == len(X)


def test_baseline_generalises_to_held_out():
    X, y = _dataset()
    pipe = train_baseline(X, y)
    held_out = [
        "Magnifique film, je recommande vivement.",
        "Vraiment mauvais, à éviter.",
    ]
    expected = [1, 0]
    metrics = evaluate_baseline(pipe, held_out, expected)
    # tolérance faible : on attend au moins l'un des deux corrects
    assert metrics.accuracy >= 0.5
