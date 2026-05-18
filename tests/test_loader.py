"""Tests du loader Allociné — sans téléchargement réseau (loader injecté)."""

from src.data.loader import AllocineSample, load_allocine


def _fake_dataset(name, split):
    assert name == "tblard/allocine"
    assert split in {"train", "validation", "test"}
    return [
        {"review": "Film magnifique, jeu d'acteurs sublime.", "label": 1},
        {"review": "Ennuyeux et mal joué, j'ai perdu mon temps.", "label": 0},
        {"review": "Une vraie pépite, à voir.", "label": 1},
        {"review": "Scénario incompréhensible.", "label": 0},
    ]


def test_load_allocine_returns_samples():
    samples = load_allocine(split="train", subset=None, loader=_fake_dataset)
    assert len(samples) == 4
    assert all(isinstance(s, AllocineSample) for s in samples)
    assert {s.label for s in samples} == {0, 1}


def test_load_allocine_respects_subset():
    samples = load_allocine(split="train", subset=2, loader=_fake_dataset)
    assert len(samples) == 2
    assert samples[0].label == 1
    assert samples[1].label == 0


def test_load_allocine_labels_are_int():
    samples = load_allocine(split="test", subset=None, loader=_fake_dataset)
    for s in samples:
        assert isinstance(s.label, int)
        assert s.label in (0, 1)
